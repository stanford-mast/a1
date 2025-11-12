"""
Code executors for running generated Python code.

Provides:
- Executor: Base class for code execution
- SimpleExecutor: Basic Python exec-based executor
"""

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Maximum iterations for loops to prevent infinite loops
MAX_LOOP_ITERATIONS = 10000


@dataclass
class CodeOutput:
    """Result of code execution."""

    output: Any
    logs: str
    is_final_answer: bool = False
    error: str | None = None


class Executor:
    """
    Base class for code executors.

    Executors are responsible for running generated Python code
    in a controlled environment.
    """

    async def execute(self, code: str, tools: list[Any] | None = None) -> CodeOutput:
        """
        Execute Python code and return the result.

        Args:
            code: Python code to execute
            tools: Optional list of Tool objects to make available during execution

        Returns:
            CodeOutput with result, logs, and any errors
        """
        raise NotImplementedError


class ToolWrapper:
    """
    Wrapper for calling tools in the execution environment.

    Handles auto-extraction of single-field Pydantic model outputs,
    so generated code doesn't need hasattr checks.

    EXCEPTION: LLM tools are NOT auto-extracted because they return structured output schemas
    that should be preserved (e.g., Output(result=value) should stay as Output instance).
    """

    def __init__(self, tool):
        self.tool = tool

    # Expose Tool attributes for compatibility when ToolWrapper is passed around
    @property
    def name(self):
        return self.tool.name

    @property
    def description(self):
        return self.tool.description

    @property
    def is_terminal(self):
        return self.tool.is_terminal

    @property
    def input_schema(self):
        return self.tool.input_schema

    @property
    def output_schema(self):
        return self.tool.output_schema

    def _convert_to_llm_content(self, content: Any) -> str:
        """
        Convert any input type to a string suitable for LLM content.

        - BaseModel: Convert to JSON via model_dump_json()
        - dict: Convert to JSON string
        - Other types: Convert to str()
        """
        import json

        from pydantic import BaseModel

        if isinstance(content, BaseModel):
            return content.model_dump_json()
        elif isinstance(content, dict):
            return json.dumps(content)
        else:
            return str(content)

    async def __call__(self, *args, **kwargs):
        # Check if this is an LLM tool (special handling needed)
        is_llm_tool = "llm" in self.tool.name.lower()

        # LLM tools: Handle positional + keyword args flexibly
        # First arg can be Any type - convert to string appropriately
        if is_llm_tool:
            # If first positional arg is provided, treat it as 'content'
            if args and len(args) > 0:
                content = args[0]
                # Convert content to string appropriately
                content_str = self._convert_to_llm_content(content)
                # First positional arg becomes 'content', merge with any other kwargs
                kwargs = {"content": content_str, **kwargs}
                result = await self.tool(**kwargs)
            elif kwargs:
                # If 'content' kwarg exists, convert it too
                if "content" in kwargs:
                    kwargs["content"] = self._convert_to_llm_content(kwargs["content"])
                result = await self.tool(**kwargs)
            else:
                result = await self.tool()
        # Regular tools: Use input schema validation
        elif hasattr(self.tool, "input_schema") and self.tool.input_schema:
            if kwargs:
                # Instantiate the input schema with the kwargs
                input_obj = self.tool.input_schema(**kwargs)
                # Call the tool with the schema object
                result = await self.tool(input_obj)
            elif args:
                # Single positional arg (schema object)
                result = await self.tool(args[0])
            else:
                # No args - let tool handle it
                result = await self.tool()
        else:
            # Tool without input schema
            # Call directly with args/kwargs
            if args:
                result = await self.tool(*args, **kwargs)
            elif kwargs:
                result = await self.tool(**kwargs)
            else:
                result = await self.tool()

        # Auto-extract single-field Pydantic model outputs
        # EXCEPTION: Don't extract for:
        # 1. LLM tools (they return structured schemas)
        # 2. Done tool (it returns the output schema that should be preserved)
        # 3. Any terminal tool (they return final output schemas)
        #
        # If output is a Pydantic model with only a 'result' field AND it's not a terminal/structured tool,
        # return the value directly so generated code doesn't need hasattr checks.
        from pydantic import BaseModel

        # Check if this is a special tool that shouldn't be auto-extracted
        is_llm_tool = "llm" in self.tool.name.lower()
        is_done_tool = self.tool.name.lower() == "done"
        is_terminal_tool = self.tool.is_terminal

        should_not_extract = is_llm_tool or is_done_tool or is_terminal_tool

        # Log LLM tool results for debugging
        if is_llm_tool:
            logger.info(f"🤖 LLM TOOL RESULT ({self.tool.name}):")
            logger.info(f"   Type: {type(result)}")
            logger.info(f"   Value: {result}")
            if isinstance(result, BaseModel):
                logger.info(f"   Fields: {list(result.__class__.model_fields.keys())}")
                for field_name in result.__class__.model_fields.keys():
                    logger.info(f"   {field_name} = {getattr(result, field_name)}")

        if not should_not_extract and isinstance(result, BaseModel):
            fields = list(result.__class__.model_fields.keys())
            if fields == ["result"]:
                # Single 'result' field - extract it automatically
                return result.result

        return result


class BaseExecutor(Executor):
    """
    Base executor implementation using Python's exec() with async support.

    Maintains state between executions and provides access to custom functions.
    Captures print outputs and supports async/await natively.

    Makes tool calls ergonomic by allowing kwargs directly instead of schema instantiation:
        calculator(a=1, b=2, operation="add")  # instead of calculator(CalculatorInput(...))

    WARNING: This executor does NOT provide sandboxing. Only use with
    trusted code generation.

    Args:
        additional_functions: Dictionary of functions to make available to code
        additional_imports: Dictionary of module imports to make available
    """

    def __init__(
        self,
        additional_functions: dict[str, Any] | None = None,
        additional_imports: dict[str, Any] | None = None,
    ):
        self.state: dict[str, Any] = {}
        self.additional_functions = additional_functions or {}
        self.additional_imports = additional_imports or {}

        # Initialize state with imports and functions
        self.state.update(self.additional_imports)
        self.state.update(self.additional_functions)

        # Track print output
        self.print_buffer: list[str] = []

    def _capture_print(self, *args, **kwargs):
        """Capture print() calls."""
        output = " ".join(str(arg) for arg in args)
        self.print_buffer.append(output)
        # Also print to actual stdout for debugging
        print(output, **kwargs)

    async def execute(self, code: str, tools: list[Any] | None = None) -> CodeOutput:
        """
        Execute Python code asynchronously.

        Args:
            code: Python code to execute
            tools: Optional list of Tool objects to make available during execution

        Returns:
            CodeOutput with result, logs, and any errors
        """
        logger.info(f"\n{'=' * 80}\nEXECUTING CODE\n{'=' * 80}\n```python\n{code}\n```\n{'-' * 80}")

        # Clear print buffer
        self.print_buffer = []

        # Create execution environment with tools available
        exec_env = self.state.copy()
        exec_env["print"] = self._capture_print

        # Add standard library imports that generated code commonly needs
        import json
        import re

        exec_env["json"] = json
        exec_env["re"] = re

        # Add Context class and context management utilities
        from .context import Context
        from .runtime import get_context as runtime_get_context

        # Use the RUNTIME's get_context so contexts are shared
        # between test code and generated code execution
        exec_env["Context"] = Context
        exec_env["get_context"] = runtime_get_context
        # Initialize CTX dictionary if not already present
        if "CTX" not in exec_env:
            exec_env["CTX"] = {"main": Context()}

        # Add tools to execution environment by name
        if tools:
            for tool in tools:
                # Wrap the tool to accept kwargs directly
                wrapped_tool = ToolWrapper(tool)
                exec_env[tool.name] = wrapped_tool
                # Also add the input/output schemas as classes
                if hasattr(tool, "input_schema") and tool.input_schema:
                    # Add the schema class itself so code can instantiate it if needed
                    exec_env[tool.input_schema.__name__] = tool.input_schema
                    # Also add all nested models from the input schema
                    from .code_utils import extract_nested_models

                    nested = extract_nested_models(tool.input_schema)
                    exec_env.update(nested)
                if hasattr(tool, "output_schema") and tool.output_schema:
                    exec_env[tool.output_schema.__name__] = tool.output_schema
                    # Also add all nested models from the output schema
                    from .code_utils import extract_nested_models

                    nested = extract_nested_models(tool.output_schema)
                    exec_env.update(nested)
                # Provide common short aliases for tools to match model expectations.
                # e.g., models may call the LLM as 'llm' even though the tool name is
                # 'llm_groq_openai_gpt_oss_20b'. Add an 'llm' alias for any tool
                # whose name contains 'llm'. This keeps JIT execution running the
                # generated code (which is intended to run standalone) without
                # requiring the definition code to have been executed.
                try:
                    lname = tool.name.lower()
                    if "llm" in lname:
                        exec_env["llm"] = wrapped_tool
                except Exception:
                    pass

        # Add generated LLM tool name aliases (llm_a, llm_b, etc.)
        # These match the names generated by codegen.generate_tool_names()
        if tools:
            llm_tools = [t for t in tools if "llm" in t.name.lower()]
            counter = 0
            for tool in llm_tools:
                wrapped_tool = exec_env.get(tool.name)
                if wrapped_tool:
                    if counter < 26:
                        short_name = chr(ord("a") + counter)
                    else:
                        first = chr(ord("a") + (counter - 26) // 26)
                        second = chr(ord("a") + (counter - 26) % 26)
                        short_name = first + second
                    exec_env[f"llm_{short_name}"] = wrapped_tool
                    counter += 1

        # Compile code - wrap in async function to support top-level await
        try:
            # Log the generated code BEFORE execution
            logger.info(f"{'=' * 80}")
            logger.info("GENERATED CODE:")
            logger.info(f"{'=' * 80}")
            for i, line in enumerate(code.split("\n"), 1):
                logger.info(f"{i:3d} | {line}")
            logger.info(f"{'=' * 80}")

            # Use code_utils to handle __future__ imports and wrapping
            from .code_utils import wrap_code_in_async_function

            wrapped_code = wrap_code_in_async_function(code)
            compiled = compile(wrapped_code, "<generated>", "exec")
        except SyntaxError as e:
            return CodeOutput(output=None, logs="", is_final_answer=False, error=f"Compilation error: {e}")

        # Execute in async context
        try:
            # Execute the wrapped function definition
            exec(compiled, exec_env, exec_env)

            # Get the wrapper function and call it
            wrapper_func = exec_env["__exec_wrapper"]
            result_locals = await wrapper_func()

            # Check if the generated code defined an async function that should be called
            # This handles the case where the LLM generates a function instead of just code
            from .code_utils import clean_execution_locals, detect_user_async_function, extract_execution_result

            user_func_info = detect_user_async_function(result_locals)

            # If there's a user-defined async function AND no output was set, try to call it
            if user_func_info and "output" not in result_locals:
                func_name, user_func = user_func_info
                logger.info(f"Calling user-defined async function: {func_name}")
                try:
                    # Try calling with no arguments first
                    result = await user_func()
                    # If the function returns something, use that as the result
                    if result is not None:
                        return CodeOutput(output=result, logs="\n".join(self.print_buffer), is_final_answer=False)
                except TypeError as e:
                    # Function requires arguments - we can't call it without knowing what they are
                    # In this case, we'll fall through and try to find 'output' or 'result'
                    logger.debug(f"Cannot call {func_name} with no arguments: {e}")
                    pass

            # Update state with results (excluding the wrapper function itself and tools)
            cleaned_locals = clean_execution_locals(result_locals, tools=tools, agent_schemas=None)
            self.state.update(cleaned_locals)

            # Find the result - prefer variables named 'output', then 'result', then last variable
            result = extract_execution_result(result_locals)

            logs = "\n".join(self.print_buffer)

            logger.info("✓ CODE EXECUTED SUCCESSFULLY")
            logger.info(f"Output: {result}")
            logger.info(f"Logs:\n{logs}")
            logger.info(f"{'=' * 80}")

            return CodeOutput(output=result, logs=logs, is_final_answer=False)

        except Exception as e:
            logs = "\n".join(self.print_buffer)
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"✗ CODE EXECUTION ERROR: {error_msg}")
            logger.error(f"Logs:\n{logs}")
            logger.error(f"{'=' * 80}")

            return CodeOutput(output=None, logs=logs, is_final_answer=False, error=error_msg)

    def send_tools(self, tools: dict[str, Any]):
        """Update available tools."""
        self.additional_functions.update(tools)
        self.state.update(tools)

    def send_variables(self, variables: dict[str, Any]):
        """Update state variables."""
        self.state.update(variables)


__all__ = [
    "Executor",
    "BaseExecutor",
    "CodeOutput",
]
