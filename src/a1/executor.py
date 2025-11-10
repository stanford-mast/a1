"""
Code executors for running generated Python code.

Provides:
- Executor: Base class for code execution
- SimpleExecutor: Basic Python exec-based executor
"""

import ast
import asyncio
import logging
from typing import Any, Dict, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Maximum iterations for loops to prevent infinite loops
MAX_LOOP_ITERATIONS = 10000


@dataclass
class CodeOutput:
    """Result of code execution."""
    output: Any
    logs: str
    is_final_answer: bool = False
    error: Optional[str] = None


class Executor:
    """
    Base class for code executors.
    
    Executors are responsible for running generated Python code
    in a controlled environment.
    """
    
    async def execute(self, code: str, tools: Optional[List[Any]] = None) -> CodeOutput:
        """
        Execute Python code and return the result.
        
        Args:
            code: Python code to execute
            tools: Optional list of Tool objects to make available during execution
        
        Returns:
            CodeOutput with result, logs, and any errors
        """
        raise NotImplementedError


class _ToolWrapper:
    """Wraps a Tool to accept kwargs directly and auto-instantiate schemas."""
    
    def __init__(self, tool: Any):
        self.tool = tool
        self.name = tool.name
        self.description = tool.description
        self.__doc__ = tool.description
    
    async def __call__(self, *args, **kwargs):
        """Call the tool, auto-instantiating input schema if needed."""
        # If tool has an input schema, instantiate it with kwargs
        if hasattr(self.tool, 'input_schema') and self.tool.input_schema:
            if kwargs:
                # Instantiate the input schema with the kwargs
                input_obj = self.tool.input_schema(**kwargs)
                # Call the tool with the schema object
                return await self.tool(input_obj)
            elif args:
                # Single positional arg (schema object)
                return await self.tool(args[0])
            else:
                # No args - let tool handle it
                return await self.tool()
        else:
            # Tool without input schema (e.g., LLM tool)
            # Call directly with args/kwargs
            if args:
                return await self.tool(*args, **kwargs)
            elif kwargs:
                # For LLM tools, typically called with content kwarg
                return await self.tool(**kwargs)
            else:
                return await self.tool()


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
        additional_functions: Optional[Dict[str, Any]] = None,
        additional_imports: Optional[Dict[str, Any]] = None,
    ):
        self.state: Dict[str, Any] = {}
        self.additional_functions = additional_functions or {}
        self.additional_imports = additional_imports or {}
        
        # Initialize state with imports and functions
        self.state.update(self.additional_imports)
        self.state.update(self.additional_functions)
        
        # Track print output
        self.print_buffer: List[str] = []
    
    def _capture_print(self, *args, **kwargs):
        """Capture print() calls."""
        output = " ".join(str(arg) for arg in args)
        self.print_buffer.append(output)
        # Also print to actual stdout for debugging
        print(output, **kwargs)
    
    async def execute(self, code: str, tools: Optional[List[Any]] = None) -> CodeOutput:
        """
        Execute Python code asynchronously.
        
        Args:
            code: Python code to execute
            tools: Optional list of Tool objects to make available during execution
            
        Returns:
            CodeOutput with result, logs, and any errors
        """
        logger.info(f"\n{'='*80}\nEXECUTING CODE\n{'='*80}\n```python\n{code}\n```\n{'-'*80}")
        
        # Clear print buffer
        self.print_buffer = []
        
        # Create execution environment with tools available
        exec_env = self.state.copy()
        exec_env['print'] = self._capture_print
        
        # Add Context class and context management utilities
        from .context import Context
        
        def get_context(name: str = 'main'):
            """Get or create a context by name. Creates if it does not exist."""
            if name not in exec_env.get('CTX', {}):
                if 'CTX' not in exec_env:
                    exec_env['CTX'] = {}
                exec_env['CTX'][name] = Context()
            return exec_env['CTX'][name]
        
        exec_env['Context'] = Context
        exec_env['get_context'] = get_context
        # Initialize CTX dictionary if not already present
        if 'CTX' not in exec_env:
            exec_env['CTX'] = {'main': Context()}
        
        # Add tools to execution environment by name
        if tools:
            for tool in tools:
                # Wrap the tool to accept kwargs directly
                wrapped_tool = _ToolWrapper(tool)
                exec_env[tool.name] = wrapped_tool
                # Also add the input/output schemas as classes
                if hasattr(tool, 'input_schema') and tool.input_schema:
                    # Add the schema class itself so code can instantiate it if needed
                    exec_env[tool.input_schema.__name__] = tool.input_schema
                    # Also add all nested models from the input schema
                    from .code_utils import extract_nested_models
                    nested = extract_nested_models(tool.input_schema)
                    exec_env.update(nested)
                if hasattr(tool, 'output_schema') and tool.output_schema:
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
                    if 'llm' in lname:
                        exec_env['llm'] = wrapped_tool
                except Exception:
                    pass
        
        # Compile code - wrap in async function to support top-level await
        try:
            # Use code_utils to handle __future__ imports and wrapping
            from .code_utils import wrap_code_in_async_function
            wrapped_code = wrap_code_in_async_function(code)
            compiled = compile(wrapped_code, '<generated>', 'exec')
        except SyntaxError as e:
            return CodeOutput(
                output=None,
                logs="",
                is_final_answer=False,
                error=f"Compilation error: {e}"
            )
        
        # Execute in async context
        try:
            # Execute the wrapped function definition
            exec(compiled, exec_env, exec_env)
            
            # Get the wrapper function and call it
            wrapper_func = exec_env['__exec_wrapper']
            result_locals = await wrapper_func()
            
            # Check if the generated code defined an async function that should be called
            # This handles the case where the LLM generates a function instead of just code
            from .code_utils import detect_user_async_function, extract_execution_result, clean_execution_locals
            
            user_func_info = detect_user_async_function(result_locals)
            
            # If there's a user-defined async function AND no output was set, try to call it
            if user_func_info and 'output' not in result_locals:
                func_name, user_func = user_func_info
                logger.info(f"Calling user-defined async function: {func_name}")
                try:
                    # Try calling with no arguments first
                    result = await user_func()
                    # If the function returns something, use that as the result
                    if result is not None:
                        return CodeOutput(
                            output=result,
                            logs="\n".join(self.print_buffer),
                            is_final_answer=False
                        )
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
            
            logger.info(f"✓ CODE EXECUTED SUCCESSFULLY")
            logger.info(f"Output: {result}")
            logger.info(f"Logs:\n{logs}")
            logger.info(f"{'='*80}")
            
            return CodeOutput(
                output=result,
                logs=logs,
                is_final_answer=False
            )
            
        except Exception as e:
            logs = "\n".join(self.print_buffer)
            error_msg = f"{type(e).__name__}: {e}"
            logger.error(f"✗ CODE EXECUTION ERROR: {error_msg}")
            logger.error(f"Logs:\n{logs}")
            logger.error(f"{'='*80}")
            
            return CodeOutput(
                output=None,
                logs=logs,
                is_final_answer=False,
                error=error_msg
            )
    
    def send_tools(self, tools: Dict[str, Any]):
        """Update available tools."""
        self.additional_functions.update(tools)
        self.state.update(tools)
    
    def send_variables(self, variables: Dict[str, Any]):
        """Update state variables."""
        self.state.update(variables)


__all__ = [
    "Executor",
    "BaseExecutor",
    "CodeOutput",
]
