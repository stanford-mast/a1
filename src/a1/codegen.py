"""
Code generation module for agent code execution mode.

Provides Generate strategies that produce single code candidates.
Runtime orchestrates parallel generation, validation, and ranking.
"""

import json
import logging
import re
from typing import Any

from .code_utils import generate_nested_pydantic_classes, normalize_generated_code

logger = logging.getLogger(__name__)


def generate_tool_names(tools: list[Any]) -> dict[str, str]:
    """
    Generate LLM-friendly tool names (llm_a, llm_b, ..., llm_z, llm_aa, etc.).

    Maps original tool names to short variable names.

    Args:
        tools: List of Tool objects

    Returns:
        Dict mapping original tool name -> generated short name
    """
    name_map = {}
    counter = 0

    for tool in tools:
        if "llm" in tool.name.lower():
            # Generate name: a, b, ..., z, aa, ab, ...
            if counter < 26:
                short_name = chr(ord("a") + counter)
            else:
                # For aa, ab, ac, ..., ba, bb, ...
                first = chr(ord("a") + (counter - 26) // 26)
                second = chr(ord("a") + (counter - 26) % 26)
                short_name = first + second

            name_map[tool.name] = f"llm_{short_name}"
            counter += 1

    return name_map


# Example code showing realistic patterns
EXAMPLE_CODE = """
E1: immediate return
output = "The response to the user's task can be immediately returned"

E2: call single tool with handling of empty tool result
x = await tool_a(42)
output = f"It is {x}" if x else "No results found"

E3: multiple tool calls
await tool_c()
x = await tool_d(id=123)
y = await tool_e(p1=x)
output = f"Completed with result: {y}"

E4: control flow with loops and conditionals
await tool_c()
x = await tool_e(123)
results = [await tool_f(item=item) for item in x if item > 3]
output = f"Summary of {len(results)} results"

E5: LLM usage
data = await tool_a()
match = llm(f"most similar item in {data} to 'abraham'")
output = llm(f"summary of {match}")
"""

# E6: Multi-part
# data = await tool_a()
# raise Exception(f"I see that data is {data}, but I need to process it further.")

EXAMPLE_FUNCTION = """
E1: simple function
async def process_data(input_data: str) -> str:
    x = await tool_a(input_data)
    return f"Processed: {x}"

E2: function with multiple tools
async def complex_task(query: str, limit: int = 10) -> dict:
    results = await tool_b(q=query)
    filtered = [await tool_c(item) for item in results[:limit]]
    return {"count": len(filtered), "data": filtered}
"""

RULES = """
- Do NOT include comments in your code
- Output must be assigned to `output` with type `Output`
"""


# ============================================================================
# Generate Base Class and Implementations
# ============================================================================


class Generate:
    """
    Base class for code generation strategies.

    Generates a single code candidate.
    Does NOT handle validation, cost estimation, or ranking - that's Runtime's job.
    """

    async def generate(
        self,
        agent: Any,  # Agent
        task: str,
        return_function: bool = False,
        past_attempts: list[tuple[str, str]] | None = None,
        context: Any | None = None,  # Context
    ) -> tuple[str, str] | None:
        """
        Generate a single code candidate.

        Args:
            agent: Agent with tools available
            task: Task description (or function description for AOT)
            return_function: If True, generate a function definition. If False, generate code block.
            past_attempts: List of (candidate_code, validation_error) tuples for retry logic
            context: Optional Context object to maintain conversation across retries

        Returns:
            Tuple of (definition_code, generated_code) where:
            - definition_code: Imports, schemas, tool signatures (for LLM reference)
            - generated_code: The actual code to execute (what LLM generates)
            Returns None if generation fails
        """
        raise NotImplementedError


class BaseGenerate(Generate):
    """
    Base code generation implementation using an LLM.

    Generates ONE candidate at a time. Runtime handles parallel generation.

    Args:
        llm_tool: Tool to use for code generation (e.g., LLM("gpt-4.1-mini")). 
                  Defaults to LLM("gpt-4.1-mini") if not provided.
        timezone: Timezone for timestamp context (default: "UTC")
    """

    def __init__(
        self,
        llm_tool: Any | None = None,  # Tool
        timezone: str = "UTC",
    ):
        if llm_tool is None:
            from .llm import LLM
            llm_tool = LLM("gpt-4.1-mini")
        self.llm_tool = llm_tool
        self.timezone = timezone

    async def generate(
        self,
        agent: Any,
        task: str,
        return_function: bool = False,
        past_attempts: list[tuple[str, str]] | None = None,
        context: Any | None = None,  # Context
    ) -> tuple[str, str] | None:
        """
        Generate a single code candidate using LLM, returning (definition_code, generated_code) tuple.
        
        Uses Context to maintain conversation across retries:
        - First call: system message + user message (definition code) 
        - Retries: user message (error) → LLM tries to fix
        - Each attempt: assistant message added with generated code
        
        Args:
            agent: Agent with tools available
            task: Task description (or function description for AOT)
            return_function: If True, generate a function definition. If False, generate code block.
            past_attempts: List of (candidate_code, validation_error) tuples for retry logic
            context: Context object to maintain conversation across retries. Required.
        
        Returns:
            Tuple of (definition_code, generated_code) or None if generation fails
        """
        if context is None:
            raise ValueError("Context is required for code generation")

        # Build definition code
        definition_code = self._build_definition_code(agent, return_function=return_function, task=task)

        # Check if this is the first attempt by seeing if context has messages
        is_first_attempt = len(context.messages) == 0
        
        if is_first_attempt:
            # First attempt: Add system message with examples
            examples = EXAMPLE_FUNCTION if return_function else EXAMPLE_CODE
            system_msg = f"""You are an expert in writing Python code that calls tools.
<examples>
Here are examples of good code to generate. Use them as reference but never copy them directly.
{examples}
</examples>
<instructions>
Generate completion of the given code block to implement the TASK.
If an error is reported, fix the previously generated code accordingly.
</instructions>
"""
            context.system(system_msg)
            
            # Add user message with definition code
            prompt_parts = []
            prompt_parts.extend(self._build_timestamp())
            
            if return_function:
                # AOT mode: show function template
                prompt_parts.append("```python")
                prompt_parts.append(definition_code)
                # DO NOT close backticks - leave open for LLM to continue
            else:
                # JIT mode: show definitions + input values
                try:
                    input_values = json.loads(task)
                except:
                    input_values = {}

                prompt_parts.append("```python")
                prompt_parts.append(definition_code)
                prompt_parts.append("")
                for key, value in input_values.items():
                    if isinstance(value, str):
                        prompt_parts.append(f"{key} = {repr(value)}")
                    else:
                        prompt_parts.append(f"{key} = {value}")
                prompt_parts.append("")
                prompt_parts.append("# RESPOND WITH YOUR CODE HERE")
            
            user_prompt = "\n".join(prompt_parts)
            context.user(user_prompt)
            
        else:
            # Retry attempt: Add user message with error from last attempt
            if past_attempts:
                last_code, last_error = past_attempts[-1]
                error_prompt = f"Previous attempt failed with error:\n{last_error}\n\nPlease fix the code."
                context.user(error_prompt)

        # Log the current context
        logger.info("=" * 80)
        logger.info("CONTEXT MESSAGES FOR CODE GENERATION:")
        logger.info("=" * 80)
        for msg in context.messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            content_preview = content[:200] + "..." if len(content) > 200 else content
            role = msg.role if hasattr(msg, 'role') else 'unknown'
            logger.info(f"{role.upper()}: {content_preview}")
        logger.info("=" * 80)

        # Call LLM with the context
        try:
            logger.info(f"Generating code for task: {task[:100]}...")
            
            # Use the LLM tool with the context
            result = await self.llm_tool(content="", context=context)

            # Extract response content
            if isinstance(result, str):
                completion = result
            elif hasattr(result, "content"):
                completion = result.content
            elif isinstance(result, dict) and "content" in result:
                completion = result["content"]
            else:
                completion = str(result)

            # Extract code from response
            generated_code = self._extract_code_from_response(completion)
            if not generated_code:
                return None

            # For AOT mode, normalize indentation
            if return_function:
                generated_code = normalize_generated_code(generated_code)

            # Add assistant message with generated code to context
            context.assistant(f"```python\n{generated_code}\n```")

            # Log the generated code
            logger.info("=" * 80)
            logger.info("GENERATED CODE FROM LLM:")
            logger.info("=" * 80)
            logger.info(generated_code)
            logger.info("=" * 80)

            return (definition_code, generated_code)

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None

    def _build_definition_code(self, agent: Any, return_function: bool = False, task: str = "") -> str:
        """Build definition code showing tool signatures and agent schemas."""
        import copy
        lines = []

        # IMPORTANT: Keep imports in definition_code so LLM can reference them.
        # The generated code can also use imports - they will work in exec()

        lines.append("from pydantic import BaseModel, Field")
        lines.append("from typing import Optional, List, Dict, Any, Union, Literal, Type")
        lines.append("")

        # Add skill module imports
        # NOTE: In future, this can be made smarter to selectively load only relevant skills
        # based on the task at hand, rather than loading all skill modules
        if hasattr(agent, "skills") and agent.skills:
            skill_modules = set()
            for skill_or_skillset in agent.skills:
                if hasattr(skill_or_skillset, "skills"):
                    # It's a SkillSet
                    for skill in skill_or_skillset.skills:
                        skill_modules.update(skill.modules)
                else:
                    # It's a Skill
                    skill_modules.update(skill_or_skillset.modules)

            for module in sorted(skill_modules):
                lines.append(f"import {module}")
            if skill_modules:
                lines.append("")

        # Add Context stub so LLM can reference it
        lines.append("class Context:")
        lines.append('    """Context object for tracking message history."""')
        lines.append("    def __init__(self, messages=None): self.messages = messages or []")
        lines.append("")

        # Add get_context helper
        lines.append("def get_context(name: str = 'main'):")
        lines.append('    """Get or create a context by name. Creates if it does not exist."""')
        lines.append("    if name not in CTX: CTX[name] = Context()")
        lines.append("    return CTX[name]")
        lines.append("")

        # Show agent's INPUT schema for function parameters
        # For JIT, declare variables (values assigned in prompt); for AOT, show the class
        if hasattr(agent.input_schema, "__name__") and hasattr(agent.input_schema, "model_fields"):
            if not return_function:
                # JIT mode: Just declare variable names with type hints
                # Actual values will be assigned in the prompt
                for field_name, field_info in agent.input_schema.model_fields.items():
                    # Get type annotation as string
                    if hasattr(field_info.annotation, "__name__"):
                        field_type = field_info.annotation.__name__
                    else:
                        field_type = str(field_info.annotation)

                    # Just declare the variable with type hint (no assignment yet)
                    lines.append(f"{field_name}: {field_type}")
                lines.append("")
            else:
                # AOT mode: show the full input schema class
                # We sanitize schemas for the definition_code to avoid huge enums being inlined
                def_schema = None
                try:
                    if hasattr(agent.input_schema, 'model_json_schema'):
                        def_schema = agent.input_schema.model_json_schema()
                except Exception:
                    def_schema = None

                # If we have a JSON schema, de-enum any property with >100 options
                if def_schema:
                    from .schema_utils import de_enum_large_enums
                    def_schema = copy.deepcopy(def_schema)
                    de_enum_large_enums(def_schema, threshold=100)

                lines.append(f"class {agent.input_schema.__name__}(BaseModel):")
                for field_name, field_info in agent.input_schema.model_fields.items():
                    # Get type annotation as string
                    if hasattr(field_info.annotation, "__name__"):
                        field_type = field_info.annotation.__name__
                    else:
                        field_type = str(field_info.annotation)

                    # Get description if available
                    if hasattr(field_info, "description") and field_info.description:
                        desc = field_info.description
                    else:
                        desc = ""

                    if field_info.is_required():
                        lines.append(f'    {field_name}: {field_type} = Field(..., description="{desc}")')
                    else:
                        lines.append(f'    {field_name}: Optional[{field_type}] = Field(None, description="{desc}")')
                lines.append("")

        # Add skill content to definition code
        # NOTE: In future, this can be made smarter to selectively load only relevant skills
        # based on task context, rather than loading all skills
        if hasattr(agent, "skills") and agent.skills:
            lines.append("# ============================================================================")
            lines.append("# AVAILABLE SKILLS AND KNOWLEDGE")
            lines.append("# ============================================================================")
            lines.append("")

            def add_skills(skills_list):
                """Recursively add skill content from skills and skillsets."""
                for skill_or_skillset in skills_list:
                    if hasattr(skill_or_skillset, "skills"):
                        # It's a SkillSet - recurse into it
                        lines.append(f"# SkillSet: {skill_or_skillset.name}")
                        lines.append(f"# {skill_or_skillset.description}")
                        lines.append("")
                        add_skills(skill_or_skillset.skills)
                    else:
                        # It's a Skill - add its content
                        lines.append(f"# Skill: {skill_or_skillset.name}")
                        lines.append(f"# {skill_or_skillset.description}")
                        lines.append(f"# Modules: {', '.join(skill_or_skillset.modules)}")
                        lines.append("# Content:")
                        # Indent the skill content
                        for content_line in skill_or_skillset.content.split("\n"):
                            lines.append(f"# {content_line}")
                        lines.append("")

            add_skills(agent.skills)
            lines.append("# ============================================================================")
            lines.append("")

        # Add tool definitions - all available tools and their schemas (non-LLM tools first)
        for tool in agent.get_all_tools():
            # Include all tools except Done and LLM (we'll add LLM at the end)
            if "done" in tool.name.lower() or "llm" in tool.name.lower():
                continue
                lines.append('    """')
                lines.append(f"    {tool.description}")
                lines.append('    """')
                lines.append("    raise NotImplementedError")
                lines.append("")
                continue

            input_model_name = None
            output_model_name = None

            # Generate Pydantic model for input if schema exists
            if hasattr(tool, "input_schema") and tool.input_schema:
                try:
                    # First, extract and add any nested Pydantic models
                    self._add_nested_pydantic_models(tool.input_schema, lines)

                    if hasattr(tool.input_schema, "model_json_schema"):
                        schema = tool.input_schema.model_json_schema()
                    elif isinstance(tool.input_schema, dict):
                        schema = tool.input_schema
                    else:
                        schema = None

                    if schema and schema.get("properties"):
                        input_model_name = f"{tool.name.title().replace('_', '')}Input"
                        # Use recursive function to handle nested schemas
                        # This will generate nested classes for all nested objects
                        generate_nested_pydantic_classes(schema, input_model_name, lines)
                except Exception as e:
                    # Skip this tool's input schema but continue to show the function
                    logger.debug(f"Could not generate input schema for {tool.name}: {e}")

            # Generate Pydantic model for output if schema exists
            if hasattr(tool, "output_schema") and tool.output_schema:
                try:
                    # Check if this is a wrapped primitive type (created by @tool decorator)
                    # If output_schema.__name__ ends with "Output" and has only a "result" field,
                    # it's likely a wrapped primitive
                    is_wrapped_primitive = False
                    if (
                        hasattr(tool.output_schema, "__name__")
                        and "output" in tool.output_schema.__name__.lower()
                        and hasattr(tool.output_schema, "model_fields")
                    ):
                        fields = list(tool.output_schema.model_fields.keys())
                        if fields == ["result"]:
                            # This is a wrapped primitive - extract the actual type
                            result_field = tool.output_schema.model_fields["result"]
                            if hasattr(result_field.annotation, "__name__"):
                                output_model_name = result_field.annotation.__name__
                            else:
                                output_model_name = str(result_field.annotation)
                            is_wrapped_primitive = True

                    if not is_wrapped_primitive:
                        # First, extract and add any nested Pydantic models from output schema
                        self._add_nested_pydantic_models(tool.output_schema, lines)

                        if hasattr(tool.output_schema, "model_json_schema"):
                            schema = tool.output_schema.model_json_schema()
                            output_model_name = tool.output_schema.__name__
                        elif isinstance(tool.output_schema, dict):
                            schema = tool.output_schema
                        else:
                            schema = None

                        if schema and schema.get("properties"):
                            if not output_model_name:
                                output_model_name = f"{tool.name.title().replace('_', '')}Output"
                            # Use recursive function to handle nested schemas
                            generate_nested_pydantic_classes(schema, output_model_name, lines)
                except Exception as e:
                    # Skip this tool's output schema but continue to show the function
                    logger.debug(f"Could not generate output schema for {tool.name}: {e}")

            # Generate function signature with proper typing
            # For LLM tool or tools without schemas, use Any
            return_type = output_model_name if output_model_name else "Any"

            # Use **kwargs for ergonomic calling, with validation inside
            if input_model_name:
                lines.append(f"async def {tool.name}(**kwargs) -> {return_type}:")
                lines.append('    """')
                lines.append(f"    {tool.description}")
                lines.append('    """')
                lines.append(f"    input = {input_model_name}(**kwargs)  # validate kwargs against schema")
                lines.append(
                    f'    raise NotImplementedError(f"Tool {tool.name} called but not provided at runtime. This should be called via the executor environment.")'
                )
            elif input_model_name:
                # Fallback (shouldn't happen)
                lines.append(f"async def {tool.name}(input: {input_model_name}) -> {return_type}:")
                lines.append('    """')
                lines.append(f"    {tool.description}")
                lines.append('    """')
                lines.append(
                    f'    raise NotImplementedError(f"Tool {tool.name} called but not provided at runtime. This should be called via the executor environment.")'
                )
            else:
                # LLM-style tools take content string
                lines.append(f"async def {tool.name}(content: str) -> {return_type}:")
                lines.append('    """')
                lines.append(f"    {tool.description}")
                lines.append('    """')
                lines.append(
                    f'    raise NotImplementedError(f"Tool {tool.name} called but not provided at runtime. This should be called via the executor environment.")'
                )
            lines.append("")

        # Output schema definition comes AFTER tool definitions
        if hasattr(agent.output_schema, "__name__") and hasattr(agent.output_schema, "model_fields"):
            # Sanitize output schema JSON for large enums before including in definition code
            try:
                out_schema = agent.output_schema.model_json_schema()
            except Exception:
                out_schema = None

            if out_schema:
                from .schema_utils import de_enum_large_enums
                import copy
                out_schema = copy.deepcopy(out_schema)
                de_enum_large_enums(out_schema, threshold=100)

            lines.append(f"class {agent.output_schema.__name__}(BaseModel):")
            for field_name, field_info in agent.output_schema.model_fields.items():
                # Get type annotation as string
                if hasattr(field_info.annotation, "__name__"):
                    field_type = field_info.annotation.__name__
                else:
                    field_type = str(field_info.annotation)

                # Get description if available
                if hasattr(field_info, "description") and field_info.description:
                    desc = field_info.description
                else:
                    desc = ""

                if field_info.is_required():
                    lines.append(f'    {field_name}: {field_type} = Field(..., description="{desc}")')
                else:
                    lines.append(f'    {field_name}: Optional[{field_type}] = Field(None, description="{desc}")')
            lines.append("")

        # Add CTX dictionary - comes after all tools but before LLM tools
        lines.append("# Context dictionary - maps context names to Context objects")
        lines.append("CTX: Dict[str, Context] = {'main': Context()}")
        lines.append("")

        # Add LLM tools with short names (llm_a, llm_b, etc.)
        llm_tools = [t for t in agent.get_all_tools() if "llm" in t.name.lower()]
        if llm_tools:
            tool_name_map = generate_tool_names(llm_tools)
            for tool in llm_tools:
                short_name = tool_name_map[tool.name]
                lines.append(
                    f"async def {short_name}(content: Any, output_schema: Optional[Type[BaseModel]] = None) -> Union[str, BaseModel]:"
                )
                lines.append('    """')
                lines.append(f"    {tool.description}")
                lines.append('    """')
                lines.append(
                    f'    raise NotImplementedError(f"Tool {short_name} called but not provided at runtime. This should be called via the executor environment.")'
                )
                lines.append("")

        # RULES from global variable (only for JIT mode)
        if not return_function:
            # Get the actual output schema name for the rule
            output_schema_name = agent.output_schema.__name__ if hasattr(agent.output_schema, "__name__") else "Output"
            lines.append("# RULES:")
            lines.append("# - Do NOT include comments in your code")
            lines.append(f"# - Output must be assigned to `output` with type `{output_schema_name}`")

        # For JIT (code blocks), end with TASK and code section
        if not return_function:
            lines.append("")
            lines.append(f"# YOUR TASK IS: {agent.description}")
            lines.append("")
        else:
            # For AOT (functions), show the function signature and let LLM fill body
            # Build function signature from input schema
            if hasattr(agent.input_schema, "model_fields"):
                params = []
                for field_name, field_info in agent.input_schema.model_fields.items():
                    # Get type annotation as string
                    if hasattr(field_info.annotation, "__name__"):
                        field_type = field_info.annotation.__name__
                    else:
                        field_type = str(field_info.annotation)
                    params.append(f"{field_name}: {field_type}")

                param_str = ", ".join(params)
            else:
                param_str = "**kwargs"

            # Get output schema name
            output_name = agent.output_schema.__name__ if hasattr(agent.output_schema, "__name__") else "Output"

            # Generate template signature - show function sig, docstring, leave body open
            lines.append("")
            lines.append(f"async def {agent.name}({param_str}) -> {output_name}:")
            if agent.description:
                lines.append('    """')
                lines.append(f"    {agent.description}")
                lines.append('    """')
            lines.append(f"    # TASK: {task}")
            lines.append("    # RESPOND WITH ONLY YOUR CODE HERE")

        return "\n".join(lines)

    def _add_nested_pydantic_models(self, schema_class: Any, lines: list[str]) -> None:
        """
        Add all nested Pydantic model class definitions for a schema.

        Recursively extracts any nested BaseModel classes referenced by the schema
        and adds their definitions to the lines list.
        """
        from pydantic import BaseModel
        from typing import get_origin, get_args, Union

        if not (isinstance(schema_class, type) and issubclass(schema_class, BaseModel)):
            return

        if not hasattr(schema_class, "model_fields"):
            return

        visited = set()  # Prevent infinite recursion

        def extract_nested(model_class):
            if model_class in visited or not hasattr(model_class, "__name__"):
                return
            visited.add(model_class)

            if hasattr(model_class, "model_fields"):
                for field_name, field_info in model_class.model_fields.items():
                    field_type = field_info.annotation

                    # Handle Optional/Union types using get_origin and get_args
                    origin = get_origin(field_type)
                    if origin is Union or str(type(field_type).__name__) == 'UnionType':
                        # For Optional[X] or Union[X, ...], get the non-None type
                        args = get_args(field_type)
                        if args:
                            for arg in args:
                                if arg is not type(None):
                                    field_type = arg
                                    break

                    # Handle List[X] and other generic types
                    origin = get_origin(field_type)
                    if origin is not None:
                        args = get_args(field_type)
                        if args:
                            field_type = args[0]

                    # If it's a nested Pydantic model, generate its schema
                    if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                        # Recursively extract its nested models first
                        extract_nested(field_type)

                        # Check if already added
                        already_added = any(f"class {field_type.__name__}(BaseModel):" in line for line in lines)
                        if not already_added:
                            # Add this model class
                            nested_schema = field_type.model_json_schema()
                            generate_nested_pydantic_classes(nested_schema, field_type.__name__, lines)

        extract_nested(schema_class)

    def _json_type_to_python(self, json_type: str) -> str:
        """Convert JSON schema type to Python type hint."""
        type_map = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
        }
        return type_map.get(json_type, "Any")

    def _build_timestamp(self) -> list[str]:
        """Build timestamp string for context."""
        from datetime import datetime

        try:
            import zoneinfo

            tz = zoneinfo.ZoneInfo(self.timezone)
            now = datetime.now(tz)
            day_of_week = now.strftime("%A")
            date_part = now.strftime("%B %d, %Y")
            time_part = now.strftime("%I:%M:%S %p")
            tz_abbr = now.strftime("%Z")
            timestamp_str = f"{day_of_week}, {date_part} {time_part} {tz_abbr}"
            return [f"Now is {timestamp_str}.", ""]
        except Exception as e:
            logger.warning(f"Failed to get current time for timezone {self.timezone}: {e}")
            return []

    def _extract_code_from_response(self, response: str) -> str | None:
        """Extract Python code from markdown code blocks."""
        # Primary: fenced code blocks
        patterns = [r"```python\s*\n(.*?)\n```", r"```py\s*\n(.*?)\n```", r"```\s*\n(.*?)\n```"]
        for pattern in patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                extracted = matches[0].strip()
                if extracted:
                    return extracted

        # Fallback heuristics
        lines = [l.rstrip() for l in response.splitlines()]
        code_like = []
        in_code = False
        for l in lines:
            stripped = l.strip()
            if not stripped:
                continue
            # Start collecting when we see common Python starters
            if not in_code and (
                stripped.startswith(
                    (
                        "result",
                        "await ",
                        "async def ",
                        "def ",
                        "for ",
                        "if ",
                        "while ",
                        "import ",
                        "from ",
                        "class ",
                        "#",
                    )
                )
                or "await " in stripped
            ):
                in_code = True
            if in_code:
                code_like.append(l)
        fallback = "\n".join(code_like).strip()
        if fallback:
            # Remove trailing triple backticks if model left them open
            fallback = re.sub(r"```+$", "", fallback).strip()
            return fallback if fallback else None
        return None


__all__ = [
    "Generate",
    "BaseGenerate",
    "EXAMPLE_CODE",
    "EXAMPLE_FUNCTION",
    "RULES",
]
