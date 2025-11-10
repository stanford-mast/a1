"""
Code generation module for agent code execution mode.

Provides Generate strategies that produce single code candidates.
Runtime orchestrates parallel generation, validation, and ranking.
"""

import json
import logging
import re
from typing import Optional, List, Tuple, Any, Dict, Union
from .code_utils import generate_nested_pydantic_classes

logger = logging.getLogger(__name__)


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
        past_attempts: Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[Tuple[str, str]]:
        """
        Generate a single code candidate.
        
        Args:
            agent: Agent with tools available
            task: Task description (or function description for AOT)
            return_function: If True, generate a function definition. If False, generate code block.
            past_attempts: List of (candidate_code, validation_error) tuples for retry logic
        
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
        llm_tool: Tool to use for code generation (e.g., LLM("gpt-4"))
        timezone: Timezone for timestamp context (default: "UTC")
    """
    
    def __init__(
        self,
        llm_tool: Any,  # Tool
        timezone: str = "UTC"
    ):
        self.llm_tool = llm_tool
        self.timezone = timezone
    
    async def generate(
        self,
        agent: Any,
        task: str,
        return_function: bool = False,
        past_attempts: Optional[List[Tuple[str, str]]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Generate a single code candidate using LLM, returning (definition_code, generated_code) tuple."""
        # Build definition code
        definition_code = self._build_definition_code(agent, return_function=return_function)
        
        # Build conversation history
        conversation = []
        
        # System message
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
        conversation.append({"role": "system", "content": system_msg})
        
        # Build prompt
        prompt_parts = []
        
        # Add timestamp
        prompt_parts.extend(self._build_timestamp())
        
        # If there are past attempts, show the error
        if past_attempts:
            last_code, last_error = past_attempts[-1]
            prompt_parts.append(f"Previous attempt failed with error:\n{last_error}")
            prompt_parts.append("")
            prompt_parts.append("Please fix the code:")
            prompt_parts.append("```python")
            prompt_parts.append(last_code)
            prompt_parts.append("```")
        else:
            # Initial generation - show definitions and ask for completion
            prompt_parts.append("```python")
            prompt_parts.append(definition_code)
            prompt_parts.append("```")
            prompt_parts.append("")
            if return_function:
                prompt_parts.append(f"TASK: {task}")
                prompt_parts.append("")
                prompt_parts.append("Generate ONLY the function body to complete the async def shown above.")
                prompt_parts.append("Do NOT repeat the function signature or redefine schemas/tools.")
                prompt_parts.append("Just write the implementation inside the function.")
            else:
                prompt_parts.append(f"TASK: {task}")
                prompt_parts.append("")
                prompt_parts.append("Generate code to accomplish the task using the tools defined above.")
            # Note: We deliberately leave it open - the LLM will generate code
        
        prompt = '\n'.join(prompt_parts)
        conversation.append({"role": "user", "content": prompt})
        
        # Call LLM
        try:
            logger.info(f"Generating code for task: {task[:100]}...")
            # The LLM tool expects content
            # We pass the conversation as a JSON string for code generation
            messages_str = json.dumps(conversation)
            result = await self.llm_tool(content=messages_str)
            
            # Extract response content - handle both string and LLMOutput
            if isinstance(result, str):
                completion = result
            elif hasattr(result, 'content'):
                completion = result.content
            elif isinstance(result, dict) and 'content' in result:
                completion = result['content']
            else:
                completion = str(result)
            
            # Extract code from response
            generated_code = self._extract_code_from_response(completion)
            if not generated_code:
                return None
            
            return (definition_code, generated_code)
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return None
    
    def _build_definition_code(self, agent: Any, return_function: bool = False) -> str:
        """Build definition code showing tool signatures and agent schemas."""
        lines = []
        
        # IMPORTANT: Keep imports in definition_code so LLM can reference them.
        # The generated code can also use imports - they will work in exec()
        
        lines.append("from pydantic import BaseModel, Field")
        lines.append("from typing import Optional, List, Dict, Any, Union, Literal")
        lines.append("")
        
        # Add Context stub so LLM can reference it
        lines.append("class Context:")
        lines.append('    """Context object for tracking message history."""')
        lines.append('    def __init__(self, messages=None): self.messages = messages or []')
        lines.append("")
        
        # Add get_context helper
        lines.append("def get_context(name: str = 'main'):")
        lines.append('    """Get or create a context by name. Creates if it does not exist."""')
        lines.append('    if name not in CTX: CTX[name] = Context()')
        lines.append('    return CTX[name]')
        lines.append("")
        
        # Show agent's INPUT schema for function parameters
        # For JIT, create actual variables; for AOT, show the class
        if hasattr(agent.input_schema, '__name__') and hasattr(agent.input_schema, 'model_fields'):
            if not return_function:
                # JIT mode: Create actual typed variables for input fields
                # These will be provided at runtime
                for field_name, field_info in agent.input_schema.model_fields.items():
                    # Get type annotation as string
                    if hasattr(field_info.annotation, '__name__'):
                        field_type = field_info.annotation.__name__
                    else:
                        field_type = str(field_info.annotation)
                    
                    # Create a variable assignment (will be replaced at runtime)
                    lines.append(f"{field_name}: {field_type} = None  # provided at runtime")
                lines.append("")
            else:
                # AOT mode: show the full input schema class
                lines.append(f"class {agent.input_schema.__name__}(BaseModel):")
                for field_name, field_info in agent.input_schema.model_fields.items():
                    # Get type annotation as string
                    if hasattr(field_info.annotation, '__name__'):
                        field_type = field_info.annotation.__name__
                    else:
                        field_type = str(field_info.annotation)
                        
                    # Get description if available
                    if hasattr(field_info, 'description') and field_info.description:
                        desc = field_info.description
                    else:
                        desc = ""
                    
                    if field_info.is_required():
                        lines.append(f'    {field_name}: {field_type} = Field(..., description="{desc}")')
                    else:
                        lines.append(f'    {field_name}: Optional[{field_type}] = Field(None, description="{desc}")')
                lines.append("")
        
        # Add tool definitions - all available tools and their schemas (non-LLM tools first)
        for tool in agent.get_all_tools():
            # Include all tools except Done and LLM (we'll add LLM at the end)
            if "done" in tool.name.lower() or "llm" in tool.name.lower():
                continue
                lines.append(f'    """')
                lines.append(f'    {tool.description}')
                lines.append(f'    """')
                lines.append(f'    raise NotImplementedError')
                lines.append('')
                continue
            
            input_model_name = None
            output_model_name = None
                
            # Generate Pydantic model for input if schema exists
            if hasattr(tool, 'input_schema') and tool.input_schema:
                try:
                    # First, extract and add any nested Pydantic models
                    self._add_nested_pydantic_models(tool.input_schema, lines)
                    
                    if hasattr(tool.input_schema, 'model_json_schema'):
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
            if hasattr(tool, 'output_schema') and tool.output_schema:
                try:
                    # First, extract and add any nested Pydantic models from output schema
                    self._add_nested_pydantic_models(tool.output_schema, lines)
                    
                    if hasattr(tool.output_schema, 'model_json_schema'):
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
                lines.append(f'    """')
                lines.append(f'    {tool.description}')
                lines.append(f'    """')
                lines.append(f'    input = {input_model_name}(**kwargs)  # validate kwargs against schema')
                lines.append(f'    raise NotImplementedError(f"Tool {tool.name} called but not provided at runtime. This should be called via the executor environment.")')
            elif input_model_name:
                # Fallback (shouldn't happen)
                lines.append(f"async def {tool.name}(input: {input_model_name}) -> {return_type}:")
                lines.append(f'    """')
                lines.append(f'    {tool.description}')
                lines.append(f'    """')
                lines.append(f'    raise NotImplementedError(f"Tool {tool.name} called but not provided at runtime. This should be called via the executor environment.")')
            else:
                # LLM-style tools take content string
                lines.append(f"async def {tool.name}(content: str) -> {return_type}:")
                lines.append(f'    """')
                lines.append(f'    {tool.description}')
                lines.append(f'    """')
                lines.append(f'    raise NotImplementedError(f"Tool {tool.name} called but not provided at runtime. This should be called via the executor environment.")')
            lines.append('')
        
        # Output schema definition comes AFTER tool definitions
        if hasattr(agent.output_schema, '__name__') and hasattr(agent.output_schema, 'model_fields'):
            lines.append(f"class {agent.output_schema.__name__}(BaseModel):")
            for field_name, field_info in agent.output_schema.model_fields.items():
                # Get type annotation as string
                if hasattr(field_info.annotation, '__name__'):
                    field_type = field_info.annotation.__name__
                else:
                    field_type = str(field_info.annotation)
                    
                # Get description if available
                if hasattr(field_info, 'description') and field_info.description:
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
        
        # Add LLM tools (simplified signatures)
        for tool in agent.get_all_tools():
            if "llm" in tool.name.lower():
                lines.append(f"async def {tool.name}(content: str) -> str:")
                lines.append(f'    """')
                lines.append(f'    {tool.description}')
                lines.append(f'    """')
                lines.append(f'    raise NotImplementedError(f"Tool {tool.name} called but not provided at runtime. This should be called via the executor environment.")')
                lines.append('')
        
        # RULES from global variable
        lines.append("# RULES:")
        for rule in RULES.strip().split('\n'):
            rule = rule.strip()
            if rule.startswith('- '):
                lines.append(f"# {rule}")
            elif rule:
                lines.append(f"# - {rule}")
        
        # For JIT (code blocks), end with TASK and code section
        if not return_function:
            lines.append("")
            lines.append(f"# TASK: {agent.description}")
            lines.append("")
            lines.append("# YOUR CODE HERE")
        else:
            # For AOT (functions), show the function signature and let LLM fill body
            # Build function signature from input schema
            if hasattr(agent.input_schema, 'model_fields'):
                params = []
                for field_name, field_info in agent.input_schema.model_fields.items():
                    # Get type annotation as string
                    if hasattr(field_info.annotation, '__name__'):
                        field_type = field_info.annotation.__name__
                    else:
                        field_type = str(field_info.annotation)
                    params.append(f"{field_name}: {field_type}")
                
                param_str = ", ".join(params)
            else:
                param_str = "**kwargs"
            
            # Get output schema name
            output_name = agent.output_schema.__name__ if hasattr(agent.output_schema, '__name__') else "Output"
            
            # Generate template signature - show function sig, docstring, but leave body open
            lines.append("#")
            lines.append(f"async def {agent.name}({param_str}) -> {output_name}:")
            if agent.description:
                lines.append(f'    """')
                lines.append(f'    {agent.description}')
                lines.append(f'    """')
        
        return '\n'.join(lines)
    
    def _add_nested_pydantic_models(self, schema_class: Any, lines: List[str]) -> None:
        """
        Add all nested Pydantic model class definitions for a schema.
        
        Recursively extracts any nested BaseModel classes referenced by the schema
        and adds their definitions to the lines list.
        """
        from pydantic import BaseModel
        
        if not (isinstance(schema_class, type) and issubclass(schema_class, BaseModel)):
            return
        
        if not hasattr(schema_class, 'model_fields'):
            return
        
        visited = set()  # Prevent infinite recursion
        
        def extract_nested(model_class):
            if model_class in visited or not hasattr(model_class, '__name__'):
                return
            visited.add(model_class)
            
            if hasattr(model_class, 'model_fields'):
                for field_name, field_info in model_class.model_fields.items():
                    field_type = field_info.annotation
                    
                    # Handle Optional/Union types
                    if hasattr(field_type, '__origin__'):
                        if hasattr(field_type, '__args__'):
                            # For Optional[X] or Union[X, ...], get the non-None type
                            for arg in field_type.__args__:
                                if arg is not type(None):
                                    field_type = arg
                                    break
                    
                    # Handle List[X]
                    if hasattr(field_type, '__origin__') and hasattr(field_type, '__args__'):
                        field_type = field_type.__args__[0]
                    
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
    
    def _build_timestamp(self) -> List[str]:
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
    
    def _extract_code_from_response(self, response: str) -> Optional[str]:
        """Extract Python code from markdown code blocks."""
        # Primary: fenced code blocks
        patterns = [
            r'```python\s*\n(.*?)\n```',
            r'```py\s*\n(.*?)\n```',
            r'```\s*\n(.*?)\n```'
        ]
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
                stripped.startswith(('result', 'await ', 'async def ', 'def ', 'for ', 'if ', 'while ', 'import ', 'from ', 'class ', '#'))
                or 'await ' in stripped
            ):
                in_code = True
            if in_code:
                code_like.append(l)
        fallback = '\n'.join(code_like).strip()
        if fallback:
            # Remove trailing triple backticks if model left them open
            fallback = re.sub(r'```+$', '', fallback).strip()
            return fallback if fallback else None
        return None


__all__ = [
    "Generate",
    "BaseGenerate",
    "EXAMPLE_CODE",
    "EXAMPLE_FUNCTION",
    "RULES",
]
