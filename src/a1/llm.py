"""
LLM tool implementation with multi-provider support.

Handles different function calling formats across providers:
- OpenAI: function_call/function_call_output with tool_calls array
- Anthropic: tool_use/tool_result with stop_reason
- Google: functionCall/functionResponse with parts
- Groq: Same as OpenAI (uses OpenAI-compatible API)

Provides unified interface via any-llm SDK.
"""

import json
import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict, SkipValidation

from .models import Tool

logger = logging.getLogger(__name__)

# Import any_llm at module level for easier mocking in tests
try:
    from any_llm import acompletion  # Use async version
except ImportError:
    acompletion = None  # Will be set in tests or when any_llm is installed


def no_context():
    """Create a throwaway context list that won't be used."""
    from .context import Context
    return Context()


class LLMInput(BaseModel):
    """Input for LLM tool - simplified for definition code."""
    content: str = Field(..., description="Input prompt or query")
    tools: Optional[SkipValidation[List[Tool]]] = Field(default=None, description="Tools available for function calling")
    context: Optional[SkipValidation[Any]] = Field(default=None, description="Context object for message history tracking")
    output_schema: Optional[SkipValidation[type[BaseModel]]] = Field(default=None, description="Optional schema to structure the output")


class LLMOutput(BaseModel):
    """Output from LLM tool - simplified for definition code."""
    content: str = Field(..., description="Text response from LLM")


def _tool_to_openai_schema(tool: Tool) -> Optional[Dict[str, Any]]:
    """
    Convert a1 Tool to OpenAI function schema.
    
    Returns None if the tool can't be serialized (e.g., LLM tools that have Tool objects in their schema).
    Uses cleaned tool name in the schema to avoid Harmony format tokens.
    """
    try:
        # Get JSON schema from Pydantic model
        schema = tool.input_schema.model_json_schema()
        
        # Clean the tool name to remove Harmony special tokens before sending to API
        clean_name = _clean_tool_name(tool.name)
        
        return {
            "type": "function",
            "function": {
                "name": clean_name,
                "description": tool.description,
                "parameters": schema
            }
        }
    except Exception as e:
        # Skip tools that can't be serialized (like LLM tools with Tool objects in their schema)
        logger.debug(f"Skipping tool {tool.name} - can't generate JSON schema: {e}")
        return None


def _clean_tool_name(name: str) -> str:
    """
    Clean tool name by removing Harmony format special tokens.
    
    The gpt-oss models use Harmony format with special tokens like <|channel|>commentary.
    Some providers may include these in the tool name, so we strip them out.
    """
    # Remove common Harmony special tokens that might appear in tool names
    # Examples: "done<|channel|>commentary" -> "done"
    special_tokens = [
        '<|channel|>commentary',
        '<|channel|>analysis', 
        '<|channel|>final',
        '<|constrain|>json',
        '<|call|>',
        '<|return|>',
        '<|end|>',
        '<|start|>',
        '<|message|>',
    ]
    
    cleaned = name
    for token in special_tokens:
        cleaned = cleaned.replace(token, '')
    
    # Remove any remaining <|...> patterns
    import re
    cleaned = re.sub(r'<\|[^|]+\|>', '', cleaned)
    
    return cleaned.strip()


def _extract_base_tool_name(name: str) -> str:
    """
    Extract the base tool name by removing all special tokens and markers.
    
    This is more aggressive than _clean_tool_name - it removes everything
    that looks like a Harmony token or special marker.
    
    Examples:
    - "done<|channel|>commentary" -> "done"
    - "calculator<|end|>" -> "calculator"
    - "llm_groq_openai_gpt_oss_20b" -> "llm_groq_openai_gpt_oss_20b"
    """
    import re
    # First do the standard cleaning
    cleaned = _clean_tool_name(name)
    # Remove any trailing underscores that might have been left
    cleaned = cleaned.strip('_')
    return cleaned


def _infer_provider(model: str) -> str:
    """Infer the provider from the model name."""
    if model.startswith("gpt"):
        return "openai"
    elif model.startswith("claude"):
        return "anthropic"
    elif model.startswith("gemini"):
        return "gemini"  # Changed from "google" to "gemini"
    elif model.startswith("llama"):
        return "groq"
    else:
        return "openai"  # Default to OpenAI


def LLM(model: str, input_schema: Optional[type[BaseModel]] = None, output_schema: Optional[type[BaseModel]] = None) -> Tool:
    """
    Create an LLM tool that can call language models with function calling support.
    
    Handles different function calling formats across providers:
    - OpenAI: tool_calls array with function objects
    - Anthropic: content blocks with tool_use type
    - Google: function_call in parts
    - Groq: OpenAI-compatible format
    
    Args:
        model: Model string with optional provider prefix (e.g., "gpt-4.1", "groq:llama-4", "claude-haiku-4-5")
        input_schema: Optional Pydantic model for structured input
        output_schema: Optional Pydantic model for structured output
    
    Returns:
        Tool that calls the LLM with function calling and history tracking
    """
    
    async def execute(
        content: str,
        tools: Optional[List[Tool]] = None,
        context: Optional[Any] = None,
        output_schema: Optional[type[BaseModel]] = None
    ):
        """
        Execute LLM call with optional function calling support.
        
        Returns typed output based on output_schema if provided, otherwise string.
        Supports JSON parsing from LLM responses to structured output types.
        
        If no tools provided, prepends "Respond with ONLY exactly what is requested: " to prompt.
        Also extracts large data structures (objects/lists) from content, labels them A, B, etc,
        and references them in the prompt for cleaner requests.
        
        Args:
            content: The prompt/query to send to the LLM
            tools: Optional list of tools available for function calling
            context: Optional Context object for message history tracking
            output_schema: Optional Pydantic model to parse LLM response into
        
        Returns:
            Instance of output_schema if provided, LLMOutput if tools called, otherwise string
        """
        from .context import Context
        import re
        
        # Determine the target output schema (passed to Done tool if needed)
        target_output_schema = output_schema
        
        # Use provided context or create throwaway
        if context is None:
            context = no_context()
        
        # Auto-add Done tool if tools provided but none are terminal
        if tools:
            has_terminal = any(t.is_terminal for t in tools)
            if not has_terminal:
                from .builtin_tools import Done
                tools = tools + [Done(output_schema=target_output_schema)]
        
        # Process content: extract large data structures and add "Respond with ONLY" prefix if no tools
        processed_content = content
        if not tools or len(tools) == 0:
            # Extract large objects/lists and label them
            data_parts = []
            label_map = {}
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            label_idx = 0
            
            # Find JSON-like structures: {...} and [...]
            obj_pattern = r'\{[^{}]*\}'  # Simple objects (no nesting for now)
            arr_pattern = r'\[[^\[\]]*\]'  # Simple arrays
            
            for pattern in [obj_pattern, arr_pattern]:
                for match in re.finditer(pattern, processed_content):
                    matched_str = match.group(0)
                    if len(matched_str) > 50:  # Only extract large structures
                        label = labels[label_idx % len(labels)]
                        label_map[matched_str] = label
                        label_idx += 1
            
            # Build the final content with data first
            if label_map:
                final_parts = []
                # Add labeled data first
                for data_str, label in label_map.items():
                    final_parts.append(f"{label} = {data_str}")
                final_parts.append("")
                # Add modified prompt with references instead of inline data
                modified_prompt = processed_content
                for data_str, label in label_map.items():
                    modified_prompt = modified_prompt.replace(data_str, label)
                final_parts.append("Respond with ONLY exactly what is requested:")
                final_parts.append(modified_prompt)
                processed_content = "\n".join(final_parts)
            else:
                # No large data structures, just add the prefix
                processed_content = f"Respond with ONLY exactly what is requested:\n{content}"
        
        # Add user message to context
        context.user(processed_content)
        
        # Prepare messages for API call
        messages = context.to_dict_list()
        
        # Convert tools to OpenAI function calling format
        api_tools = None
        if tools:
            schemas = [_tool_to_openai_schema(tool) for tool in tools]
            api_tools = [s for s in schemas if s is not None]
            if not api_tools:
                api_tools = None
        
        # Parse provider from model string if it contains ":"
        if ":" in model:
            provider, model_name = model.split(":", 1)
        else:
            provider = _infer_provider(model)
            model_name = model
        
        logger.info(f"Calling {provider}:{model_name} with {len(messages)} messages")
        
        # Call LLM via any-llm
        call_params = {
            "model": model_name,
            "provider": provider,
            "messages": messages,
        }
        
        if api_tools:
            call_params["tools"] = api_tools
            call_params["tool_choice"] = "auto"
        
        # Call LLM with retry logic for tool validation errors
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = await acompletion(**call_params)
                break  # Success
            except Exception as e:
                # Retry without tool_choice if this looks like a tool validation error
                error_msg = str(e)
                if "tool call validation failed" in error_msg and attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt+1}: Tool validation error, retrying")
                    call_params.pop("tool_choice", None)
                    continue
                raise
        
        # Extract response
        message = response.choices[0].message
        response_content = message.content or ""
        tool_calls = getattr(message, 'tool_calls', None)
        
        # Add assistant message to context
        if tool_calls:
            tool_call_dicts = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in tool_calls
            ]
            context.assistant(response_content, tool_calls=tool_call_dicts)
        else:
            context.assistant(response_content)
        
        # Execute tool calls if present
        tools_called_list = []
        if tool_calls and tools:
            logger.info(f"Executing {len(tool_calls)} tool calls")
            
            for tool_call in tool_calls:
                func_name = _clean_tool_name(tool_call.function.name)
                base_name = _extract_base_tool_name(func_name)
                func_args = json.loads(tool_call.function.arguments)
                
                # Find tool by name (try multiple strategies)
                tool = next((t for t in tools if 
                    t.name == func_name or 
                    t.name == base_name or
                    _clean_tool_name(t.name) == func_name or
                    _extract_base_tool_name(t.name) == base_name
                ), None)
                
                if tool:
                    try:
                        logger.info(f"Calling tool: {func_name}({func_args})")
                        result = await tool(**func_args)
                        
                        # Add result to context
                        context.tool(
                            content=json.dumps(result) if isinstance(result, dict) else str(result),
                            name=func_name,
                            tool_call_id=tool_call.id
                        )
                        logger.info(f"Tool {func_name} result: {result}")
                        tools_called_list.append(tool)
                        
                        # If terminal tool, return the result
                        if tool.is_terminal:
                            # If target_output_schema is set and result matches it, return as-is
                            if target_output_schema and isinstance(result, target_output_schema):
                                return result
                            # Try to convert result to target schema if needed
                            elif target_output_schema:
                                if isinstance(result, dict):
                                    return target_output_schema(**result)
                                else:
                                    field_name = list(target_output_schema.model_fields.keys())[0]
                                    return target_output_schema(**{field_name: result})
                            # Default: return as LLMOutput
                            else:
                                return LLMOutput(content=str(result), tools_called=[tool])
                    except Exception as e:
                        logger.error(f"Error executing {func_name}: {e}")
                        context.tool(
                            content=f"Error: {str(e)}",
                            name=func_name,
                            tool_call_id=tool_call.id
                        )
                else:
                    logger.warning(f"Tool {func_name} not found")
        
        # Return based on output_schema or tools called
        if tools_called_list:
            # If tools were called, return LLMOutput
            return LLMOutput(content=response_content, tools_called=tools_called_list)
        elif output_schema and output_schema != LLMOutput:
            # Try to parse response into output_schema
            try:
                # First try parsing as JSON
                parsed_data = json.loads(response_content)
                if isinstance(parsed_data, dict):
                    return output_schema(**parsed_data)
                else:
                    # If JSON is a primitive, wrap it in the schema
                    field_name = list(output_schema.model_fields.keys())[0]
                    return output_schema(**{field_name: parsed_data})
            except (json.JSONDecodeError, ValueError, TypeError) as e:
                # If JSON parsing fails, try wrapping the content directly
                logger.debug(f"Could not parse response as JSON, wrapping in schema: {e}")
                try:
                    field_name = list(output_schema.model_fields.keys())[0]
                    return output_schema(**{field_name: response_content})
                except Exception as e2:
                    # Fall back to returning string
                    logger.warning(f"Could not construct output schema: {e2}")
                    return response_content
        else:
            # Default: return string content
            return response_content
    
    return Tool(
        name=f"llm_{model.replace(':', '_').replace('-', '_').replace('/', '_')}",
        description=f"Call {model} language model with function calling support",
        input_schema=input_schema or LLMInput,
        output_schema=output_schema or LLMOutput,
        execute=execute,
        is_terminal=False
    )


__all__ = [
    "LLM",
    "LLMInput",
    "LLMOutput",
    "no_context",
]
