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
from typing import Any

from pydantic import BaseModel, Field, SkipValidation

from .models import Tool
from .em import EM, _stringify_item, _pseudo_embed
from .schema_utils import (
    reduce_large_enums,
    clean_schema_for_openai,
    prepare_response_format,
    reduce_large_enums_in_tool_schemas
)
import math

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
    tools: SkipValidation[list[Tool]] | None = Field(default=None, description="Tools available for function calling")
    context: SkipValidation[Any] | None = Field(default=None, description="Context object for message history tracking")
    output_schema: SkipValidation[type[BaseModel]] | None = Field(
        default=None, description="Optional schema to structure the output"
    )


class LLMOutput(BaseModel):
    """Output from LLM tool - simplified for definition code."""

    content: str = Field(..., description="Text response from LLM")


def _tool_to_openai_schema(tool: Tool) -> dict[str, Any] | None:
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
            "function": {"name": clean_name, "description": tool.description, "parameters": schema},
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
        "<|channel|>commentary",
        "<|channel|>analysis",
        "<|channel|>final",
        "<|constrain|>json",
        "<|call|>",
        "<|return|>",
        "<|end|>",
        "<|start|>",
        "<|message|>",
    ]

    cleaned = name
    for token in special_tokens:
        cleaned = cleaned.replace(token, "")

    # Remove any remaining <|...> patterns
    import re

    cleaned = re.sub(r"<\|[^|]+\|>", "", cleaned)

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

    # First do the standard cleaning
    cleaned = _clean_tool_name(name)
    # Remove any trailing underscores that might have been left
    cleaned = cleaned.strip("_")
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


def LLM(
    model: str,
    input_schema: type[BaseModel] | None = None,
    output_schema: type[BaseModel] | None = None,
    retry_strategy: Any | None = None,  # Will be RetryStrategy from models
) -> Tool:
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
        retry_strategy: Optional RetryStrategy for retry logic when output_schema validation fails
                       (default: RetryStrategy(max_iterations=3, num_candidates=3))

    Returns:
        Tool that calls the LLM with function calling and history tracking
    """
    # Import here to avoid circular dependency
    from .models import RetryStrategy

    # Default to 3 parallel candidates with 3 retries each
    if retry_strategy is None:
        retry_strategy = RetryStrategy(max_iterations=3, num_candidates=3)

    # Apply default output schema and capture it before the execute parameter shadows it
    tool_output_schema = output_schema or LLMOutput

    async def execute(
        content: str,
        tools: list[Tool] | None = None,
        context: Any | None = None,
        output_schema: type[BaseModel] | None = None,
    ):
        """
        Execute LLM call with optional function calling support.

        SCHEMA HANDLING OVERVIEW:
        ========================

        INPUT SCHEMAS (what the LLM tool receives):
        - content: str - the prompt
        - tools: List[Tool] - tools that might have input_schema (can be primitive or complex Pydantic)
        - context: Context - message history
        - output_schema: Optional[type[BaseModel]] - target output type for final result

        TOOL SCHEMAS (what tools define):
        - tool.input_schema: Pydantic model for validating tool arguments
        - tool.output_schema: Pydantic model for tool return values (ALWAYS Pydantic wrapped)

        TOOL EXECUTION BEHAVIOR:
        1. Non-terminal tools (calculator, search, etc):
           - Input: kwargs validated against input_schema
           - Output: ALWAYS a Pydantic model instance (auto-wrapped by Tool.__call__)
           - Usage: Result added to context, loop continues for next LLM turn

        2. Terminal tools (Done):
           - Input: kwargs validated against input_schema
           - Output: ALWAYS a Pydantic model instance matching output_schema
           - Usage: Result returned directly as final output

        LLM TOOL RETURN VALUE:
        - If terminal tool called: return its output (Pydantic model matching output_schema)
        - If no terminal tool but tools called: return response_content (string)
        - If output_schema provided and no tools called: parse response_content into output_schema
        - Otherwise: return response_content (string)

        GENERATED CODE EXPECTATIONS:
        - When generated code calls LLM: expects either string OR output_schema instance
        - When generated code calls non-LLM tool: auto-extraction happens in _ToolWrapper

        EXECUTOR AUTO-EXTRACTION (_ToolWrapper):
        - If tool result is Pydantic model with ONLY 'result' field: extract the value
        - Example: CalculatorOutput(result=42) -> 42
        - Reason: Generated code written as `output = Output(sum=int(result))`
                  not `output = Output(sum=int(result.result))`

        Returns typed output based on output_schema if provided, otherwise string.
        Supports JSON parsing from LLM responses to structured output types.
        """
        import re

        # Determine the target output schema (passed to Done tool if needed)
        # Use the output_schema parameter if provided, otherwise fall back to the tool's declared output_schema
        target_output_schema = output_schema if output_schema is not None else tool_output_schema
        logger.info(
            f"LLM execute: output_schema param = {output_schema.__name__ if output_schema else 'None'}, "
            f"tool_output_schema = {tool_output_schema.__name__ if tool_output_schema else 'None'}, "
            f"target = {target_output_schema.__name__ if target_output_schema else 'None'}"
        )

        # Use provided context or create new tracked context
        if context is None:
            from .runtime import new_context

            context = new_context("intermediate")

        # Auto-add Done tool if tools provided but none are terminal
        if tools:
            # Normalize tools to Tool objects. Tools may be passed as:
            # - a1.models.Tool instances
            # - executor.ToolWrapper instances (wrapping a Tool)
            # - dicts produced by Pydantic model_dump (legacy/serialization)
            from pydantic import create_model

            from .models import Tool as ToolClass

            normalized = []
            for t in tools:
                # Already a Tool
                if isinstance(t, ToolClass):
                    normalized.append(t)
                    continue
                # ToolWrapper from executor - unwrap
                if hasattr(t, "tool") and isinstance(getattr(t, "tool"), ToolClass):
                    normalized.append(getattr(t, "tool"))
                    continue
                # Dict-like (from model_dump) - reconstruct minimal Tool
                if isinstance(t, dict):
                    name = t.get("name") or t.get("tool") or "unknown_tool"
                    desc = t.get("description", "")
                    # Create minimal input/output schemas so we can produce JSON schema
                    InputModel = create_model(f"{name}_Input")
                    OutputModel = create_model(f"{name}_Output", result=(Any, ...))
                    try:
                        reconstructed = ToolClass(
                            name=name,
                            description=desc,
                            input_schema=InputModel,
                            output_schema=OutputModel,
                            execute=(lambda **k: None),
                            is_terminal=bool(t.get("is_terminal", False)),
                        )
                        normalized.append(reconstructed)
                        continue
                    except Exception:
                        # Fallthrough - skip if reconstruction fails
                        continue
                # Unknown type - skip
            tools = normalized

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
            obj_pattern = r"\{[^{}]*\}"  # Simple objects (no nesting for now)
            arr_pattern = r"\[[^\[\]]*\]"  # Simple arrays

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

        # Add ORIGINAL user message to context (not the processed version with system prompt)
        context.user(content)

        # Convert tools to OpenAI function calling format
        api_tools = None
        if tools:
            schemas = [_tool_to_openai_schema(tool) for tool in tools]
            api_tools = [s for s in schemas if s is not None]
            if not api_tools:
                api_tools = None
        
        # If we have candidate api_tools schemas, run reduction on each one's parameters
        if api_tools:
            logger.info(f"Checking {len(api_tools)} tool schemas for large enums...")
            try:
                from .runtime import get_runtime
                runtime = get_runtime()
                await reduce_large_enums_in_tool_schemas(
                    api_tools,
                    context_text=processed_content if 'processed_content' in locals() else content,
                    runtime=runtime,
                    threshold=100,
                    target_size=100
                )
            except Exception as e:
                # Non-fatal - continue without reduction
                logger.warning(f'Failed to reduce enums in api_tools: {e}')
        
        # Parse provider from model string if it contains ":"
        if ":" in model:
            provider, model_name = model.split(":", 1)
        else:
            provider = _infer_provider(model)
            model_name = model

        # Loop until terminal tool is called or no tools are invoked
        max_tool_calls = 10  # Prevent infinite loops
        tool_call_count = 0

        while tool_call_count < max_tool_calls:
            # Prepare messages for API call
            messages = context.to_dict_list()

            logger.info(
                f"Calling {provider}:{model_name} with {len(messages)} messages (tool_call_count={tool_call_count})"
            )

            # Call LLM via any-llm
            call_params = {
                "model": model_name,
                "provider": provider,
                "messages": messages,
            }

            if api_tools:
                call_params["tools"] = api_tools
                call_params["tool_choice"] = "auto"

            # If output_schema is provided and we're not using tools, use response_format
            # Convert Pydantic model to JSON schema dict, reduce large enums, and prepare for OpenAI
            if target_output_schema and not api_tools:
                logger.info(f"Preparing response_format for {target_output_schema.__name__ if hasattr(target_output_schema, '__name__') else 'unknown'}...")
                try:
                    import copy
                    from .runtime import get_runtime
                    
                    runtime = get_runtime()
                    
                    # Step 1: Convert Pydantic model to JSON schema
                    schema_dict = target_output_schema.model_json_schema()
                    
                    # Step 2: Reduce large enums (>100 values) using semantic similarity
                    schema_dict = reduce_large_enums(
                        schema_dict,
                        context_text=content,
                        runtime=runtime,
                        threshold=100,
                        target_size=100
                    )
                    
                    # Step 3: Clean schema for OpenAI strict mode (remove extra keys from $ref)
                    schema_dict = clean_schema_for_openai(schema_dict)
                    
                    # Step 4: Wrap in proper response_format structure
                    response_format = prepare_response_format(
                        schema_dict,
                        name=target_output_schema.__name__,
                        strict=True
                    )
                    
                    logger.info(f"Successfully prepared response_format for {target_output_schema.__name__}")
                    call_params["response_format"] = response_format
                    
                except Exception as e:
                    # Fallback - use original schema (might fail but at least we tried)
                    logger.warning(f"Failed to prepare response_format: {e}, using original schema")
                    call_params["response_format"] = target_output_schema
            
            # Call LLM with retry logic using exponential backoff
            # Use retry_strategy.max_iterations if available, default to 3
            import asyncio

            max_retries = retry_strategy.max_iterations if retry_strategy else 3
            base_delay = 0.1  # 100ms initial delay

            for attempt in range(max_retries):
                try:
                    response = await acompletion(**call_params)
                    break  # Success
                except Exception as e:
                    is_last_attempt = attempt >= max_retries - 1

                    if is_last_attempt:
                        # No more retries, raise the error
                        raise

                    # Log the retry attempt
                    logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {type(e).__name__}: {str(e)[:100]}")

                    # Check if this looks like a tool-related error - if so, remove tool_choice
                    error_msg = str(e).lower()
                    is_tool_error = any(
                        [
                            "tool call validation failed" in error_msg,
                            "failed to call a function" in error_msg,
                            "tool_use_failed" in error_msg,
                            "tool use failed" in error_msg,
                        ]
                    )

                    if is_tool_error:
                        logger.warning("Tool error detected, removing tool_choice for retry")
                        call_params.pop("tool_choice", None)

                    # Exponential backoff: 100ms, 200ms, 400ms, 800ms, etc.
                    delay = base_delay * (2**attempt)
                    logger.debug(f"Retrying in {delay * 1000:.0f}ms...")
                    await asyncio.sleep(delay)

            # Extract response
            message = response.choices[0].message
            response_content = message.content or ""
            tool_calls = getattr(message, "tool_calls", None)

            logger.debug(f"LLM response content: {repr(response_content)}")
            logger.debug(f"LLM tool_calls: {tool_calls}")

            # Add assistant message to context
            if tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": tc.type,
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
                context.assistant(response_content, tool_calls=tool_call_dicts)
            else:
                context.assistant(response_content)

            # Execute tool calls if present
            if tool_calls and tools:
                logger.info(f"Executing {len(tool_calls)} tool calls")
                any_terminal_called = False

                for tool_call in tool_calls:
                    func_name = _clean_tool_name(tool_call.function.name)
                    base_name = _extract_base_tool_name(func_name)
                    func_args = json.loads(tool_call.function.arguments)

                    # Find tool by name (try multiple strategies)
                    tool = next(
                        (
                            t
                            for t in tools
                            if t.name == func_name
                            or t.name == base_name
                            or _clean_tool_name(t.name) == func_name
                            or _extract_base_tool_name(t.name) == base_name
                        ),
                        None,
                    )

                    if tool:
                        try:
                            logger.info(f"Calling tool: {func_name}({func_args})")
                            result = await tool(**func_args)

                            # Add result to context - convert Pydantic model to string for context
                            if hasattr(result, "model_dump"):
                                result_str = json.dumps(result.model_dump())
                            elif isinstance(result, dict):
                                result_str = json.dumps(result)
                            else:
                                result_str = str(result)

                            context.tool(content=result_str, name=func_name, tool_call_id=tool_call.id)
                            logger.info(f"Tool {func_name} result: {result}")

                            # If terminal tool, return the result
                            if tool.is_terminal:
                                any_terminal_called = True
                                # Result from terminal tool (Done) should match or be convertible to output_schema
                                logger.debug(
                                    f"Terminal tool called. result type: {type(result).__name__}, "
                                    f"target_output_schema: {target_output_schema.__name__ if target_output_schema else 'None'}"
                                )

                                if target_output_schema:
                                    # Check if result already matches target schema
                                    if isinstance(result, target_output_schema):
                                        logger.info(f"Result is already {target_output_schema.__name__}")
                                        return result

                                    # Try to convert to target schema
                                    # Handle case where result is wrapped in Done's output schema
                                    if isinstance(result, BaseModel):
                                        # Extract fields from result and convert to target schema
                                        try:
                                            # Get result's fields
                                            result_fields = result.__class__.model_fields.keys()

                                            # If result has exactly one field, extract it
                                            if len(result_fields) == 1:
                                                field_name = list(result_fields)[0]
                                                extracted_value = getattr(result, field_name)

                                                # Now check target schema
                                                target_fields = target_output_schema.model_fields.keys()
                                                if len(target_fields) == 1:
                                                    # Both are single-field schemas - wrap extracted value
                                                    target_field = list(target_fields)[0]
                                                    logger.info(
                                                        f"Converting {field_name}={extracted_value} to {target_field}"
                                                    )
                                                    return target_output_schema(**{target_field: extracted_value})
                                                else:
                                                    # Target is multi-field - try full conversion
                                                    if isinstance(extracted_value, dict):
                                                        return target_output_schema(**extracted_value)
                                                    elif isinstance(extracted_value, BaseModel):
                                                        return target_output_schema(**extracted_value.model_dump())
                                                    else:
                                                        # Can't convert
                                                        target_field = list(target_fields)[0]
                                                        return target_output_schema(**{target_field: extracted_value})
                                            else:
                                                # Multi-field result - try full dict conversion
                                                return target_output_schema(**result.model_dump())
                                        except Exception as e:
                                            logger.warning(
                                                f"Could not convert {type(result).__name__} to {target_output_schema.__name__}: {e}"
                                            )
                                            # Fallback: wrap in first field of target schema
                                            target_field = list(target_output_schema.model_fields.keys())[0]
                                            return target_output_schema(**{target_field: result})
                                    elif isinstance(result, dict):
                                        return target_output_schema(**result)
                                    else:
                                        # Primitive value - wrap in target schema
                                        field_name = list(target_output_schema.model_fields.keys())[0]
                                        return target_output_schema(**{field_name: result})
                                else:
                                    # No target schema - return result as-is
                                    logger.info("No target output schema, returning result as-is")
                                    return result
                        except Exception as e:
                            logger.error(f"Error executing {func_name}: {e}")
                            context.tool(content=f"Error: {str(e)}", name=func_name, tool_call_id=tool_call.id)
                    else:
                        logger.warning(f"Tool {func_name} not found")

                # If terminal tool was called, we should have returned by now
                # Otherwise, loop continues to next LLM turn
                if any_terminal_called:
                    break

                tool_call_count += 1
                continue
            else:
                # No tool calls - exit loop
                break

        # Return based on target_output_schema or response content
        # Use target_output_schema which incorporates both the parameter and the tool's default
        # Special case: if no output_schema was passed AND tool default is LLMOutput, return raw string
        # This allows generated code to call llm("question") and get back a string
        if target_output_schema and not (output_schema is None and target_output_schema == LLMOutput):
            # Get retry settings from retry_strategy (should always be set with defaults now)
            max_iterations = retry_strategy.max_iterations if retry_strategy else 3
            num_candidates = retry_strategy.num_candidates if retry_strategy else 3

            # Try to parse response into output_schema with parallel candidates + retries
            import asyncio

            async def try_validate_response(response_text: str, iteration: int, candidate_idx: int) -> BaseModel | None:
                """Try to validate a single response against output_schema."""
                try:
                    # First try parsing as JSON
                    parsed_data = json.loads(response_text)
                    if isinstance(parsed_data, dict):
                        result = target_output_schema(**parsed_data)
                        logger.info(f"Candidate {candidate_idx} iteration {iteration}: Successfully validated")
                        return result
                    else:
                        # If JSON is a primitive, wrap it in the schema
                        field_name = list(target_output_schema.model_fields.keys())[0]
                        result = target_output_schema(**{field_name: parsed_data})
                        logger.info(
                            f"Candidate {candidate_idx} iteration {iteration}: Successfully validated (wrapped)"
                        )
                        return result
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    # If JSON parsing fails, try wrapping the content directly
                    logger.debug(f"Candidate {candidate_idx} iteration {iteration}: JSON parse failed: {e}")
                    try:
                        field_name = list(target_output_schema.model_fields.keys())[0]
                        result = target_output_schema(**{field_name: response_text})
                        logger.info(
                            f"Candidate {candidate_idx} iteration {iteration}: Successfully validated (direct wrap)"
                        )
                        return result
                    except Exception as e2:
                        logger.debug(f"Candidate {candidate_idx} iteration {iteration}: Validation failed: {e2}")
                        return None

            async def generate_and_validate_candidate(candidate_idx: int) -> BaseModel | None:
                """Generate responses with retries for a single candidate."""
                candidate_response = response_content  # Start with initial response

                for iteration in range(max_iterations):
                    # Try to validate current response
                    validated = await try_validate_response(candidate_response, iteration, candidate_idx)
                    if validated is not None:
                        return validated

                    # Validation failed - retry if we have iterations left
                    if iteration < max_iterations - 1:
                        logger.warning(
                            f"Candidate {candidate_idx} iteration {iteration}: Validation failed, retrying..."
                        )

                        # Re-call LLM with stronger instruction
                        retry_messages = context.to_dict_list()
                        retry_messages.append(
                            {
                                "role": "user",
                                "content": f"Your previous response could not be validated. Please respond with valid JSON matching this exact schema: {target_output_schema.model_json_schema()}",
                            }
                        )

                        retry_params = {
                            "model": model_name,
                            "provider": provider,
                            "messages": retry_messages,
                        }

                        if target_output_schema and not api_tools:
                            retry_params["response_format"] = target_output_schema

                        try:
                            retry_response = await acompletion(**retry_params)
                            candidate_response = retry_response.choices[0].message.content or ""
                            logger.debug(
                                f"Candidate {candidate_idx} iteration {iteration}: Got retry response: {candidate_response[:100]}..."
                            )
                        except Exception as retry_error:
                            logger.error(
                                f"Candidate {candidate_idx} iteration {iteration}: Retry call failed: {retry_error}"
                            )
                            break

                return None  # All iterations failed for this candidate

            # Try initial response first (candidate 0, iteration 0)
            initial_result = await try_validate_response(response_content, 0, 0)
            if initial_result is not None:
                return initial_result

            # If we have multiple candidates or iterations, run them in parallel
            if num_candidates > 1 or max_iterations > 1:
                logger.info(
                    f"Initial validation failed. Trying {num_candidates} candidates with {max_iterations} iterations each..."
                )

                # Create tasks for all candidates
                tasks = [generate_and_validate_candidate(i) for i in range(num_candidates)]

                # Wait for first successful result
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    if result is not None:
                        logger.info(f"Got successful validation from one of {num_candidates} candidates")
                        # Cancel remaining tasks
                        for task in tasks:
                            if not task.done():
                                task.cancel()
                        return result

            # All attempts failed - fall back to returning string
            logger.warning(
                f"All validation attempts failed ({num_candidates} candidates × {max_iterations} iterations). Returning raw content."
            )
            return response_content
        else:
            # No output_schema requested - return string content
            # Special case: if response is in LLMOutput format (from response_format), extract content field
            if response_content and isinstance(response_content, str):
                try:
                    parsed = json.loads(response_content)
                    if isinstance(parsed, dict) and "content" in parsed and len(parsed) == 1:
                        # This is LLMOutput format - extract the content field
                        return parsed["content"]
                except (json.JSONDecodeError, KeyError):
                    pass
            return response_content

    return Tool(
        name=f"llm_{model.replace(':', '_').replace('-', '_').replace('/', '_').replace('.', '_')}",
        description=f"Call {model} language model with function calling support",
        input_schema=input_schema or LLMInput,
        output_schema=tool_output_schema,  # Use the already-computed default
        execute=execute,
        is_terminal=False,
    )


__all__ = [
    "LLM",
    "LLMInput",
    "LLMOutput",
    "no_context",
]
