"""Tool model - the fundamental building block for executable functions."""

import copy
import inspect
from collections.abc import Callable
from types import NoneType
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, create_model


class Tool(BaseModel):
    """
    A tool is a callable function with schema validation.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description
        input_schema: Pydantic model for input validation
        output_schema: Pydantic model for output validation
        execute: Async function to execute
        is_terminal: Whether this tool ends execution
    """

    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    execute: Callable
    is_terminal: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """
        Custom JSON schema for the Tool class.

        IMPORTANT: This is NOT used to get schemas for individual tools!
        Individual tool schemas come from: tool.input_schema.model_json_schema()

        Why this is needed:
        - Tool has `execute: Callable` which cannot be serialized to JSON schema
        - When generating schemas for models that contain Tool (e.g., LLMInput with tools: list[Tool]),
          Pydantic needs to generate Tool's class schema to understand the type structure
        - Without this, Pydantic fails with: "Cannot generate JsonSchema for CallableSchema"

        Solution:
        - Remove the `execute` field from the core schema before processing
        - Let Pydantic's handler process the rest normally (proper type checking!)
        - This preserves type validation for name, description, input_schema, output_schema, is_terminal

        Note: input_schema/output_schema are type[BaseModel] (class objects), so their schema
        just indicates "subclass of BaseModel". Actual schemas obtained via tool.input_schema.model_json_schema().
        """
        # Create a copy and remove the execute field
        modified_schema = copy.deepcopy(core_schema)
        if "schema" in modified_schema and modified_schema["schema"].get("type") == "model-fields":
            fields = modified_schema["schema"].get("fields", {})
            if "execute" in fields:
                del fields["execute"]

        # Let Pydantic process the rest with proper type checking
        return handler(modified_schema)

    async def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool with validation.

        Supports multiple styles:
        - Positional BaseModel: await tool(input_obj)  where input_obj is an InputSchema instance
        - Positional string: await tool(content_str)  where tool has 'content' as first parameter
        - Keyword: await tool(field1=value1, field2=value2, ...)
        """
        # If called with a single positional argument
        if len(args) == 1:
            arg = args[0]
            # If it's a BaseModel instance, unpack it
            if isinstance(arg, BaseModel):
                input_obj = arg
                # Extract all fields from the input object as kwargs without serializing
                # Use attribute access to preserve runtime objects (e.g., ToolWrapper)
                try:
                    # Pydantic V2: model_fields contains field names
                    field_names = list(getattr(input_obj.__class__, "model_fields", {}).keys())
                except Exception:
                    field_names = []

                if field_names:
                    kwargs_from_obj = {k: getattr(input_obj, k) for k in field_names}
                else:
                    # Fallback: attempt to use model_dump but keep raw attributes where possible
                    try:
                        dumped = input_obj.model_dump()
                        kwargs_from_obj = {}
                        for k, v in dumped.items():
                            # If the attribute exists on the object, prefer the raw attribute
                            if hasattr(input_obj, k):
                                kwargs_from_obj[k] = getattr(input_obj, k)
                            else:
                                kwargs_from_obj[k] = v
                    except Exception:
                        # Last resort: iterate input schema fields
                        kwargs_from_obj = {
                            k: getattr(input_obj, k) for k in getattr(self.input_schema, "model_fields", {})
                        }

                kwargs = {**kwargs_from_obj, **kwargs}
            # If it's a string and the first input field is 'content', use it for that field
            elif isinstance(arg, str) and "content" in self.input_schema.model_fields:
                kwargs["content"] = arg
            else:
                raise TypeError(f"Tool {self.name} does not accept positional arguments of type {type(arg).__name__}")
        elif args:
            raise TypeError(f"Tool {self.name} does not accept multiple positional arguments")

        # Separate schema fields from extra parameters
        schema_fields = set(self.input_schema.model_fields.keys())
        schema_kwargs = {k: v for k, v in kwargs.items() if k in schema_fields}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in schema_fields}

        # Validate input against schema
        validated_input = self.input_schema(**schema_kwargs)

        # Extract values without serialization to preserve Python objects
        exec_kwargs = {k: getattr(validated_input, k) for k in schema_fields}

        # Add extra parameters (like 'context') that aren't part of the schema
        exec_kwargs.update(extra_kwargs)

        # Execute
        if inspect.iscoroutinefunction(self.execute):
            result = await self.execute(**exec_kwargs)
        else:
            result = self.execute(**exec_kwargs)

        # Validate output
        if isinstance(result, dict):
            validated_output = self.output_schema(**result)
        else:
            # If result is already a Pydantic model or primitive
            validated_output = self.output_schema(result=result) if hasattr(self.output_schema, "result") else result

        return validated_output

    async def execute_with_runtime(self, **kwargs) -> Any:
        """
        Execute the tool and track in runtime context.

        This is a thin wrapper around Runtime.execute() for convenience.
        Uses the global runtime to execute and track this tool call.

        Args:
            **kwargs: Tool inputs

        Returns:
            Tool output
        """
        from ..runtime import get_runtime

        runtime = get_runtime()
        return await runtime.execute(self, **kwargs)


def tool(name: str | None = None, description: str | None = None, is_terminal: bool = False):
    """
    Decorator to convert a Pydantic-typed function into a Tool.

    Example:
        @tool(name="add", description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b
    """

    def decorator(func: Callable) -> Tool:
        # Get function metadata
        func_name = name or func.__name__
        func_desc = description or (func.__doc__ or "").strip()

        # Get type hints
        hints = get_type_hints(func)
        return_type = hints.pop("return", Any)

        # Create input schema from parameters
        sig = inspect.signature(func)
        input_fields = {}
        for param_name, param in sig.parameters.items():
            annotation = hints.get(param_name, Any)
            if param.default is inspect._empty:
                default = ...
            else:
                default = param.default
            input_fields[param_name] = (annotation, default)

        input_model = create_model(f"{func_name}_Input", **input_fields)

        # Create output schema from return type
        if return_type in [Any, NoneType]:
            output_model = create_model(f"{func_name}_Output", result=(Any, ...))
        elif isinstance(return_type, type) and issubclass(return_type, BaseModel):
            output_model = return_type
        else:
            output_model = create_model(f"{func_name}_Output", result=(return_type, ...))

        return Tool(
            name=func_name,
            description=func_desc,
            input_schema=input_model,
            output_schema=output_model,
            execute=func,
            is_terminal=is_terminal,
        )

    return decorator


__all__ = ["Tool", "tool"]
