"""
Serialization/deserialization for a1 models.

This module provides JSON serialization and file I/O for Agent, Tool, ToolSet,
Skill, SkillSet, and Runtime. Special handling is required for:
- Callable functions (execute, builder functions, LLM tools)
- Pydantic model types (input_schema, output_schema)
- Recursive Union types (Tool|ToolSet, Skill|SkillSet)

Limitations:
- Callables and Python functions cannot be perfectly serialized to JSON.
- For Tool.execute, we serialize a reference (module.function_name).
- For complex/dynamic functions, use a custom serializer.
"""

import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, create_model

from .models import Agent, Skill, SkillSet, Strategy, Tool, ToolSet


def serialize_callable(func: Any) -> dict[str, Any]:
    """
    Serialize a callable to a reference that can be re-imported.

    Args:
        func: A callable (function, method, or lambda)

    Returns:
        Dict with type, module, name (for regular functions)
        or type, source (for lambdas/closures)

    Raises:
        ValueError: If function cannot be serialized
    """
    if func is None:
        return {"type": "none"}

    # Check if it's a builtin function
    if inspect.isbuiltin(func) or inspect.ismethod(func):
        return {"type": "builtin", "module": func.__module__, "name": func.__qualname__}

    # Try to get module and name
    try:
        module = inspect.getmodule(func)
        if module and hasattr(func, "__name__"):
            return {"type": "function", "module": module.__name__, "name": func.__qualname__}
    except (TypeError, AttributeError):
        pass

    # For lambdas or complex callables, serialize source code
    try:
        source = inspect.getsource(func)
        return {"type": "source", "source": source, "name": getattr(func, "__name__", "callable")}
    except (OSError, TypeError):
        raise ValueError(f"Cannot serialize callable: {func}")


def deserialize_callable(data: dict[str, Any]) -> Any:
    """
    Deserialize a callable from its serialized form.

    Args:
        data: Serialized callable data

    Returns:
        The deserialized callable

    Raises:
        ImportError: If module/function cannot be imported
        ValueError: If source code cannot be executed
    """
    if data.get("type") == "none":
        return None

    if data["type"] in ("builtin", "function"):
        module = importlib.import_module(data["module"])
        return getattr(module, data["name"])

    if data["type"] == "source":
        # Execute source code to get the function
        namespace = {}
        exec(data["source"], namespace)
        return namespace[data["name"]]

    raise ValueError(f"Unknown callable type: {data.get('type')}")


def serialize_model_type(model_type: type[BaseModel]) -> dict[str, Any]:
    """
    Serialize a Pydantic model type to JSON-serializable format.

    Args:
        model_type: A Pydantic model class

    Returns:
        Dict with type info and schema
    """
    if model_type is None:
        return {"type": "none"}

    # For built-in/predefined models, try to get module path
    try:
        module = inspect.getmodule(model_type)
        if module and hasattr(model_type, "__name__"):
            return {"type": "reference", "module": module.__name__, "name": model_type.__qualname__}
    except (TypeError, AttributeError):
        pass

    # For dynamically created models, serialize the schema
    if hasattr(model_type, "model_json_schema"):
        schema = model_type.model_json_schema()
        return {"type": "schema", "name": model_type.__name__, "schema": schema}

    raise ValueError(f"Cannot serialize model type: {model_type}")


def deserialize_model_type(data: dict[str, Any]) -> type[BaseModel]:
    """
    Deserialize a Pydantic model type.

    Args:
        data: Serialized model type data

    Returns:
        The Pydantic model class
    """
    if data.get("type") == "none":
        return None

    if data["type"] == "reference":
        module = importlib.import_module(data["module"])
        return getattr(module, data["name"])

    if data["type"] == "schema":
        # Reconstruct model from schema (simplified - full reconstruction would be complex)
        schema = data["schema"]
        fields = {}
        for prop_name, prop_data in schema.get("properties", {}).items():
            # Simplified type inference
            field_type = _infer_type_from_schema(prop_data)
            is_required = prop_name in schema.get("required", [])
            fields[prop_name] = (field_type, ... if is_required else None)

        return create_model(data["name"], **fields)

    raise ValueError(f"Unknown model type: {data.get('type')}")


def _infer_type_from_schema(schema: dict[str, Any]) -> type:
    """Infer Python type from JSON schema."""
    schema_type = schema.get("type")

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    return type_map.get(schema_type, str)


class ToolEncoder(json.JSONEncoder):
    """JSON encoder for Tool objects."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, Tool):
            return {
                "_type": "Tool",
                "name": obj.name,
                "description": obj.description,
                "input_schema": serialize_model_type(obj.input_schema),
                "output_schema": serialize_model_type(obj.output_schema),
                "execute": serialize_callable(obj.execute),
                "is_terminal": obj.is_terminal,
            }
        elif isinstance(obj, ToolSet):
            return {
                "_type": "ToolSet",
                "name": obj.name,
                "description": obj.description,
                "tools": obj.tools,
            }
        elif isinstance(obj, Skill):
            return {
                "_type": "Skill",
                "name": obj.name,
                "description": obj.description,
                "content": obj.content,
                "modules": obj.modules,
            }
        elif isinstance(obj, SkillSet):
            return {
                "_type": "SkillSet",
                "name": obj.name,
                "description": obj.description,
                "skills": obj.skills,
            }
        elif isinstance(obj, Agent):
            return {
                "_type": "Agent",
                "name": obj.name,
                "description": obj.description,
                "input_schema": serialize_model_type(obj.input_schema),
                "output_schema": serialize_model_type(obj.output_schema),
                "tools": obj.tools,
                "skills": obj.skills,
            }
        elif isinstance(obj, Strategy):
            return {
                "_type": "Strategy",
                **obj.model_dump(),
            }
        return super().default(obj)


def tool_decoder(dct: dict[str, Any]) -> Any:
    """JSON decoder for Tool objects."""
    if "_type" not in dct:
        return dct

    obj_type = dct.pop("_type")

    if obj_type == "Tool":
        return Tool(
            name=dct["name"],
            description=dct["description"],
            input_schema=deserialize_model_type(dct["input_schema"]),
            output_schema=deserialize_model_type(dct["output_schema"]),
            execute=deserialize_callable(dct["execute"]),
            is_terminal=dct["is_terminal"],
        )

    elif obj_type == "ToolSet":
        return ToolSet(
            name=dct["name"],
            description=dct["description"],
            tools=dct["tools"],
        )

    elif obj_type == "Skill":
        return Skill(
            name=dct["name"],
            description=dct["description"],
            content=dct["content"],
            modules=dct.get("modules", []),
        )

    elif obj_type == "SkillSet":
        return SkillSet(
            name=dct["name"],
            description=dct["description"],
            skills=dct["skills"],
        )

    elif obj_type == "Agent":
        return Agent(
            name=dct["name"],
            description=dct["description"],
            input_schema=deserialize_model_type(dct["input_schema"]),
            output_schema=deserialize_model_type(dct["output_schema"]),
            tools=dct["tools"],
            skills=dct.get("skills", []),
        )

    elif obj_type == "Strategy":
        dct.pop("_type", None)
        return Strategy(**dct)

    return dct


def serialize_agent(agent: Agent) -> str:
    """
    Serialize an Agent to JSON string.

    Args:
        agent: Agent to serialize

    Returns:
        JSON string representation
    """
    return json.dumps(agent, cls=ToolEncoder, indent=2)


def deserialize_agent(json_str: str) -> Agent:
    """
    Deserialize an Agent from JSON string.

    Args:
        json_str: JSON string representation

    Returns:
        Deserialized Agent
    """
    data = json.loads(json_str, object_hook=tool_decoder)
    if isinstance(data, Agent):
        return data
    raise ValueError(f"Expected Agent, got {type(data)}")


def serialize_tool(tool: Tool | ToolSet) -> str:
    """Serialize a Tool or ToolSet to JSON string."""
    return json.dumps(tool, cls=ToolEncoder, indent=2)


def deserialize_tool(json_str: str) -> Tool | ToolSet:
    """Deserialize a Tool or ToolSet from JSON string."""
    data = json.loads(json_str, object_hook=tool_decoder)
    if isinstance(data, (Tool, ToolSet)):
        return data
    raise ValueError(f"Expected Tool or ToolSet, got {type(data)}")


def serialize_skill(skill: Skill | SkillSet) -> str:
    """Serialize a Skill or SkillSet to JSON string."""
    return json.dumps(skill, cls=ToolEncoder, indent=2)


def deserialize_skill(json_str: str) -> Skill | SkillSet:
    """Deserialize a Skill or SkillSet from JSON string."""
    data = json.loads(json_str, object_hook=tool_decoder)
    if isinstance(data, (Skill, SkillSet)):
        return data
    raise ValueError(f"Expected Skill or SkillSet, got {type(data)}")


# File I/O convenience methods


def save_agent_to_file(agent: Agent, path: str | Path) -> None:
    """
    Save an Agent to a JSON file.

    Args:
        agent: Agent to save
        path: File path to save to
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(serialize_agent(agent))


def load_agent_from_file(path: str | Path) -> Agent:
    """
    Load an Agent from a JSON file.

    Args:
        path: File path to load from

    Returns:
        Loaded Agent
    """
    with open(path) as f:
        return deserialize_agent(f.read())


def save_tool_to_file(tool: Tool | ToolSet, path: str | Path) -> None:
    """Save a Tool or ToolSet to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(serialize_tool(tool))


def load_tool_from_file(path: str | Path) -> Tool | ToolSet:
    """Load a Tool or ToolSet from a JSON file."""
    with open(path) as f:
        return deserialize_tool(f.read())


def save_skill_to_file(skill: Skill | SkillSet, path: str | Path) -> None:
    """Save a Skill or SkillSet to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(serialize_skill(skill))


def load_skill_from_file(path: str | Path) -> Skill | SkillSet:
    """Load a Skill or SkillSet from a JSON file."""
    with open(path) as f:
        return deserialize_skill(f.read())


__all__ = [
    "serialize_callable",
    "deserialize_callable",
    "serialize_model_type",
    "deserialize_model_type",
    "serialize_agent",
    "deserialize_agent",
    "serialize_tool",
    "deserialize_tool",
    "serialize_skill",
    "deserialize_skill",
    "save_agent_to_file",
    "load_agent_from_file",
    "save_tool_to_file",
    "load_tool_from_file",
    "save_skill_to_file",
    "load_skill_from_file",
]
