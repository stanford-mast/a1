"""
Built-in tools for a1 agents.

Provides:
- LLM: Tool for calling language models with function calling (see llm.py)
- Done: Terminal tool for returning agent output
"""

from typing import Any

from pydantic import BaseModel, create_model

from .llm import LLM, LLMInput, LLMOutput  # Import from llm.py
from .models import Tool


def Done(output_schema: type[BaseModel] | None = None) -> Tool:
    """
    Create a Done tool that marks agent execution as complete.

    The Done tool simply returns its input as output, but marks itself
    as a terminal tool to signal completion.

    Args:
        output_schema: Optional Pydantic model for output validation

    Returns:
        Terminal tool that returns its input
    """
    # Use provided schema or create a generic one
    if output_schema is None:
        OutputModel = create_model("DoneOutput", result=(Any, ...))
    else:
        OutputModel = output_schema

    async def execute(**kwargs) -> dict[str, Any]:
        """Return input as output."""
        return kwargs

    return Tool(
        name="done",
        description="Mark task as complete and return result",
        input_schema=output_schema or create_model("DoneInput", result=(Any, ...)),
        output_schema=OutputModel,
        execute=execute,
        is_terminal=True,
    )


__all__ = [
    "LLM",
    "Done",
    "LLMInput",
    "LLMOutput",
]
