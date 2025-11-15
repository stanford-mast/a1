"""Message model for chat contexts."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message in a context."""

    role: str = Field(..., description="Message role (user, assistant, system, tool)")
    content: str = Field(..., description="Message content")
    name: str | None = Field(None, description="Name of tool or function")
    tool_call_id: str | None = Field(None, description="ID of tool call")
    tool_calls: list[dict[str, Any]] | None = Field(None, description="Tool calls made")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message creation timestamp")
    message_id: str = Field(default_factory=lambda: uuid4().hex, description="Unique message ID for deduplication")


__all__ = ["Message"]
