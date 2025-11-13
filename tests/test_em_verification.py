"""
Test that agents with large enums require EM tool.

This verifies that the compiler fails at AOT time (not runtime) if:
- Agent has large enum (>100 values) in input/output/tool schemas
- EM tool is not available in agent tools
"""
import pytest
import asyncio
from enum import Enum
from pydantic import BaseModel, Field, ConfigDict
from a1 import Agent, tool, LLM, Runtime, EM


# Create a large enum (200 values)
LargeCategory = Enum(
    'LargeCategory',
    {f'CATEGORY_{i:03d}': f'category_{i}' for i in range(200)}
)


class InputWithLargeEnum(BaseModel):
    """Input with large enum."""
    model_config = ConfigDict(extra='forbid')
    description: str = Field(description="Description")
    category: LargeCategory = Field(description="Category selection")


class OutputWithLargeEnum(BaseModel):
    """Output with large enum."""
    model_config = ConfigDict(extra='forbid')
    selected_category: LargeCategory = Field(description="Selected category")
    confidence: float = Field(description="Confidence score")


class SimpleInput(BaseModel):
    """Simple input without large enum."""
    description: str = Field(description="Description")


class SimpleOutput(BaseModel):
    """Simple output without large enum."""
    result: str = Field(description="Result")


@pytest.mark.asyncio
async def test_large_enum_without_em_fails():
    """Test that agent with large enum but no EM tool fails compilation."""
    agent = Agent(
        name="test_agent",
        description="Agent with large enum in output",
        input_schema=SimpleInput,
        output_schema=OutputWithLargeEnum,  # Has 200-value enum
        tools=[LLM("gpt-4o-mini")]  # No EM tool! (Done auto-added by LLM)
    )
    
    runtime = Runtime()
    
    with pytest.raises(ValueError) as exc_info:
        await runtime.aot(agent)
    
    error_msg = str(exc_info.value)
    assert 'large enum' in error_msg.lower()
    assert 'em' in error_msg.lower()
    assert '200 values' in error_msg


@pytest.mark.asyncio
async def test_large_enum_with_em_succeeds():
    """Test that agent with large enum AND EM tool succeeds compilation."""
    agent = Agent(
        name="test_agent",
        description="Agent with large enum in output",
        input_schema=SimpleInput,
        output_schema=OutputWithLargeEnum,  # Has 200-value enum
        tools=[EM(), LLM("gpt-4o-mini")]  # Has EM tool! (Done auto-added by LLM)
    )
    
    runtime = Runtime()
    compiled = await runtime.aot(agent)
    
    assert compiled is not None
    assert compiled.name == "test_agent"


@pytest.mark.asyncio
async def test_small_enum_without_em_succeeds():
    """Test that agent with small enum (<100 values) succeeds without EM."""
    # Create small enum (50 values)
    SmallCategory = Enum(
        'SmallCategory',
        {f'CAT_{i:02d}': f'cat_{i}' for i in range(50)}
    )
    
    class OutputWithSmallEnum(BaseModel):
        model_config = ConfigDict(extra='forbid')
        category: SmallCategory = Field(description="Category")
        result: str = Field(description="Result")
    
    agent = Agent(
        name="test_agent",
        description="Agent with small enum",
        input_schema=SimpleInput,
        output_schema=OutputWithSmallEnum,  # Only 50 values
        tools=[LLM("gpt-4o-mini")]  # No EM needed (Done auto-added by LLM)
    )
    
    runtime = Runtime()
    compiled = await runtime.aot(agent)
    
    assert compiled is not None
    assert compiled.name == "test_agent"
