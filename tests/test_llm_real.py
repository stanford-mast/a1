"""
Real-world tests for LLM tool with actual API calls.

Tests with multiple providers:
- OpenAI (gpt-4.1)
- Anthropic (claude-haiku-4-5)
- Google (gemini-2.5-flash)
- Groq (groq:meta-llama/llama-4-maverick-17b-128e-instruct, groq:openai/gpt-oss-20b)

Run with: pytest tests/test_llm_real.py -v -s
"""

import os

import pytest
from pydantic import BaseModel, Field

from a1 import LLM, Agent, Runtime, Tool, get_context


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, or divide")


class CalculatorOutput(BaseModel):
    """Output from calculator tool."""

    result: float = Field(..., description="Result of the calculation")


async def calculate_execute(a: float, b: float, operation: str) -> dict:
    """Perform a calculation."""
    if operation == "add":
        return {"result": a + b}
    elif operation == "subtract":
        return {"result": a - b}
    elif operation == "multiply":
        return {"result": a * b}
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return {"result": a / b}
    else:
        raise ValueError(f"Unknown operation: {operation}")


@pytest.fixture
def calculator_tool():
    """Create a calculator tool for testing."""
    return Tool(
        name="calculator",
        description="Perform basic arithmetic operations",
        input_schema=CalculatorInput,
        output_schema=CalculatorOutput,
        execute=calculate_execute,
        is_terminal=False,
    )


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_openai_gpt_4_1_function_calling(calculator_tool):
    """Test OpenAI GPT-4.1 with function calling."""
    runtime = Runtime()
    llm_tool = LLM("gpt-4.1-mini")

    agent = Agent(name="calc_agent", description="Calculator agent", tools=[calculator_tool, llm_tool])
    runtime.current_agent = agent

    # Execute LLM with calculator tool
    result = await runtime.execute(
        llm_tool,
        **{"content": "What is 15 multiplied by 7?", "tools": [calculator_tool], "context": get_context("test_openai")},
    )

    print(f"\nOpenAI response: {result}")

    # Check context has proper message flow
    ctx = get_context("test_openai")
    assert len(ctx.messages) >= 3  # user, assistant with tool_calls, tool result
    assert ctx.messages[0].role == "user"
    assert ctx.messages[1].role == "assistant"

    # Should have called calculator
    assert any(msg.role == "tool" and msg.name == "calculator" for msg in ctx.messages)


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
@pytest.mark.asyncio
async def test_claude_haiku_function_calling(calculator_tool):
    """Test Anthropic Claude Haiku 4.5 with function calling."""
    runtime = Runtime()
    llm_tool = LLM("claude-haiku-4-5")

    agent = Agent(name="calc_agent", tools=[calculator_tool, llm_tool])
    runtime.current_agent = agent

    result = await runtime.execute(
        llm_tool,
        **{"content": "Calculate 42 divided by 6", "tools": [calculator_tool], "context": get_context("test_claude")},
    )

    print(f"\nClaude response: {result}")

    ctx = get_context("test_claude")
    assert len(ctx.messages) >= 3
    assert any(msg.role == "tool" and msg.name == "calculator" for msg in ctx.messages)


@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_gemini_flash_function_calling(calculator_tool):
    """Test Google Gemini 2.5 Flash with function calling."""
    runtime = Runtime()
    llm_tool = LLM("gemini-2.5-flash")

    agent = Agent(name="calc_agent", tools=[calculator_tool, llm_tool])
    runtime.current_agent = agent

    result = await runtime.execute(
        llm_tool, **{"content": "Add 123 and 456", "tools": [calculator_tool], "context": get_context("test_gemini")}
    )

    print(f"\nGemini response: {result}")

    ctx = get_context("test_gemini")
    assert len(ctx.messages) >= 3
    assert any(msg.role == "tool" and msg.name == "calculator" for msg in ctx.messages)


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_groq_llama_function_calling(calculator_tool):
    """Test Groq with Llama 4 Maverick function calling."""
    runtime = Runtime()
    llm_tool = LLM("gpt-4.1-mini")

    agent = Agent(name="calc_agent", tools=[calculator_tool, llm_tool])
    runtime.current_agent = agent

    result = await runtime.execute(
        llm_tool,
        **{"content": "Subtract 50 from 100", "tools": [calculator_tool], "context": get_context("test_groq_llama")},
    )

    print(f"\nGroq Llama response: {result}")

    ctx = get_context("test_groq_llama")
    assert len(ctx.messages) >= 3
    assert any(msg.role == "tool" and msg.name == "calculator" for msg in ctx.messages)


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_groq_gpt_oss_function_calling(calculator_tool):
    """Test Groq with GPT OSS function calling."""
    runtime = Runtime()
    llm_tool = LLM("gpt-4.1-mini")

    agent = Agent(name="calc_agent", tools=[calculator_tool, llm_tool])
    runtime.current_agent = agent

    result = await runtime.execute(
        llm_tool, **{"content": "Multiply 9 by 8", "tools": [calculator_tool], "context": get_context("test_groq_gpt")}
    )

    print(f"\nGroq GPT OSS response: {result}")

    ctx = get_context("test_groq_gpt")
    assert len(ctx.messages) >= 3
    assert any(msg.role == "tool" and msg.name == "calculator" for msg in ctx.messages)


@pytest.mark.asyncio
async def test_multi_turn_conversation(calculator_tool):
    """Test multi-turn conversation with tool calling."""
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    runtime = Runtime()
    llm_tool = LLM("gpt-4.1-mini")

    agent = Agent(name="calc_agent", tools=[calculator_tool, llm_tool])
    runtime.current_agent = agent

    # First turn
    result1 = await runtime.execute(
        llm_tool, **{"content": "Calculate 10 + 5", "tools": [calculator_tool], "context": get_context("multi_turn")}
    )
    print(f"\nFirst turn: {result1}")

    # Second turn - should remember context
    result2 = await runtime.execute(
        llm_tool,
        **{
            "content": "Now multiply that result by 2",
            "tools": [calculator_tool],
            "context": get_context("multi_turn"),
        },
    )
    print(f"\nSecond turn: {result2}")

    # Check context has both exchanges
    ctx = get_context("multi_turn")
    assert len(ctx.messages) >= 6  # 2 user messages, 2 assistant, 2 tool results

    # Both calculator calls should be in context
    tool_msgs = [msg for msg in ctx.messages if msg.role == "tool" and msg.name == "calculator"]
    assert len(tool_msgs) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
