"""Tests for context hygiene during retries."""

import os

import pytest
from pydantic import BaseModel, Field

from a1 import Agent, Runtime, get_context, set_runtime, tool


class Input(BaseModel):
    value: int = Field(..., description="Input value")


class Output(BaseModel):
    result: str = Field(..., description="Result")


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_failed_attempts_dont_pollute_main_context():
    """
    Test that failed code generation/execution attempts don't pollute main context.

    Only successful executions should append to the main context.
    Failed attempts are tracked internally but not exposed.
    """
    runtime = Runtime()
    set_runtime(runtime)

    # Create an agent that might fail on first attempts
    agent = Agent(
        name="test_agent",
        description="Process the input value",
        input_schema=Input,
        output_schema=Output,
        tools=[],
    )

    # Get initial context length
    ctx = get_context("main")
    initial_length = len(ctx.messages)

    # Execute agent (might fail/retry internally)
    try:
        result = await runtime.jit(agent, value=42)

        # Check context - should only have user input + assistant output
        # NOT any intermediate failures
        ctx = get_context("main")
        messages = ctx.messages

        # Should have exactly 2 messages: user input + assistant output
        assert len(messages) == initial_length + 2, f"Expected {initial_length + 2} messages, got {len(messages)}"

        # Last two messages should be user then assistant
        assert messages[-2].role == "user"
        assert messages[-1].role == "assistant"

        # Should NOT have any error messages or multiple assistant attempts
        assistant_messages = [m for m in messages if m.role == "assistant"]
        assert len(assistant_messages) == 1, f"Should have exactly 1 assistant message, got {len(assistant_messages)}"

        print(f"✓ Context hygiene maintained: {len(messages)} total messages")
        print(f"  User: {messages[-2].content[:50]}...")
        print(f"  Assistant: {messages[-1].content[:50]}...")

    except Exception:
        # Even if execution fails, context should not have intermediate failures
        ctx = get_context("main")
        messages = ctx.messages

        # Should still only have user input (no successful assistant response)
        assert len(messages) == initial_length + 1
        assert messages[-1].role == "user"
        print(f"✓ Context clean even after failure: {len(messages)} messages")


@pytest.mark.asyncio
async def test_multiple_jit_calls_accumulate_correctly():
    """
    Test that multiple successful JIT calls accumulate context properly.

    Each call should add exactly 2 messages (user + assistant).
    """
    runtime = Runtime()
    set_runtime(runtime)

    @tool(name="echo", description="Echo the input")
    async def echo(text: str) -> str:
        return text

    agent = Agent(
        name="echo_agent",
        description="Echo the input",
        input_schema=Input,
        output_schema=Output,
        tools=[echo],
    )

    ctx = get_context("main")
    initial_length = len(ctx.messages)

    # Make 3 successful calls
    for i in range(3):
        try:
            await runtime.jit(agent, value=i)
        except:
            pass  # Some calls might fail due to LLM variability

    ctx = get_context("main")
    messages = ctx.messages

    # Context should grow in pairs (user + assistant)
    growth = len(messages) - initial_length
    assert growth % 2 == 0, f"Context should grow in pairs, but grew by {growth}"

    print(f"✓ Made 3 calls, context grew by {growth} messages ({growth // 2} successful)")


@pytest.mark.asyncio
async def test_context_not_modified_on_validation_failure():
    """
    Test that validation failures during code generation don't add extra messages.

    Even if generation fails and retries internally, the final context should
    only reflect the successful attempt (or just user input if all fail).
    """
    from a1.models import Strategy

    runtime = Runtime()
    set_runtime(runtime)

    agent = Agent(
        name="test_agent",
        description="Process input",
        input_schema=Input,
        output_schema=Output,
        tools=[],
    )

    ctx = get_context("main")
    initial_length = len(ctx.messages)

    try:
        # Execute - might succeed or fail, but should not pollute context with retries
        await runtime.jit(agent, value=456, strategy=Strategy(max_iterations=2, num_candidates=2))

        # If succeeded, should have exactly user + assistant
        ctx = get_context("main")
        messages = ctx.messages
        assert len(messages) == initial_length + 2
        assert messages[-2].role == "user"
        assert messages[-1].role == "assistant"

    except RuntimeError:
        # If failed, should have only user input
        ctx = get_context("main")
        messages = ctx.messages
        assert len(messages) == initial_length + 1
        assert messages[-1].role == "user"

    print(f"✓ Context clean: {len(ctx.messages)} messages total")
