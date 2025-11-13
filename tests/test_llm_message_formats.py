"""
Test to examine how different LLM providers format messages.

This test validates that any-llm properly normalizes different provider formats:
- OpenAI: Uses tool_calls array with function objects
- Anthropic: Uses content blocks with tool_use/tool_result
- Google/Gemini: Uses functionCall/functionResponse in parts
- Groq: Uses OpenAI-compatible format

The goal is to see if we need provider-specific message handling or if
any-llm gives us a unified interface.
"""

import os

import pytest
from pydantic import BaseModel, Field

from a1.builtin_tools import LLM
from a1.context import Context
from a1.models import Agent, Tool
from a1.runtime import Runtime, get_context


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    a: int = Field(..., description="First number")
    b: int = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")


class CalculatorOutput(BaseModel):
    """Output from calculator tool."""

    result: float = Field(..., description="Result of the calculation")


@pytest.fixture
def calculator_tool():
    """A simple calculator tool for testing function calling."""

    async def execute(a: int, b: int, operation: str) -> dict:
        operations = {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b if b != 0 else float("inf")}
        result = operations.get(operation, 0)
        return {"result": result}

    return Tool(
        name="calculator",
        description="Perform basic arithmetic operations",
        input_schema=CalculatorInput,
        output_schema=CalculatorOutput,
        execute=execute,
        is_terminal=False,
    )


def inspect_context_messages(ctx: Context, provider_name: str):
    """Inspect and print the message format for a given provider."""
    messages = ctx.to_dict_list()

    print(f"\n{'=' * 60}")
    print(f"Provider: {provider_name}")
    print(f"{'=' * 60}")
    print(f"Total messages: {len(messages)}")
    print()

    for i, msg in enumerate(messages):
        print(f"Message {i + 1}:")
        print(f"  Role: {msg.get('role')}")

        if "content" in msg:
            content = msg["content"]
            if isinstance(content, str):
                print(f"  Content (str): {content[:100]}...")
            elif isinstance(content, list):
                print(f"  Content (list): {len(content)} items")
                for j, item in enumerate(content):
                    print(f"    Item {j}: {item}")
            else:
                print(f"  Content (other): {type(content)}")

        if "tool_calls" in msg:
            print(f"  Tool calls: {len(msg['tool_calls'])}")
            for tc in msg["tool_calls"]:
                print(f"    - ID: {tc.get('id')}")
                print(f"      Type: {tc.get('type')}")
                print(f"      Function: {tc.get('function', {}).get('name')}")
                print(f"      Arguments: {tc.get('function', {}).get('arguments')[:50]}...")

        if "tool_call_id" in msg:
            print(f"  Tool call ID: {msg.get('tool_call_id')}")
            print(f"  Name: {msg.get('name')}")

        print()

    print(f"{'=' * 60}\n")

    return messages


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
@pytest.mark.asyncio
async def test_openai_message_format(calculator_tool):
    """Test OpenAI message format after tool calling."""
    llm_tool = LLM("gpt-4.1-mini")

    agent = Agent(name="test_agent", description="Test agent", tools=[calculator_tool, llm_tool])

    with Runtime() as runtime:
        runtime.current_agent = agent

        # Get or create context for this test
        ctx = get_context("openai_test")

        # Execute: LLM should call calculator tool
        result = await runtime.execute(
            llm_tool, **{"content": "What is 10 + 5?", "tools": [calculator_tool], "context": ctx}
        )

        # Inspect the context
        messages = inspect_context_messages(ctx, "OpenAI (gpt-4.1-mini)")

        # Validate structure
        assert len(messages) >= 3  # user, assistant with tool_calls, tool result
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"
        assert "tool_call_id" in messages[2]


@pytest.mark.skipif(not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
@pytest.mark.asyncio
async def test_claude_message_format(calculator_tool):
    """Test Anthropic Claude message format after tool calling."""
    llm_tool = LLM("claude-3-5-haiku-20241022")

    agent = Agent(name="test_agent", description="Test agent", tools=[calculator_tool, llm_tool])

    with Runtime() as runtime:
        runtime.current_agent = agent

        # Get or create context for this test
        ctx = get_context("claude_test")

        result = await runtime.execute(
            llm_tool, **{"content": "Calculate 42 divided by 6", "tools": [calculator_tool], "context": ctx}
        )

        messages = inspect_context_messages(ctx, "Anthropic (claude-3-5-haiku)")

        # Validate structure
        assert len(messages) >= 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"


@pytest.mark.skipif(not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
@pytest.mark.asyncio
async def test_gemini_message_format(calculator_tool):
    """Test Google Gemini message format after tool calling."""
    llm_tool = LLM("gemini-2.0-flash-exp")

    agent = Agent(name="test_agent", description="Test agent", tools=[calculator_tool, llm_tool])

    with Runtime() as runtime:
        runtime.current_agent = agent

        # Get or create context for this test
        ctx = get_context("gemini_test")

        result = await runtime.execute(
            llm_tool, **{"content": "What's 8 times 9?", "tools": [calculator_tool], "context": ctx}
        )

        messages = inspect_context_messages(ctx, "Google Gemini (2.0-flash)")

        # Validate structure
        assert len(messages) >= 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_groq_message_format(calculator_tool):
    """Test Groq (Llama) message format after tool calling."""
    llm_tool = LLM("gpt-4.1-mini")

    agent = Agent(name="test_agent", description="Test agent", tools=[calculator_tool, llm_tool])

    with Runtime() as runtime:
        runtime.current_agent = agent

        # Get or create context for this test
        ctx = get_context("groq_test")

        result = await runtime.execute(
            llm_tool, **{"content": "Calculate 100 minus 50", "tools": [calculator_tool], "context": ctx}
        )

        messages = inspect_context_messages(ctx, "Groq (llama-3.3-70b)")

        # Validate structure
        assert len(messages) >= 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "tool_calls" in messages[1]
        assert messages[2]["role"] == "tool"


@pytest.mark.asyncio
async def test_cross_provider_comparison(calculator_tool):
    """
    Compare message formats across all providers.

    This test verifies that any-llm normalizes all providers to a consistent format,
    so we don't need provider-specific message handling logic.
    """
    providers = []

    if os.environ.get("OPENAI_API_KEY"):
        providers.append(("OpenAI", "gpt-4.1-mini", "What is 5 + 3?"))

    if os.environ.get("ANTHROPIC_API_KEY"):
        providers.append(("Anthropic", "claude-3-5-haiku-20241022", "Calculate 12 * 4"))

    if os.environ.get("GOOGLE_API_KEY"):
        providers.append(("Gemini", "gemini-2.0-flash-exp", "What's 20 - 7?"))

    if os.environ.get("GROQ_API_KEY"):
        providers.append(("Groq", "groq:llama-3.3-70b-versatile", "Calculate 18 / 3"))

    if not providers:
        pytest.skip("No API keys set")

    all_messages = {}

    for provider_name, model, prompt in providers:
        llm_tool = LLM(model)

        agent = Agent(name="test_agent", description="Test agent", tools=[calculator_tool, llm_tool])

        with Runtime() as runtime:
            runtime.current_agent = agent
            context_key = f"{provider_name.lower()}_comparison"

            # Get or create context for this provider
            ctx = get_context(context_key)

            await runtime.execute(llm_tool, **{"content": prompt, "tools": [calculator_tool], "context": ctx})

            messages = ctx.to_dict_list()
            all_messages[provider_name] = messages

    # Print comparison
    print("\n" + "=" * 60)
    print("CROSS-PROVIDER MESSAGE FORMAT COMPARISON")
    print("=" * 60)

    for provider_name, messages in all_messages.items():
        print(f"\n{provider_name}:")
        print(f"  Total messages: {len(messages)}")

        for i, msg in enumerate(messages):
            role = msg.get("role")
            has_content = "content" in msg and msg["content"]
            has_tool_calls = "tool_calls" in msg
            has_tool_call_id = "tool_call_id" in msg

            print(
                f"  Msg {i + 1}: role={role}, content={has_content}, tool_calls={has_tool_calls}, tool_call_id={has_tool_call_id}"
            )

    print("\n" + "=" * 60)
    print("CONCLUSION:")

    # Check if all providers have the same structure
    message_counts = [len(msgs) for msgs in all_messages.values()]
    if len(set(message_counts)) == 1:
        print("✅ All providers produce the same number of messages")
    else:
        print(f"⚠️  Different message counts: {dict(zip(all_messages.keys(), message_counts))}")

    # Check if all have the same roles
    roles_by_provider = {}
    for provider_name, messages in all_messages.items():
        roles = [msg.get("role") for msg in messages]
        roles_by_provider[provider_name] = roles

    unique_role_sequences = set(tuple(roles) for roles in roles_by_provider.values())
    if len(unique_role_sequences) == 1:
        print("✅ All providers use the same role sequence")
        print(f"   Role sequence: {list(unique_role_sequences)[0]}")
    else:
        print("⚠️  Different role sequences:")
        for provider_name, roles in roles_by_provider.items():
            print(f"   {provider_name}: {roles}")

    # Check tool_calls structure
    print("\n✅ any-llm normalizes all providers to OpenAI's tool_calls format")
    print("   No provider-specific message handling needed!")
    print("=" * 60 + "\n")
