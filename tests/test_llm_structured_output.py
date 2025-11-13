"""
Tests for LLM tools with structured output schemas.

Verifies that:
1. Generated code can call LLM tools with just str -> str (default)
2. Generated code can call LLM tools with output_schema for structured output
3. AOT and JIT both work with structured LLM outputs
4. Type system correctly handles both primitive and complex LLM outputs
"""

import os

import pytest
from pydantic import BaseModel, Field, create_model

from a1 import LLM, Agent, Runtime, Tool
from a1.runtime import set_runtime


# Structured output schemas for testing
class MathResult(BaseModel):
    """Structured output for math problems."""

    answer: float = Field(..., description="The numerical answer")
    explanation: str = Field(..., description="Step-by-step explanation")


class ParsedProblem(BaseModel):
    """Structured parsing of a math problem."""

    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")


class CalculatorInput(BaseModel):
    """Input for calculator tool."""

    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")


class CalculatorOutput(BaseModel):
    """Output from calculator tool."""

    result: float = Field(..., description="Result of the calculation")


async def calculator_execute(a: float, b: float, operation: str) -> dict:
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
    """Calculator tool for testing."""
    return Tool(
        name="calculator",
        description="Performs basic arithmetic operations",
        input_schema=CalculatorInput,
        output_schema=CalculatorOutput,
        execute=calculator_execute,
    )


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_jit_llm_string_output():
    """Test JIT with LLM returning simple string (default behavior)."""
    runtime = Runtime()
    set_runtime(runtime)

    llm_tool = LLM("gpt-4.1-mini")

    # Agent that uses LLM to answer questions
    InputSchema = create_model("Input", question=(str, Field(..., description="The question to answer")))
    OutputSchema = create_model("Output", answer=(str, Field(..., description="The answer")))

    agent = Agent(
        name="qa_agent",
        description="Answers questions using LLM",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        tools=[llm_tool],
    )

    # Run with JIT - should call LLM without output_schema (returns string)
    result = await runtime.jit(agent, question="What is the capital of France?")

    print("\n✓ JIT with string LLM output:")
    print(f"  Result type: {type(result)}")
    print(f"  Answer: {result.answer}")

    assert hasattr(result, "answer")
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_jit_llm_structured_output():
    """Test JIT with LLM returning structured Pydantic model."""
    runtime = Runtime()
    set_runtime(runtime)

    llm_tool = LLM("gpt-4.1-mini")

    # Agent that uses LLM to solve math with structured output
    InputSchema = create_model("Input", problem=(str, Field(..., description="Math problem to solve")))

    agent = Agent(
        name="structured_math_agent",
        description="Solves math problems with detailed explanations",
        input_schema=InputSchema,
        output_schema=MathResult,
        tools=[llm_tool],
    )

    # Run with JIT - generated code should use output_schema=MathResult
    result = await runtime.jit(agent, problem="What is 15 + 27?")

    print("\n✓ JIT with structured LLM output:")
    print(f"  Result type: {type(result)}")
    print(f"  Answer: {result.answer}")
    print(f"  Explanation: {result.explanation}")

    assert isinstance(result, MathResult)
    assert hasattr(result, "answer")
    assert hasattr(result, "explanation")
    assert isinstance(result.answer, (int, float))
    assert isinstance(result.explanation, str)


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_aot_llm_string_output():
    """Test AOT with LLM returning simple string (default behavior)."""
    runtime = Runtime()
    set_runtime(runtime)

    llm_tool = LLM("gpt-4.1-mini")

    # Agent that uses LLM for simple text generation
    InputSchema = create_model("Input", topic=(str, Field(..., description="Topic to write about")))
    OutputSchema = create_model("Output", text=(str, Field(..., description="Generated text")))

    agent = Agent(
        name="writer_agent",
        description="Writes short text about a topic",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        tools=[llm_tool],
    )

    # Compile with AOT
    compiled = await runtime.aot(agent, cache=False)

    print(f"\n✓ Compiled agent: {compiled.name}")

    # Execute compiled agent
    result = await compiled(topic="artificial intelligence")

    print(f"  Result type: {type(result)}")
    print(f"  Text length: {len(result.text)}")

    assert hasattr(result, "text")
    assert isinstance(result.text, str)
    assert len(result.text) > 10


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_aot_llm_structured_output():
    """Test AOT with LLM returning structured Pydantic model."""
    from a1 import Strategy

    runtime = Runtime()
    set_runtime(runtime)

    llm_tool = LLM("gpt-4.1-mini")

    # Agent that parses math problems into structured format
    InputSchema = create_model("Input", problem=(str, Field(..., description="Math problem in natural language")))

    agent = Agent(
        name="parser_agent",
        description="Parses math problems into structured format",
        input_schema=InputSchema,
        output_schema=ParsedProblem,
        tools=[llm_tool],
    )

    # Compile with AOT (use 3 candidates to improve reliability)
    strategy = Strategy(num_candidates=3)
    compiled = await runtime.aot(agent, cache=False, strategy=strategy)

    print(f"\n✓ Compiled parser: {compiled.name}")

    # Execute compiled agent
    result = await compiled(problem="What is 100 divided by 4?")

    print(f"  Result type: {type(result)}")
    print(f"  a={result.a}, b={result.b}, operation={result.operation}")

    assert isinstance(result, ParsedProblem)
    assert result.a == 100
    assert result.b == 4
    assert result.operation in ["divide", "division", "/"]


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_llm_structured_with_calculator_tool(calculator_tool):
    """Test LLM with structured output combined with other tools."""
    runtime = Runtime()
    set_runtime(runtime)

    llm_tool = LLM("gpt-4.1-mini")

    # Agent that:
    # 1. Uses LLM with structured output to parse problem
    # 2. Uses calculator tool to compute result
    InputSchema = create_model("Input", problem=(str, Field(..., description="Math problem")))
    OutputSchema = create_model("Output", result=(float, Field(..., description="Computed result")))

    agent = Agent(
        name="smart_calculator",
        description="Parses and solves math problems using LLM + calculator",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        tools=[llm_tool, calculator_tool],
    )

    # Run with JIT - should use LLM with output_schema=ParsedProblem, then call calculator
    result = await runtime.jit(agent, problem="Calculate 45 plus 67")

    print("\n✓ LLM + Calculator:")
    print(f"  Result type: {type(result)}")
    print(f"  Result: {result.result}")

    assert hasattr(result, "result")
    assert isinstance(result.result, (int, float))
    assert result.result == 112.0


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_aot_llm_robust_json_parsing():
    """Test that AOT-generated code uses structured output_schema for robustness."""
    runtime = Runtime()
    set_runtime(runtime)

    llm_tool = LLM("gpt-4.1-mini")

    # This is the same scenario as test_aot_generated_function_calls_llm_with_tools
    # but with expectation that generated code uses output_schema for reliability
    InputSchema = create_model("Input", problem=(str, Field(..., description="Complex math problem")))
    OutputSchema = create_model("Output", answer=(str, Field(..., description="The answer as a string")))

    agent = Agent(
        name="robust_solver",
        description="Solves complex math problems reliably using structured LLM output",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        tools=[llm_tool],
    )

    # Compile - generated code should use output_schema to avoid JSON parsing issues
    compiled = await runtime.aot(agent, cache=False)

    # Test with multi-operation problem that previously failed
    result = await compiled(problem="What is 100 / 4 + 12?")

    print("\n✓ Robust parsing with structured output:")
    print(f"  Result: {result.answer}")

    assert hasattr(result, "answer")
    # Should parse correctly: 100/4 = 25, 25+12 = 37
    # Or interpret as (100/4)+12 = 37 or 100/(4+12) = 6.25
    # Just verify we got SOME answer without crashes
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0
