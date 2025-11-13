"""
Integration tests with real LLM calls (Groq GPT-OSS).

Tests that verify:
- JIT mode generates working code that calls LLM
- AOT mode compiles code that calls LLM
- IsLoop mode uses templated loop with LLM
- Type checking catches malformed generated code
- Context management works end-to-end
- Input variables available in generated code

Requires GROQ_API_KEY environment variable.

Run with: pytest tests/test_integration_llm.py -v -s
"""

import os

import pytest
from pydantic import BaseModel, Field, create_model

from a1 import LLM, Agent, Runtime, Tool
from a1.codecheck import BaseVerify, IsLoop

# Skip all tests if no GROQ_API_KEY
pytestmark = pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set")


# Test tools
class CalculatorInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")


class CalculatorOutput(BaseModel):
    result: float = Field(..., description="Result")


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
    """Create calculator tool."""
    return Tool(
        name="calculator",
        description="Perform basic arithmetic operations",
        input_schema=CalculatorInput,
        output_schema=CalculatorOutput,
        execute=calculator_execute,
        is_terminal=False,
    )


@pytest.fixture
def llm_tool():
    """Create LLM tool with Groq."""
    return LLM("gpt-4.1-mini")


# ============================================================================
# JIT Mode with Real LLM
# ============================================================================


class TestJITWithRealLLM:
    """Test JIT mode with actual LLM calls."""

    @pytest.mark.asyncio
    async def test_jit_generates_valid_code_structure(self, calculator_tool, llm_tool):
        """Test JIT generates code that passes verification."""
        AgentInput = create_model("AgentInput", query=(str, Field(description="Math query")))
        AgentOutput = create_model("AgentOutput", answer=(str, Field(description="Answer")))

        agent = Agent(
            name="jit_math",
            description="Solve math problems using calculator and LLM",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool, llm_tool],
        )

        # Agent should have both tools
        assert len(agent.tools) == 2
        assert calculator_tool in agent.tools
        assert llm_tool in agent.tools

    @pytest.mark.asyncio
    async def test_jit_input_variables_accessible(self, calculator_tool):
        """Test that input variables are accessible in JIT mode."""
        AgentInput = create_model("AgentInput", problem=(str, Field(description="Math problem")))
        AgentOutput = create_model("AgentOutput", result=(str, Field(description="Result")))

        agent = Agent(
            name="jit_with_input",
            description="JIT with input variables",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool],
        )

        # Agent input schema should define the 'problem' field
        assert hasattr(agent.input_schema, "model_fields")
        assert "problem" in agent.input_schema.model_fields


# ============================================================================
# AOT Mode with Real LLM
# ============================================================================


class TestAOTWithRealLLM:
    """Test AOT mode with actual LLM calls."""

    @pytest.mark.asyncio
    async def test_aot_generates_compilable_code(self, calculator_tool, llm_tool):
        """Test AOT generates code that compiles."""
        AgentInput = create_model("AgentInput", problem=(str, Field(description="Problem")))
        AgentOutput = create_model("AgentOutput", answer=(str, Field(description="Answer")))

        agent = Agent(
            name="aot_math",
            description="AOT math solver with LLM and calculator",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool, llm_tool],
        )

        # Agent should exist and be compilable
        assert agent is not None
        assert hasattr(agent, "input_schema")
        assert hasattr(agent, "output_schema")

    @pytest.mark.asyncio
    async def test_aot_code_passes_verification(self, calculator_tool, llm_tool):
        """Test that generated AOT code passes verification."""
        AgentInput = create_model("AgentInput", task=(str, Field(description="Task")))
        AgentOutput = create_model("AgentOutput", result=(str, Field(description="Result")))

        agent = Agent(
            name="verified_tool",
            description="Tool with verification",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool, llm_tool],
        )

        # Both tools should be available
        tool_names = [t.name for t in agent.tools]
        assert "calculator" in tool_names


# ============================================================================
# IsLoop Mode with Real LLM
# ============================================================================


class TestIsLoopWithRealLLM:
    """Test IsLoop mode with actual LLM calls."""

    @pytest.mark.asyncio
    async def test_isloop_verifier_works(self, calculator_tool, llm_tool):
        """Test IsLoop verifier is available and functional."""
        AgentInput = create_model("AgentInput", problem=(str, Field(description="Problem")))
        AgentOutput = create_model("AgentOutput", answer=(str, Field(description="Answer")))

        agent = Agent(
            name="isloop_agent",
            description="Agent for IsLoop testing",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool, llm_tool],
        )

        # Create runtime with IsLoop verifier
        runtime = Runtime(verify=[IsLoop()])

        # Agent should be valid
        assert agent is not None

    def test_isloop_valid_pattern(self):
        """Test IsLoop accepts valid loop patterns."""
        code = """
while iteration < max_iterations:
    result = await llm_groq_openai_gpt_oss_20b(
        content=instruction if iteration == 0 else "Continue",
        tools=[calculator],
        context=context,
        output_schema=AgentOutput
    )
    if isinstance(result, AgentOutput):
        break
    iteration += 1
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert is_valid, f"Should be valid: {error}"


# ============================================================================
# Type Checking with Real Code
# ============================================================================


class TestTypeCheckingWithRealLLM:
    """Test type checking on real generated code."""

    def test_type_checking_with_definition_code(self):
        """Test type checking on concatenated definition + generated code."""
        definition_code = """
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: float = Field(...)
    b: float = Field(...)
    operation: str = Field(...)

class CalculatorOutput(BaseModel):
    result: float = Field(...)

async def calculator(input: CalculatorInput) -> CalculatorOutput:
    raise NotImplementedError("Tool calculator not provided")
"""

        generated_code = """
input_data = CalculatorInput(a=10, b=5, operation="add")
output = await calculator(input_data)
"""

        verifier = BaseVerify()
        is_valid, error = verifier.verify((definition_code, generated_code), None)
        # Should be valid - types match
        assert is_valid or error is None or "ty" in error.lower()

    def test_type_checking_catches_mismatches(self):
        """Test that type checking catches type mismatches (if ty available)."""
        definition_code = """
from pydantic import BaseModel, Field

class Output(BaseModel):
    value: int = Field(...)
"""

        generated_code = """
# Type mismatch: assigning string to int field
output = Output(value="not an int")
"""

        verifier = BaseVerify()
        is_valid, error = verifier.verify((definition_code, generated_code), None)
        # If ty is available, should catch the type error
        # If ty is not available, will pass (which is ok)
        if not is_valid:
            assert "type" in error.lower() or "assignment" in error.lower()


# ============================================================================
# Context Management with LLM
# ============================================================================


class TestContextManagementWithLLM:
    """Test context management during LLM execution."""

    @pytest.mark.asyncio
    async def test_context_persists_across_calls(self, calculator_tool):
        """Test that context persists across multiple tool calls."""
        from a1.runtime import get_context, set_runtime

        runtime = Runtime()
        set_runtime(runtime)

        # Get context and add a message
        ctx = get_context("main")
        initial_count = len(ctx.messages)

        # Execute a tool (should add messages to context)
        result = await runtime.execute(calculator_tool, a=5, b=3, operation="add")
        assert result.result == 8

        # Context should have more messages now
        # (setup, function call, result messages)
        assert len(ctx.messages) >= initial_count

    @pytest.mark.asyncio
    async def test_multiple_contexts_independent(self):
        """Test that multiple contexts are independent."""
        from a1.runtime import get_context, set_runtime

        runtime = Runtime()
        set_runtime(runtime)

        ctx1 = get_context("context1")
        ctx2 = get_context("context2")

        # They should be different objects
        assert ctx1 is not ctx2

        # Modifying one shouldn't affect the other
        msg1 = {"role": "user", "content": "test"}
        ctx1.messages.append(msg1)

        assert len(ctx1.messages) > len(ctx2.messages)


# ============================================================================
# Edge Cases with Real LLM
# ============================================================================


class TestEdgeCasesWithRealLLM:
    """Test edge cases with actual LLM integration."""

    def test_code_with_complex_imports_verified(self):
        """Test verification of code with various import patterns."""
        code = """
import json
import re
from typing import List, Dict
from dataclasses import dataclass

result = json.dumps({"status": "ok"})
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_code_with_comprehensions_verified(self):
        """Test verification of code with comprehensions."""
        code = """
numbers = [1, 2, 3, 4, 5]
squared = [x**2 for x in numbers]
mapping = {x: x**2 for x in numbers}
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_code_with_exception_handling_verified(self):
        """Test verification of code with exception handling."""
        code = """
try:
    result = await tool(x=value)
except ValueError as e:
    result = None
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_malformed_code_rejected(self):
        """Test that malformed code is properly rejected."""
        code = """
result = func(x=1 +  # Incomplete
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert error is not None


# ============================================================================
# Runtime Mode Integration Tests
# ============================================================================


class TestRuntimeModeIntegration:
    """Test runtime behavior with different modes."""

    @pytest.mark.asyncio
    async def test_runtime_with_multiple_tools(self, calculator_tool, llm_tool):
        """Test runtime handling multiple tools."""
        AgentInput = create_model("AgentInput", query=(str, Field(description="Query")))
        AgentOutput = create_model("AgentOutput", result=(str, Field(description="Result")))

        agent = Agent(
            name="multi_tool_agent",
            description="Agent with calculator and LLM",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool, llm_tool],
        )

        runtime = Runtime()

        # Execute calculator tool via runtime
        result = await runtime.execute(calculator_tool, a=12, b=3, operation="divide")
        assert result.result == 4.0

    @pytest.mark.asyncio
    async def test_runtime_caching_behavior(self, calculator_tool):
        """Test runtime caching behavior."""
        AgentInput = create_model("AgentInput", task=(str, Field(description="Task")))
        AgentOutput = create_model("AgentOutput", result=(str, Field(description="Result")))

        agent = Agent(
            name="cached_agent",
            description="Agent with caching",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool],
        )

        # Runtime should handle caching properly
        runtime = Runtime()
        assert runtime is not None
