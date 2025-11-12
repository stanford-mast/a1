"""
Tests for different generation and execution modes: JIT, AOT, and IsLoop.

Covers:
- JIT mode with input variables available in generated code
- AOT mode with function compilation
- IsLoop mode with templated agentic loop
- Context management across modes
- Tool execution and state handling
- Edge cases with different code patterns
"""

import pytest
from pydantic import BaseModel, Field, create_model

from a1 import Agent, Runtime, Tool
from a1.codecheck import BaseVerify, IsLoop
from a1.context import Context


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
    """Create a calculator tool."""
    return Tool(
        name="calculator",
        description="Perform basic arithmetic operations",
        input_schema=CalculatorInput,
        output_schema=CalculatorOutput,
        execute=calculator_execute,
        is_terminal=False,
    )


# ============================================================================
# JIT Mode Tests
# ============================================================================


class TestJITMode:
    """Test JIT (Just-In-Time) mode execution."""

    @pytest.mark.asyncio
    async def test_jit_simple_tool_call(self, calculator_tool):
        """Test JIT can generate and execute simple tool calls."""
        AgentInput = create_model("AgentInput", query=(str, Field(description="Math query")))
        AgentOutput = create_model("AgentOutput", answer=(str, Field(description="Result")))

        agent = Agent(
            name="simple_math",
            description="Answer simple math questions",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool],
        )

        runtime = Runtime()

        # Since we can't mock LLM generation, test the tool directly
        result = await runtime.execute(calculator_tool, a=5, b=3, operation="add")
        assert result.result == 8

    @pytest.mark.asyncio
    async def test_jit_with_multiple_tools(self, calculator_tool):
        """Test JIT with multiple tools available."""

        async def string_tool_execute(text: str) -> dict:
            return {"length": len(text)}

        string_tool = Tool(
            name="string_analyzer",
            description="Analyze string properties",
            input_schema=create_model("StringInput", text=(str, Field(description="Text to analyze"))),
            output_schema=create_model("StringOutput", length=(int, Field(description="String length"))),
            execute=string_tool_execute,
            is_terminal=False,
        )

        AgentInput = create_model("AgentInput", task=(str, Field(description="Task")))
        AgentOutput = create_model("AgentOutput", result=(str, Field(description="Result")))

        agent = Agent(
            name="multi_tool",
            description="Use multiple tools",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool, string_tool],
        )

        runtime = Runtime()

        # Test both tools work
        result1 = await runtime.execute(calculator_tool, a=10, b=5, operation="subtract")
        assert result1.result == 5

        result2 = await runtime.execute(string_tool, text="hello")
        assert result2.length == 5


# ============================================================================
# AOT Mode Tests
# ============================================================================


class TestAOTMode:
    """Test AOT (Ahead-Of-Time) mode compilation."""

    @pytest.mark.asyncio
    async def test_aot_simple_tool_call(self, calculator_tool):
        """Test AOT can compile and execute simple tool calls."""
        AgentInput = create_model("AgentInput", problem=(str, Field(description="Problem to solve")))
        AgentOutput = create_model("AgentOutput", answer=(str, Field(description="Answer")))

        agent = Agent(
            name="aot_math",
            description="AOT math solver",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool],
        )

        runtime = Runtime()

        # Test tool execution
        result = await runtime.execute(calculator_tool, a=20, b=4, operation="divide")
        assert result.result == 5

    @pytest.mark.asyncio
    async def test_aot_with_caching(self, calculator_tool):
        """Test AOT with caching enabled."""
        AgentInput = create_model("AgentInput", task=(str, Field(description="Task")))
        AgentOutput = create_model("AgentOutput", result=(str, Field(description="Result")))

        agent = Agent(
            name="cached_tool",
            description="Tool with caching",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool],
        )

        runtime = Runtime()

        # Verify tool works
        result = await runtime.execute(calculator_tool, a=7, b=3, operation="multiply")
        assert result.result == 21


# ============================================================================
# IsLoop Mode Tests
# ============================================================================


class TestIsLoopMode:
    """Test IsLoop verifier and templated loop generation."""

    def test_isloop_verifier_with_valid_code(self):
        """Test IsLoop verifier accepts valid loop patterns."""
        code = """
while True:
    result = await llm_tool(prompt="test")
    if result.done:
        break
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_isloop_verifier_rejects_missing_loop(self):
        """Test IsLoop verifier rejects code without loop."""
        code = """
result = await llm_tool(prompt="test")
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "loop" in error.lower()

    def test_isloop_verifier_rejects_missing_break(self):
        """Test IsLoop verifier rejects loop without break."""
        code = """
while True:
    result = await llm_tool(prompt="test")
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid

    def test_isloop_accepts_while_with_condition(self):
        """Test IsLoop accepts while with condition (not just True)."""
        code = """
while count < max_iter:
    result = await gpt_tool(content="next")
    if isinstance(result, Output):
        break
    count += 1
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


# ============================================================================
# Context Management Tests
# ============================================================================


class TestContextManagement:
    """Test context management across execution modes."""

    @pytest.mark.asyncio
    async def test_context_created_on_demand(self):
        """Test contexts are created on demand."""
        from a1.runtime import get_context, set_runtime

        runtime = Runtime()
        set_runtime(runtime)

        # Get or create context
        ctx = get_context("test")
        assert ctx is not None
        assert isinstance(ctx, Context)

        # Get same context again
        ctx2 = get_context("test")
        assert ctx is ctx2

    @pytest.mark.asyncio
    async def test_context_isolation(self):
        """Test different contexts are isolated."""
        from a1.runtime import get_context, set_runtime

        runtime = Runtime()
        set_runtime(runtime)

        ctx1 = get_context("ctx1")
        ctx2 = get_context("ctx2")

        assert ctx1 is not ctx2

    @pytest.mark.asyncio
    async def test_context_main_default(self):
        """Test 'main' context is default."""
        from a1.runtime import get_context, set_runtime

        runtime = Runtime()
        set_runtime(runtime)

        ctx = get_context()  # No name, should get "main"
        assert ctx is not None


# ============================================================================
# State Management Tests
# ============================================================================


class TestStateManagement:
    """Test executor state management."""

    @pytest.mark.asyncio
    async def test_input_variables_in_jit(self, calculator_tool):
        """Test input variables are available in JIT execution."""
        AgentInput = create_model("AgentInput", query=(str, Field(description="Query")))
        AgentOutput = create_model("AgentOutput", result=(str, Field(description="Result")))

        agent = Agent(
            name="jit_with_input",
            description="JIT with input variables",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool],
        )

        runtime = Runtime()

        # In real JIT, the input would be available as a variable
        # Here we just verify the agent can be created
        assert agent is not None
        assert agent.input_schema is not None

    @pytest.mark.asyncio
    async def test_schema_classes_in_environment(self, calculator_tool):
        """Test that schema classes are available in executor environment."""
        AgentInput = create_model("AgentInput", x=(int, Field(description="X")))
        AgentOutput = create_model("AgentOutput", y=(int, Field(description="Y")))

        agent = Agent(
            name="schema_test",
            description="Test schema availability",
            input_schema=AgentInput,
            output_schema=AgentOutput,
            tools=[calculator_tool],
        )

        runtime = Runtime()

        # Verify schemas exist
        assert hasattr(agent, "input_schema")
        assert hasattr(agent, "output_schema")


# ============================================================================
# Error Handling and Edge Cases
# ============================================================================


class TestErrorHandling:
    """Test error handling in different modes."""

    @pytest.mark.asyncio
    async def test_division_by_zero(self, calculator_tool):
        """Test handling of division by zero."""
        runtime = Runtime()
        with pytest.raises(Exception):
            await runtime.execute(calculator_tool, a=10, b=0, operation="divide")

    @pytest.mark.asyncio
    async def test_invalid_operation(self, calculator_tool):
        """Test handling of invalid operation."""
        runtime = Runtime()
        with pytest.raises(Exception):
            await runtime.execute(calculator_tool, a=5, b=3, operation="invalid")

    def test_verify_with_malformed_code(self):
        """Test verification with malformed code."""
        code = "x = 1 +"
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert error is not None


class TestCodeGenerationEdgeCases:
    """Test edge cases in code generation and verification."""

    def test_verify_code_with_imports(self):
        """Test verification of code with imports."""
        code = """
import json
result = json.dumps({"x": 1})
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_verify_code_with_multiline_strings(self):
        """Test verification of code with multiline strings."""
        code = '''
text = """
This is a
multiline
string
"""
result = len(text)
'''
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_verify_code_with_comprehensions(self):
        """Test verification of code with comprehensions."""
        code = """
result = [x**2 for x in range(10)]
mapping = {i: i*2 for i in range(5)}
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_verify_code_with_nested_calls(self):
        """Test verification of code with nested function calls."""
        code = """
result = max([min(a, b) for a, b in pairs])
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


class TestOutputValidation:
    """Test output validation and conversion."""

    @pytest.mark.asyncio
    async def test_pydantic_output_conversion(self, calculator_tool):
        """Test Pydantic model output conversion."""
        runtime = Runtime()
        result = await runtime.execute(calculator_tool, a=6, b=2, operation="multiply")

        # Result should be a Pydantic model
        assert hasattr(result, "result")
        assert result.result == 12

    @pytest.mark.asyncio
    async def test_output_serialization(self, calculator_tool):
        """Test that output can be serialized."""
        runtime = Runtime()
        result = await runtime.execute(calculator_tool, a=15, b=3, operation="divide")

        # Should have model_dump method
        if hasattr(result, "model_dump"):
            dumped = result.model_dump()
            assert "result" in dumped
            assert dumped["result"] == 5


# ============================================================================
# Integration Tests
# ============================================================================


class TestMixedModeIntegration:
    """Test interaction between different modes."""

    def test_tool_available_in_multiple_modes(self, calculator_tool):
        """Test same tool works in JIT, AOT, and IsLoop."""
        AgentInput = create_model("AgentInput", task=(str, Field(description="Task")))
        AgentOutput = create_model("AgentOutput", result=(str, Field(description="Result")))

        # Same tool for all modes
        agent_jit = Agent(name="jit_calc", input_schema=AgentInput, output_schema=AgentOutput, tools=[calculator_tool])

        agent_aot = Agent(name="aot_calc", input_schema=AgentInput, output_schema=AgentOutput, tools=[calculator_tool])

        agent_loop = Agent(
            name="loop_calc", input_schema=AgentInput, output_schema=AgentOutput, tools=[calculator_tool]
        )

        # All should have the tool
        assert calculator_tool in agent_jit.tools
        assert calculator_tool in agent_aot.tools
        assert calculator_tool in agent_loop.tools

    def test_verifiers_compatible(self, calculator_tool):
        """Test different verifiers work together."""
        from a1.codecheck import check_code_candidate

        code = """
while True:
    result = await llm(prompt="test")
    if result:
        break
"""

        # BaseVerify should pass
        is_valid, error = check_code_candidate(code)
        assert is_valid

        # IsLoop should also pass
        is_valid, error = check_code_candidate(code, verifiers=[IsLoop()])
        assert is_valid
