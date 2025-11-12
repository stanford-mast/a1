"""
Tests for code cost estimation module.
"""

from pydantic import BaseModel

from a1 import Agent, tool
from a1.cfg_builder import BasicBlock
from a1.codecost import (
    BaseCost,
    compute_block_cost,
    compute_code_cost,
)


class TestComputeCodeCost:
    """Test code cost computation."""

    def test_simple_tool_call(self):
        """Test cost for simple tool call."""
        code = """
result = await llm(prompt="test")
"""
        cost = compute_code_cost(code, {"llm": 10.0})
        assert cost == 10.0

    def test_multiple_tool_calls(self):
        """Test cost for multiple tool calls."""
        code = """
x = await search(query="test")
y = await llm(prompt="analyze")
"""
        costs = {"search": 5.0, "llm": 10.0}
        cost = compute_code_cost(code, costs)
        assert cost == 15.0

    def test_loop_multiplier(self):
        """Test cost multiplier for loops."""
        code = """
for item in items:
    result = await llm(prompt=item)
"""
        cost = compute_code_cost(code, {"llm": 10.0})
        # Base cost 10.0 * loop multiplier 10 = 100.0
        assert cost == 100.0

    def test_nested_loops(self):
        """Test cost for nested loops."""
        code = """
for i in range(10):
    for j in range(5):
        result = await tool(i, j)
"""
        cost = compute_code_cost(code, {"tool": 1.0})
        # Base cost 1.0 * 10^2 = 100.0
        assert cost == 100.0

    def test_comprehension_cost(self):
        """Test cost for list comprehension."""
        code = """
results = [await tool(x) for x in items]
"""
        cost = compute_code_cost(code, {"tool": 1.0})
        # Comprehension depth 1 = multiplier 10
        assert cost == 10.0

    def test_nested_comprehension(self):
        """Test cost for nested comprehension."""
        code = """
results = [await tool(x, y) for x in items for y in x.subitems]
"""
        cost = compute_code_cost(code, {"tool": 1.0})
        # Two generators = depth 2 = multiplier 100
        assert cost == 100.0

    def test_mixed_loop_comprehension(self):
        """Test cost for loop containing comprehension."""
        code = """
for item in items:
    results = [await tool(x) for x in item.children]
"""
        cost = compute_code_cost(code, {"tool": 1.0})
        # Loop depth 1 + comprehension depth 1 = total depth 2
        assert cost == 100.0

    def test_unknown_tool_zero_cost(self):
        """Test that unknown tools have zero cost."""
        code = """
result = await unknown_tool(param="value")
"""
        cost = compute_code_cost(code, {})
        assert cost == 0.0

    def test_syntax_error_fallback(self):
        """Test fallback to code length on syntax error."""
        code = "this is not valid python!!!"
        cost = compute_code_cost(code)
        assert cost == float(len(code))

    def test_branching_takes_max_path(self):
        """Test that branching takes maximum cost path."""
        code = """
if condition:
    await expensive_tool()  
else:
    await cheap_tool()
"""
        costs = {"expensive_tool": 100.0, "cheap_tool": 1.0}
        cost = compute_code_cost(code, costs)
        # Should take expensive path
        assert cost == 100.0


class TestBaseCost:
    """Test BaseCost strategy class."""

    def test_simple_cost_creation(self):
        """Test creating BaseCost with custom costs."""
        strategy = BaseCost(tool_costs={"custom_tool": 50.0}, loop_multiplier=5.0)

        assert strategy.tool_costs["custom_tool"] == 50.0
        assert strategy.loop_multiplier == 5.0

    def test_simple_cost_compute(self):
        """Test computing cost with BaseCost."""

        # Create mock agent
        class Input(BaseModel):
            x: int

        @tool(name="test_tool")
        async def test_tool(x: int) -> int:
            return x * 2

        agent = Agent(name="test", description="Test agent", input_schema=Input, output_schema=Input, tools=[test_tool])

        strategy = BaseCost(tool_costs={"test_tool": 10.0})

        code = """
result = await test_tool(x=5)
"""
        cost = strategy.compute_cost(code, agent)
        assert cost == 10.0

    def test_simple_cost_with_agent_tools(self):
        """Test that BaseCost uses agent tools for cost map."""

        class Input(BaseModel):
            x: int

        @tool(name="tool_a")
        async def tool_a(x: int) -> int:
            return x

        @tool(name="tool_b")
        async def tool_b(x: int) -> int:
            return x

        agent = Agent(name="test", description="Test", input_schema=Input, output_schema=Input, tools=[tool_a, tool_b])

        strategy = BaseCost()

        code = """
await tool_a(x=1)
await tool_b(x=2)
"""
        cost = strategy.compute_cost(code, agent)
        # Unknown tools default to 1.0 each
        assert cost == 2.0


class TestComputeBlockCost:
    """Test computing cost for individual blocks."""

    def test_block_with_no_calls(self):
        """Test block with no function calls."""
        block = BasicBlock(bid=1)
        cost = compute_block_cost(block, loop_depth=0)
        assert cost == 0.0

    def test_block_with_calls(self):
        """Test block with function calls."""
        block = BasicBlock(bid=1)
        # Add calls: (func_name, call_node, comp_depth)
        block.calls.append(("llm", None, 0))
        block.calls.append(("search", None, 0))

        costs = {"llm": 10.0, "search": 5.0}
        cost = compute_block_cost(block, loop_depth=0, tool_name_to_cost=costs)
        assert cost == 15.0

    def test_block_in_loop(self):
        """Test block cost when in a loop."""
        block = BasicBlock(bid=1)
        block.calls.append(("tool", None, 0))

        cost = compute_block_cost(block, loop_depth=1, tool_name_to_cost={"tool": 1.0})
        # Base 1.0 * 10^1 = 10.0
        assert cost == 10.0

    def test_block_with_comprehension_depth(self):
        """Test block cost with comprehension depth."""
        block = BasicBlock(bid=1)
        block.calls.append(("tool", None, 2))  # comp_depth=2

        cost = compute_block_cost(block, loop_depth=0, tool_name_to_cost={"tool": 1.0})
        # Base 1.0 * 10^2 = 100.0
        assert cost == 100.0

    def test_block_combined_depths(self):
        """Test block with both loop and comprehension depth."""
        block = BasicBlock(bid=1)
        block.calls.append(("tool", None, 1))  # comp_depth=1

        cost = compute_block_cost(block, loop_depth=1, tool_name_to_cost={"tool": 1.0})
        # Base 1.0 * 10^(1+1) = 100.0
        assert cost == 100.0


# ============================================================================
# LLM-Specific Cost Tests
# ============================================================================


class TestLLMCosts:
    """Test proper LLM cost calculation."""

    def test_single_llm_call(self):
        """Test cost of single LLM call."""
        code = """
result = await llm_groq_openai_gpt_oss_20b(prompt="test")
"""
        cost = compute_code_cost(code, {"llm_groq_openai_gpt_oss_20b": 10.0})
        assert cost == 10.0

    def test_llm_in_loop(self):
        """Test LLM call inside loop (scales exponentially)."""
        code = """
for iteration in range(max_iterations):
    result = await llm_groq_openai_gpt_oss_20b(prompt=instruction)
    if result.done:
        break
"""
        # Base cost 10.0 * loop multiplier 10 = 100.0
        cost = compute_code_cost(code, {"llm_groq_openai_gpt_oss_20b": 10.0})
        assert cost == 100.0

    def test_multiple_llm_calls_sequential(self):
        """Test multiple sequential LLM calls."""
        code = """
result1 = await llm_tool(prompt="first")
result2 = await llm_tool(prompt="second")
result3 = await llm_tool(prompt="third")
"""
        cost = compute_code_cost(code, {"llm_tool": 10.0})
        # 3 * 10.0 = 30.0
        assert cost == 30.0

    def test_llm_with_calculator_in_loop(self):
        """Test agentic loop with LLM and calculator calls."""
        code = """
while iteration < max_iterations:
    llm_output = await llm(
        content=instruction,
        tools=[calculator],
        output_schema=Output
    )
    
    if isinstance(llm_output, Output):
        break
    
    calc_result = await calculator(a=llm_output.a, b=llm_output.b, operation="add")
    iteration += 1
"""
        costs = {"llm": 10.0, "calculator": 1.0}
        cost = compute_code_cost(code, costs)
        # Loop depth 1: (10.0 + 1.0) * 10 = 110.0
        assert cost == 110.0

    def test_llm_in_nested_loops(self):
        """Test LLM in nested loops."""
        code = """
for i in range(n_outer):
    for j in range(n_inner):
        result = await llm(prompt=f"iteration {i},{j}")
"""
        cost = compute_code_cost(code, {"llm": 10.0})
        # Nested loops: depth 2, so 10.0 * 10^2 = 1000.0
        assert cost == 1000.0

    def test_llm_with_branching(self):
        """Test LLM costs with if/else branches."""
        code = """
if complex_problem:
    result = await expensive_llm(prompt=problem)
else:
    result = await cheap_llm(prompt=problem)
"""
        costs = {"expensive_llm": 100.0, "cheap_llm": 10.0}
        cost = compute_code_cost(code, costs)
        # Takes the expensive path
        assert cost == 100.0


# ============================================================================
# Loop Cost Multiplier Tests
# ============================================================================


class TestLoopCostMultipliers:
    """Test loop cost multiplier calculations."""

    def test_loop_multiplier_1_depth(self):
        """Test loop multiplier for depth 1."""
        code = """
for item in items:
    await tool()
"""
        cost = compute_code_cost(code, {"tool": 10.0})
        # 10.0 * 10^1 = 100.0
        assert cost == 100.0

    def test_loop_multiplier_2_depth(self):
        """Test loop multiplier for depth 2."""
        code = """
for i in range(10):
    for j in range(10):
        await tool()
"""
        cost = compute_code_cost(code, {"tool": 10.0})
        # 10.0 * 10^2 = 1000.0
        assert cost == 1000.0

    def test_loop_multiplier_3_depth(self):
        """Test loop multiplier for depth 3."""
        code = """
for i in range(10):
    for j in range(10):
        for k in range(10):
            await tool()
"""
        cost = compute_code_cost(code, {"tool": 1.0})
        # 1.0 * 10^3 = 1000.0
        assert cost == 1000.0

    def test_while_loop_cost(self):
        """Test while loop has same multiplier as for loop."""
        code = """
while condition:
    result = await tool()
"""
        cost = compute_code_cost(code, {"tool": 10.0})
        # 10.0 * 10^1 = 100.0
        assert cost == 100.0

    def test_loop_with_multiple_tools(self):
        """Test loop with multiple different tools."""
        code = """
for i in range(10):
    result1 = await llm(prompt="analyze")
    result2 = await calculator(a=1, b=2)
    result3 = await search(query="info")
"""
        costs = {"llm": 10.0, "calculator": 1.0, "search": 5.0}
        cost = compute_code_cost(code, costs)
        # (10.0 + 1.0 + 5.0) * 10^1 = 160.0
        assert cost == 160.0


# ============================================================================
# Complex Scenario Tests
# ============================================================================


class TestComplexScenarios:
    """Test complex real-world scenarios."""

    def test_isloop_agentic_pattern(self):
        """Test IsLoop agentic loop cost."""
        code = """
iteration = 0
while iteration < max_iterations:
    output = await llm_groq_openai_gpt_oss_20b(
        content=instruction if iteration == 0 else "Continue",
        tools=[calculator],
        context=context,
        output_schema=AgentOutput
    )
    
    if isinstance(output, AgentOutput):
        break
    
    iteration += 1
"""
        cost = compute_code_cost(code, {"llm_groq_openai_gpt_oss_20b": 10.0, "calculator": 1.0})
        # While loop depth 1: 10.0 * 10^1 = 100.0
        assert cost == 100.0

    def test_multi_step_reasoning(self):
        """Test multi-step reasoning with LLM calls."""
        code = """
# Step 1: Analyze problem
analysis = await llm(prompt="Analyze this problem")

# Step 2: Generate solution
for approach in num_approaches:
    solution = await llm(prompt=f"Try approach {approach}")
    
    # Step 3: Evaluate solution
    score = await evaluator(solution=solution)
"""
        costs = {"llm": 10.0, "evaluator": 5.0}
        cost = compute_code_cost(code, costs)
        # Sequential: 10.0 + (10.0 + 5.0) * 10 = 10.0 + 150.0 = 160.0
        assert cost == 160.0

    def test_tool_cost_accumulation(self):
        """Test that tool costs properly accumulate."""
        code = """
step1 = await tool_a(x=1)
step2 = await tool_b(x=2)
step3 = await tool_c(x=3)
step4 = await tool_d(x=4)
"""
        costs = {"tool_a": 1.0, "tool_b": 2.0, "tool_c": 3.0, "tool_d": 4.0}
        cost = compute_code_cost(code, costs)
        # 1.0 + 2.0 + 3.0 + 4.0 = 10.0
        assert cost == 10.0

    def test_expensive_llm_variants(self):
        """Test costs for different LLM providers."""
        code = """
result = await llm_provider(prompt="test")
"""
        # Different providers have different costs
        llm_costs = {
            "gpt4": 50.0,
            "gpt4o": 10.0,
            "gpt4_mini": 5.0,
            "claude_opus": 30.0,
            "claude_haiku": 3.0,
            "groq_oss": 1.0,
        }

        for provider, expected_cost in llm_costs.items():
            test_code = code.replace("llm_provider", f"llm_{provider}")
            cost = compute_code_cost(test_code, {f"llm_{provider}": expected_cost})
            assert cost == expected_cost, f"Failed for {provider}: expected {expected_cost}, got {cost}"
