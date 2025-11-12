"""
Code cost computation for ranking code candidates.

Provides functions to compute cost/quality scores for generated code.
Lower costs are better. Cost is based on estimated latency of tool calls
considering loop nesting levels via CFG traversal.

Loop depth calculation:
- For/while loops: Detected via CFG blocks containing ast.For/ast.While
- Comprehensions: Tracked via comprehension_depth in BasicBlock.calls
- Multi-generator comprehensions: depth = number of generators
- Total depth = loop_depth (from CFG) + comprehension_depth
- Cost multiplier = LOOP_MULTIPLIER ^ total_depth

Examples:
- await tool_call("x"): 10s (depth 0)
- for x in items: await tool_call("x"): 100s (loop_depth 1)
- [await tool_call(x) for x in items]: 100s (comp_depth 1)
- for x in items: [await tool_call(y) for y in x.children]: 1000s (depth 1+1=2)
- [await tool_call(x) for x in items for y in subitems]: 1000s (comp_depth 2)
"""

import ast
import logging
from typing import Any

from .cfg_builder import BasicBlock, CFGBuilder

logger = logging.getLogger(__name__)


# Tool latency estimates in seconds
TOOL_LATENCIES: dict[str, float] = {
    # Default tool costs
    "llm": 10.0,
    "search": 5.0,
    "read": 1.0,
    "write": 1.0,
    # Any tool not in this dict defaults to 0
}

# Loop multiplier - assume each loop runs 10 times
LOOP_MULTIPLIER = 10


def compute_block_cost(
    block: BasicBlock, loop_depth: int = 0, tool_name_to_cost: dict[str, float] | None = None
) -> float:
    """
    Compute cost for all tool calls in a basic block.

    Args:
        block: Basic block with tool calls
        loop_depth: Current loop nesting depth (from for/while loops)
        tool_name_to_cost: Mapping from tool function names to base costs

    Returns:
        Total cost for this block in seconds
    """
    if tool_name_to_cost is None:
        tool_name_to_cost = TOOL_LATENCIES

    total_cost = 0.0

    for func_name, call_node, comp_depth in block.calls:
        # Total depth = loop depth from for/while + comprehension depth
        total_depth = loop_depth + comp_depth
        multiplier = LOOP_MULTIPLIER**total_depth

        # Look up base latency based on containing the name in it
        base_latency = next((cost for tool_name, cost in tool_name_to_cost.items() if tool_name in func_name), 0.0)
        cost = base_latency * multiplier
        total_cost += cost

    return total_cost


def traverse_cfg_for_cost(
    blocks: dict[int, BasicBlock],
    start_block: BasicBlock,
    tool_name_to_cost: dict[str, float] | None = None,
    loop_depth: int = 0,
    visited: set[int] | None = None,
    loop_headers: set[int] | None = None,
) -> float:
    """
    Traverse CFG and compute total estimated cost.

    Uses depth-first traversal with loop detection. When entering a loop body
    (detected via back-edges), increases loop depth.

    Args:
        blocks: All basic blocks
        start_block: Block to start traversal from
        tool_name_to_cost: Mapping from tool function names to base costs
        loop_depth: Current loop nesting depth
        visited: Set of visited block IDs (to prevent infinite loops)
        loop_headers: Set of block IDs that are loop headers (have back-edges)

    Returns:
        Total estimated latency cost in seconds
    """
    if visited is None:
        visited = set()

    if tool_name_to_cost is None:
        tool_name_to_cost = TOOL_LATENCIES

    if loop_headers is None:
        # Detect loop headers - blocks with for/while statements
        loop_headers = set()
        for bid, block in blocks.items():
            if block.stmts:
                for stmt in block.stmts:
                    if isinstance(stmt, (ast.For, ast.While)):
                        loop_headers.add(bid)
                        break

    # Prevent infinite recursion
    if start_block.bid in visited:
        return 0.0

    visited.add(start_block.bid)

    # Cost for this block
    block_cost = compute_block_cost(start_block, loop_depth, tool_name_to_cost)

    # If no successors, return block cost
    if not start_block.next:
        return block_cost

    # Check if current block is a loop header
    is_loop_header = start_block.bid in loop_headers

    if is_loop_header:
        # Loop header: first successor is after-loop, rest are loop body
        after_loop_cost = 0.0
        loop_body_cost = 0.0

        for idx, next_bid in enumerate(start_block.next):
            next_block = blocks[next_bid]

            if next_bid not in visited:
                if idx == 0:
                    # After-loop edge: keep same depth
                    successor_cost = traverse_cfg_for_cost(
                        blocks, next_block, tool_name_to_cost, loop_depth, visited.copy(), loop_headers
                    )
                    after_loop_cost = successor_cost
                else:
                    # Loop body edge: increment depth
                    next_depth = loop_depth + 1
                    successor_cost = traverse_cfg_for_cost(
                        blocks, next_block, tool_name_to_cost, next_depth, visited.copy(), loop_headers
                    )
                    loop_body_cost = max(loop_body_cost, successor_cost)

        return block_cost + loop_body_cost + after_loop_cost

    # For non-loop blocks, traverse all successor paths and take maximum cost path
    max_successor_cost = 0.0
    for next_bid in start_block.next:
        next_block = blocks[next_bid]

        if next_bid not in visited:
            successor_cost = traverse_cfg_for_cost(
                blocks, next_block, tool_name_to_cost, loop_depth, visited.copy(), loop_headers
            )
            max_successor_cost = max(max_successor_cost, successor_cost)

    return block_cost + max_successor_cost


def compute_code_cost(code: str, tool_costs: dict[str, float] | None = None) -> float:
    """
    Compute cost/score for a code candidate based on estimated latency.
    Lower is better.

    Cost calculation:
    - Builds CFG from AST using cfg_builder
    - Finds all tool calls in each basic block
    - Applies base latency for each tool type
    - Multiplies by loop nesting level detected via CFG (10x per loop level)
    - Sums total estimated latency via CFG traversal

    Args:
        code: Generated Python code
        tool_costs: Dictionary mapping tool names to base costs

    Returns:
        Cost score in seconds (estimated latency, lower is better)
    """
    try:
        tree = ast.parse(code)

        # Build CFG
        cfg_builder = CFGBuilder()
        start_block, blocks = cfg_builder.build(tree)

        # Use provided tool costs or defaults
        cost_map = tool_costs or TOOL_LATENCIES

        # Compute cost from start block
        total_cost = traverse_cfg_for_cost(blocks, start_block, cost_map)

        # Find orphaned blocks (not reachable from start) that have tool calls
        reachable = set()

        def mark_reachable(bid):
            if bid in reachable or bid not in blocks:
                return
            reachable.add(bid)
            for next_bid in blocks[bid].next:
                mark_reachable(next_bid)

        mark_reachable(start_block.bid)

        # For any unreachable blocks with tool calls, traverse from them too
        for bid, block in blocks.items():
            if bid not in reachable and block.calls:
                cost_from_orphan = traverse_cfg_for_cost(blocks, block, cost_map, visited=set())
                total_cost = max(total_cost, cost_from_orphan)

        return total_cost

    except SyntaxError as e:
        logger.warning(f"Syntax error in code cost analysis: {e}")
        # Fall back to code length as cost
        return float(len(code))
    except Exception as e:
        logger.warning(f"Error in code cost analysis: {e}")
        # Fall back to code length as cost
        return float(len(code))


# Base class for cost strategies
class Cost:
    """
    Base class for cost estimation strategies.

    Estimates the cost/complexity of generated code to rank candidates.
    """

    def compute_cost(self, code, agent: Any) -> float:
        """
        Compute estimated cost for code.

        Args:
            code: Generated Python code (str) or tuple of (definition_code, generated_code)
            agent: Agent specification

        Returns:
            Cost estimate (lower is better)
        """
        raise NotImplementedError

    def _extract_code(self, code):
        """Extract generated_code from code (str or tuple)."""
        if isinstance(code, tuple):
            # If tuple, use only the generated_code part (second element)
            return code[1] if len(code) > 1 else code[0]
        return code


class BaseCost(Cost):
    """
    Base cost estimation implementation based on:
    - Number of tool calls
    - Loop nesting depth
    - Specific tool costs

    Uses CFG-based analysis for accurate loop depth tracking.

    Args:
        tool_costs: Dict mapping tool names to base costs
        loop_multiplier: Multiplier for each loop nesting level
    """

    def __init__(self, tool_costs: dict[str, float] | None = None, loop_multiplier: float = 10.0):
        self.tool_costs = tool_costs or TOOL_LATENCIES.copy()
        self.loop_multiplier = loop_multiplier

        # Update global loop multiplier
        global LOOP_MULTIPLIER
        LOOP_MULTIPLIER = loop_multiplier

    def compute_cost(self, code, agent: Any) -> float:
        """Compute cost based on tool calls and loops using CFG analysis."""
        # Extract just the generated code
        generated_code = self._extract_code(code)

        # Extract tool costs from agent if available
        tool_cost_map = self.tool_costs.copy()
        if hasattr(agent, "get_all_tools"):
            for tool in agent.get_all_tools():
                if tool.name not in tool_cost_map:
                    # Default cost for unknown tools
                    tool_cost_map[tool.name] = 1.0

        return compute_code_cost(generated_code, tool_cost_map)


class QuantitativeCriteria(Cost):
    """
    LLM-based cost estimation using natural language criteria.

    Prompts an LLM to score code against quantitative criteria (returns number).
    Supports multiple samples with parallel execution and aggregation (min/max/avg/med).

    Args:
        expression: Natural language cost criteria (e.g., "How complex is this code? (0=simple, 10=very complex)")
        llm: Tool instance for LLM calls (e.g., LLM("gpt-4.1-mini"))
        min: Minimum valid score (default: 0)
        max: Maximum valid score (default: 10)
        agg: Aggregation method: 'min', 'max', 'avg', 'med' (default: 'avg')
        num_samples: Number of parallel LLM calls to make (default: 1)
        min_samples_for_aggregation: Minimum successful samples needed (default: 1)
    """

    def __init__(
        self,
        expression: str,
        llm: Any,  # Tool
        min: float = 0.0,
        max: float = 10.0,
        agg: str = "avg",  # 'min', 'max', 'avg', 'med'
        num_samples: int = 1,
        min_samples_for_aggregation: int = 1,
    ):
        self.expression = expression
        self.llm = llm
        self.min = min
        self.max = max
        self.agg = agg
        self.num_samples = num_samples
        self.min_samples_for_aggregation = min_samples_for_aggregation

    def compute(self, code, agent: Any) -> float:
        """
        Compute cost using LLM-based quantitative criteria with optional sampling.

        Returns:
            Aggregated cost score
        """
        # Extract just the generated code
        generated_code = self._extract_code(code)

        import asyncio
        import statistics

        async def _compute_async():
            # Create evaluation prompt
            prompt = f"""Evaluate the following Python code and provide a numeric score:

Criteria: {self.expression}

Valid range: {self.min} to {self.max} (inclusive)

Code:
```python
{generated_code}
```

Return ONLY a number between {self.min} and {self.max} (no explanation, no units).
"""

            # Run multiple samples in parallel if requested
            if self.num_samples <= 1:
                # Single sample
                result = await self.llm(content=prompt)
                # Handle both string and LLMOutput responses
                if isinstance(result, str):
                    response = result.strip()
                elif hasattr(result, "content"):
                    response = (result.content or "").strip()
                else:
                    response = result.get("content", "").strip()

                try:
                    score = float(response)
                    if self.min <= score <= self.max:
                        return score
                    else:
                        # Out of range - clamp to bounds
                        return max(self.min, min(self.max, score))
                except ValueError:
                    # Invalid response - return midpoint
                    return (self.min + self.max) / 2

            # Multiple samples - run in parallel
            tasks = [self.llm(content=prompt) for _ in range(self.num_samples)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Parse results and collect valid scores
            valid_scores = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                # Handle both string and LLMOutput responses
                if isinstance(result, str):
                    response = result.strip()
                elif hasattr(result, "content"):
                    response = (result.content or "").strip()
                else:
                    response = result.get("content", "").strip()
                try:
                    score = float(response)
                    if self.min <= score <= self.max:
                        valid_scores.append(score)
                except ValueError:
                    continue

            # Check if we have enough valid samples
            if len(valid_scores) < self.min_samples_for_aggregation:
                # Not enough valid samples - return midpoint
                return (self.min + self.max) / 2

            # Aggregate scores based on strategy
            if self.agg == "min":
                return min(valid_scores)
            elif self.agg == "max":
                return max(valid_scores)
            elif self.agg == "med":
                return statistics.median(valid_scores)
            else:  # avg (default)
                return statistics.mean(valid_scores)

        # Run async computation - handle both sync and async contexts
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # We are in an async context, run in thread pool
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _compute_async())
                return future.result()
        except RuntimeError:
            # No event loop running, use asyncio.run
            return asyncio.run(_compute_async())


__all__ = [
    "Cost",
    "BaseCost",
    "QuantitativeCriteria",
    "compute_code_cost",
    "compute_block_cost",
    "traverse_cfg_for_cost",
    "TOOL_LATENCIES",
    "LOOP_MULTIPLIER",
]
