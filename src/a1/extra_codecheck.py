"""
Extended code verification strategies.

Additional verifiers for specific use cases:
- IsFunction: AOT function signature validation
- QualitativeCriteria: LLM-based qualitative code evaluation
"""

import ast
import asyncio
import logging
from typing import Any

from .codecheck import Verify

logger = logging.getLogger(__name__)


class IsFunction(Verify):
    """
    Verify that code contains exactly one async function with correct signature.

    Used by AOT mode to ensure generated code:
    1. Contains exactly one async function (not a stub)
    2. Function name matches agent.name
    3. Parameters match agent.input_schema fields (names and order)
    4. Return type matches agent.output_schema
    5. No code exists outside the function definition

    Args passed via kwargs:
        agent: Agent instance with name, input_schema, output_schema
    """

    def verify(self, code, **kwargs) -> tuple[bool, str | None]:
        """Check if code has exactly one async function with correct signature."""
        # Extract agent from kwargs
        agent = kwargs.get("agent")

        # Extract just the generated code (not definition code)
        generated_code = self._extract_code(code)

        # Parse code
        try:
            tree = ast.parse(generated_code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Find async function definitions at module level, excluding stubs
        func_defs = []
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef):
                # Check if it's a stub (raises NotImplementedError)
                is_stub = False
                for stmt in node.body:
                    if isinstance(stmt, ast.Raise):
                        if isinstance(stmt.exc, ast.Call):
                            if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                                is_stub = True
                                break

                # Only count non-stub functions
                if not is_stub:
                    func_defs.append(node)

        if len(func_defs) == 0:
            return False, "No async function implementation found. AOT mode requires a function (stubs don't count)."

        # Check that there's only one function definition
        if len(func_defs) > 1:
            func_names = [f.name for f in func_defs]
            return False, f"Multiple function definitions found: {func_names}. AOT mode requires exactly one function."

        # Get the single function
        func_def = func_defs[0]

        # Check that the function is async (should always be true here)
        if not isinstance(func_def, ast.AsyncFunctionDef):
            return False, "Function must be async (use 'async def')"

        # Check that there's no code outside the function
        non_func_nodes = [node for node in tree.body if not isinstance(node, ast.AsyncFunctionDef)]
        # Allow imports at the top
        non_import_nodes = [node for node in non_func_nodes if not isinstance(node, (ast.Import, ast.ImportFrom))]
        if non_import_nodes:
            return (
                False,
                "Found code outside the function definition. AOT mode requires ONLY the function, no extra code.",
            )

        # Extract metadata
        func_name = func_def.name
        func_args = [arg.arg for arg in func_def.args.args]

        # Get return annotation if present
        return_annotation = None
        if func_def.returns:
            if isinstance(func_def.returns, ast.Name):
                return_annotation = func_def.returns.id
            elif isinstance(func_def.returns, ast.Constant):
                return_annotation = func_def.returns.value

        # Log extracted metadata
        logger.debug(f"Function metadata - name: {func_name}, args: {func_args}, return: {return_annotation}")

        # Validate against agent schema if provided
        if agent:
            # Validate function name
            if func_name != agent.name:
                return False, f"Function name '{func_name}' doesn't match agent name '{agent.name}'"

            # Validate parameters match input schema
            if hasattr(agent.input_schema, "model_fields"):
                expected_params = list(agent.input_schema.model_fields.keys())
                if func_args != expected_params:
                    return False, f"Function parameters {func_args} don't match input schema fields {expected_params}"

            # Validate return type matches output schema
            if hasattr(agent.output_schema, "__name__"):
                expected_return = agent.output_schema.__name__
                if return_annotation != expected_return:
                    return (
                        False,
                        f"Return type annotation '{return_annotation}' doesn't match output schema '{expected_return}'",
                    )

        return True, None


class QualitativeCriteria(Verify):
    """
    LLM-based verification using natural language criteria.

    Prompts an LLM to evaluate code against qualitative criteria (returns boolean).
    Supports multiple samples with parallel execution and majority voting.

    Args:
        expression: Natural language criteria description (e.g., "Code is readable and well-structured")
        llm: Tool instance for LLM calls (e.g., LLM("gpt-4.1-mini"))
        num_samples: Number of parallel LLM calls to make (default: 1)
        min_samples_for_aggregation: Minimum successful samples needed (default: 1)
        min_pass: Minimum samples that must return True for overall True (default: 1)
    """

    def __init__(
        self,
        expression: str,
        llm: Any,  # Tool
        num_samples: int = 1,
        min_samples_for_aggregation: int = 1,
        min_pass: int = 1,
    ):
        self.expression = expression
        self.llm = llm
        self.num_samples = num_samples
        self.min_samples_for_aggregation = min_samples_for_aggregation
        self.min_pass = min_pass

    def verify(self, code: str, agent: Any) -> tuple[bool, str | None]:
        """
        Verify code using LLM-based qualitative criteria with optional sampling.

        Returns:
            Tuple of (passes_criteria, error_message)
        """

        async def _verify_async():
            # Create evaluation prompt
            prompt = f"""Evaluate the following Python code against this criteria:

Criteria: {self.expression}

Code:
```python
{code}
```

Return ONLY "true" or "false" (lowercase, no quotes, no explanation).
"""

            # Run multiple samples in parallel if requested
            if self.num_samples <= 1:
                # Single sample
                result = await self.llm(content=prompt)
                # Handle both string and LLMOutput responses
                if isinstance(result, str):
                    response = result.strip().lower()
                elif hasattr(result, "content"):
                    response = (result.content or "").strip().lower()
                else:
                    response = result.get("content", "").strip().lower()

                if response == "true":
                    return True, None
                elif response == "false":
                    return False, f"Code does not meet criteria: {self.expression}"
                else:
                    return False, f"Invalid LLM response: {response}"

            # Multiple samples - run in parallel
            tasks = [self.llm(content=prompt) for _ in range(self.num_samples)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Parse results
            valid_responses = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                # Handle both string and LLMOutput responses
                if isinstance(result, str):
                    response = result.strip().lower()
                elif hasattr(result, "content"):
                    response = (result.content or "").strip().lower()
                else:
                    response = result.get("content", "").strip().lower()
                if response in ("true", "false"):
                    valid_responses.append(response == "true")

            # Check if we have enough valid samples
            if len(valid_responses) < self.min_samples_for_aggregation:
                return (
                    False,
                    f"Insufficient valid LLM responses: {len(valid_responses)}/{self.min_samples_for_aggregation}",
                )

            # Count how many passed
            num_passed = sum(valid_responses)

            if num_passed >= self.min_pass:
                return True, None
            else:
                return (
                    False,
                    f"Code failed criteria ({num_passed}/{len(valid_responses)} samples passed, need {self.min_pass}): {self.expression}",
                )

        # Run async verification - handle both sync and async contexts
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # We are in an async context, so we need to await directly
            # Create a new task and wait for it
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _verify_async())
                return future.result()
        except RuntimeError:
            # No event loop running, use asyncio.run
            return asyncio.run(_verify_async())


__all__ = ["IsFunction", "QualitativeCriteria"]
