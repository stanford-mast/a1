#!/usr/bin/env python3
"""Test the README example with both AOT and JIT execution."""

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from a1 import LLM, Agent, tool


# Define a simple tool
@tool(name="add", description="Add two numbers")
async def add(a: int, b: int) -> int:
    return a + b


# Define input/output schemas
class MathInput(BaseModel):
    problem: str


class MathOutput(BaseModel):
    answer: int


# Create an agent with tools and LLM
agent = Agent(
    name="math_agent",
    description="Solves simple math problems",
    input_schema=MathInput,
    output_schema=MathOutput,
    tools=[add, LLM(model="gpt-4.1")],
)


async def test_aot_compilation():
    """Test AOT compilation works with primitive return types."""
    print("\n--- Testing AOT Compilation ---")

    # Compile ahead-of-time
    compiled = await agent.aot()
    print(f"✓ Compiled agent to tool: {compiled.name}")
    assert compiled.name == "math_agent"
    assert compiled.description == "Solves simple math problems"

    return True


async def test_jit_execution():
    """Test JIT execution works with primitive return types and simple input."""
    print("\n--- Testing JIT Execution ---")

    # Execute just-in-time
    result = await agent.jit(problem="What is 2 + 2?")
    print(f"✓ JIT result type: {type(result).__name__}")
    print(f"✓ JIT result: {result}")

    # Result should be MathOutput with an int answer
    assert isinstance(result, MathOutput), f"Expected MathOutput, got {type(result).__name__}"
    assert hasattr(result, "answer"), "Result should have 'answer' field"
    assert isinstance(result.answer, int), f"Answer should be int, got {type(result.answer).__name__}"

    return True


async def main():
    """Run README example tests."""
    print("\n" + "=" * 70)
    print("Testing README Example with AOT and JIT")
    print("=" * 70)

    tests = [
        ("AOT Compilation", test_aot_compilation),
        ("JIT Execution", test_jit_execution),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, True, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test Results")
    print("=" * 70)

    passed = 0
    failed = 0
    for test_name, success, error in results:
        if success:
            print(f"✓ {test_name}: PASSED")
            passed += 1
        else:
            print(f"✗ {test_name}: FAILED - {error}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"Total: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
