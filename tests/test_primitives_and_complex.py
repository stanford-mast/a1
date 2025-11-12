#!/usr/bin/env python3
"""Test that code generation handles both primitive and complex Pydantic types correctly."""

import asyncio
import os
from pathlib import Path

from pydantic import BaseModel, Field

# Load environment variables
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

from a1 import LLM, Agent, tool


# Test 1: Primitive return types
@tool(name="add", description="Add two numbers")
async def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool(name="greet", description="Greet someone")
async def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


@tool(name="is_positive", description="Check if number is positive")
async def is_positive(num: int) -> bool:
    """Check if a number is positive."""
    return num > 0


# Test 2: Complex Pydantic types
class PersonInfo(BaseModel):
    name: str = Field(..., description="Person's name")
    age: int = Field(..., description="Person's age")


class GreetingResponse(BaseModel):
    greeting: str = Field(..., description="Greeting message")
    person: PersonInfo = Field(..., description="Person information")


@tool(name="create_greeting", description="Create a greeting with person info")
async def create_greeting(name: str, age: int) -> GreetingResponse:
    """Create a greeting response with person information."""
    return GreetingResponse(greeting=f"Hello, {name}!", person=PersonInfo(name=name, age=age))


# Test agents
class MathInput(BaseModel):
    problem: str


class MathOutput(BaseModel):
    answer: int


class PersonGreetingInput(BaseModel):
    query: str


class PersonGreetingOutput(BaseModel):
    result: GreetingResponse


async def test_primitive_int():
    """Test that primitive int return types work correctly."""
    print("\n--- Testing Primitive Int Return Type ---")
    agent = Agent(
        name="math_agent",
        description="Solves math problems",
        input_schema=MathInput,
        output_schema=MathOutput,
        tools=[add, LLM(model="gpt-4.1")],
    )

    compiled = await agent.aot()
    print(f"✓ Compiled primitive int agent: {compiled.name}")
    assert compiled.name == "math_agent"
    return True


async def test_primitive_string():
    """Test that primitive string return types work correctly."""
    print("\n--- Testing Primitive String Return Type ---")

    class GreetInput(BaseModel):
        name: str

    class GreetOutput(BaseModel):
        message: str

    agent = Agent(
        name="greeting_agent",
        description="Greets people",
        input_schema=GreetInput,
        output_schema=GreetOutput,
        tools=[greet, LLM(model="gpt-4.1")],
    )

    compiled = await agent.aot()
    print(f"✓ Compiled primitive string agent: {compiled.name}")
    assert compiled.name == "greeting_agent"
    return True


async def test_primitive_bool():
    """Test that primitive bool return types work correctly."""
    print("\n--- Testing Primitive Bool Return Type ---")

    class CheckInput(BaseModel):
        num: int

    class CheckOutput(BaseModel):
        is_positive: bool

    agent = Agent(
        name="check_agent",
        description="Checks if positive",
        input_schema=CheckInput,
        output_schema=CheckOutput,
        tools=[is_positive, LLM(model="gpt-4.1")],
    )

    compiled = await agent.aot()
    print(f"✓ Compiled primitive bool agent: {compiled.name}")
    assert compiled.name == "check_agent"
    return True


async def test_complex_pydantic():
    """Test that complex Pydantic types work correctly."""
    print("\n--- Testing Complex Pydantic Return Type ---")

    agent = Agent(
        name="person_greeting_agent",
        description="Creates greeting with person info",
        input_schema=PersonGreetingInput,
        output_schema=PersonGreetingOutput,
        tools=[create_greeting, LLM(model="gpt-4.1")],
    )

    compiled = await agent.aot()
    print(f"✓ Compiled complex Pydantic agent: {compiled.name}")
    assert compiled.name == "person_greeting_agent"
    return True


async def test_tool_execution():
    """Test that tools execute correctly with their actual return types."""
    print("\n--- Testing Direct Tool Execution ---")

    # Test primitive int
    result = await add(a=5, b=3)
    print(f"✓ add(5, 3) returned: {result} (type: {type(result).__name__})")
    assert result == 8

    # Test primitive string
    result = await greet(name="Alice")
    print(f"✓ greet('Alice') returned: {result} (type: {type(result).__name__})")
    assert "Hello, Alice" in result

    # Test primitive bool
    result = await is_positive(num=5)
    print(f"✓ is_positive(5) returned: {result} (type: {type(result).__name__})")
    assert result is True

    # Test complex Pydantic
    result = await create_greeting(name="Bob", age=30)
    print(f"✓ create_greeting('Bob', 30) returned: {result}")
    assert isinstance(result, GreetingResponse)
    assert result.person.name == "Bob"
    assert result.person.age == 30

    return True


async def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("Testing Code Generation for Primitive and Complex Types")
    print("=" * 70)

    tests = [
        ("Primitive Int", test_primitive_int),
        ("Primitive String", test_primitive_string),
        ("Primitive Bool", test_primitive_bool),
        ("Complex Pydantic", test_complex_pydantic),
        ("Tool Execution", test_tool_execution),
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
