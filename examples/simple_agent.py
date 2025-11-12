"""
Simple example demonstrating a1 usage with AOT and JIT modes.

Creates a math agent that can perform calculations using a calculator tool
and LLM reasoning.
"""

import asyncio
import logging
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from a1 import LLM, Agent, Runtime, Tool

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

# Enable logging to see generated code
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")


# Define input and output schemas
class MathInput(BaseModel):
    """Input for math agent."""

    problem: str = Field(..., description="Math problem to solve")


class MathOutput(BaseModel):
    """Output from math agent."""

    answer: str = Field(..., description="Solution to the problem")


# Define calculator tool
class CalculatorInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, or divide")


class CalculatorOutput(BaseModel):
    result: float = Field(..., description="Result of the calculation")


async def calculator_execute(a: float, b: float, operation: str) -> dict:
    """Execute calculator operations."""
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


async def main():
    """Run the example."""
    # Create calculator tool
    calculator = Tool(
        name="calculator",
        description="Perform basic arithmetic operations (add, subtract, multiply, divide)",
        input_schema=CalculatorInput,
        output_schema=CalculatorOutput,
        execute=calculator_execute,
        is_terminal=False,
    )

    # Create LLM tool
    llm = LLM("groq:openai/gpt-oss-20b")

    # Create agent with tools
    agent = Agent(
        name="math_agent",
        description="Solves math problems using calculator and LLM reasoning",
        input_schema=MathInput,
        output_schema=MathOutput,
        tools=[calculator, llm],
    )

    # Create runtime
    runtime = Runtime()

    print("=== Math Agent Example ===\n")

    # Example 1: AOT compilation (Ahead-Of-Time)
    print("1. AOT Compilation:")
    print("   Generates and compiles agent code once, then reuses")
    print("   (Requires GROQ_API_KEY set)\n")
    try:
        compiled = await runtime.aot(agent, cache=False)
        result = await compiled(problem="What is 42 divided by 7?")
        print("   Input: What is 42 divided by 7?")
        print(f"   Output: {result.answer}\n")
    except Exception as e:
        print(f"   Skipped: {e}\n")

    # Example 2: JIT execution (Just-In-Time)
    print("2. JIT Execution:")
    print("   Generates and executes agent code on-the-fly")
    print("   (Requires GROQ_API_KEY set)\n")
    try:
        result = await runtime.jit(agent, problem="What is 5 times 6?")
        print("   Input: What is 5 times 6?")
        print(f"   Output: {result.answer}\n")
    except Exception as e:
        print(f"   Skipped: {e}\n")

    # Example 3: Direct tool execution
    print("3. Direct Tool Execution:")
    print("   Execute tools without agent compilation")
    result = await runtime.execute(calculator, a=10, b=3, operation="add")
    print(f"   Calculator(10, 3, 'add') = {result.result}\n")

    print("=== Example Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
