"""
Test @tool decorator and direct function passing to Agent.

This tests:
1. @tool decorator with docstrings and type hints
2. Direct function passing to Agent (auto-conversion)
3. Various type combinations (primitives, list, dict, BaseModel)
"""

from typing import Any

import pytest
from pydantic import BaseModel

from a1 import Agent, Tool, tool


class TestToolDecorator:
    """Test the @tool decorator."""

    def test_tool_decorator_with_primitives(self):
        """Test @tool with primitive types."""

        @tool(name="add", description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, Tool)
        assert add.name == "add"
        assert add.description == "Add two numbers"
        assert add.input_schema.__name__ == "add_Input"
        assert add.output_schema.__name__ == "add_Output"
        print("✓ @tool with primitives works")

    def test_tool_decorator_with_docstring(self):
        """Test @tool using function's docstring."""

        @tool()
        async def multiply(x: float, y: float) -> float:
            """Multiply two numbers together."""
            return x * y

        assert isinstance(multiply, Tool)
        assert multiply.name == "multiply"
        assert multiply.description == "Multiply two numbers together."
        print("✓ @tool extracts docstring")

    def test_tool_decorator_with_list_return(self):
        """Test @tool with list return type."""

        @tool(name="split_words")
        async def split_words(text: str) -> list[str]:
            """Split text into words."""
            return text.split()

        assert isinstance(split_words, Tool)
        assert "split_words" in split_words.name
        print("✓ @tool with list[str] return type works")

    def test_tool_decorator_with_dict_param(self):
        """Test @tool with dict parameter."""

        @tool()
        async def process_data(data: dict[str, int]) -> int:
            """Sum values in dictionary."""
            return sum(data.values())

        assert isinstance(process_data, Tool)
        print("✓ @tool with dict parameter works")

    def test_tool_decorator_with_pydantic_models(self):
        """Test @tool with Pydantic models."""

        class PersonInput(BaseModel):
            name: str
            age: int

        class PersonOutput(BaseModel):
            greeting: str

        @tool()
        async def greet_person(person: PersonInput) -> PersonOutput:
            """Greet a person."""
            return PersonOutput(greeting=f"Hello, {person.name}!")

        assert isinstance(greet_person, Tool)
        print("✓ @tool with Pydantic models works")

    @pytest.mark.asyncio
    async def test_tool_decorator_execution(self):
        """Test that decorated tool can be executed."""

        @tool(name="calculate", description="Calculate sum")
        async def calculate(a: int, b: int) -> int:
            return a + b

        result = await calculate(a=5, b=3)
        # Tool returns the actual result directly
        assert result == 8
        print("✓ @tool decorated function executes correctly")


class TestDirectFunctionPassing:
    """Test passing raw functions to Agent (auto-conversion)."""

    def test_agent_with_raw_function(self):
        """Test Agent auto-converts raw functions to Tools."""

        async def my_tool(x: int, y: int) -> int:
            """Add two numbers."""
            return x + y

        agent = Agent(name="test_agent", tools=[my_tool])

        # Should have converted function to Tool
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], Tool)
        assert agent.tools[0].name == "my_tool"
        assert agent.tools[0].description == "Add two numbers."
        print("✓ Agent auto-converts raw function to Tool")

    def test_agent_with_mixed_tools(self):
        """Test Agent with mix of Tools and raw functions."""

        @tool(name="decorated_tool")
        async def decorated(a: int) -> int:
            """Decorated tool."""
            return a * 2

        async def raw_function(b: int) -> int:
            """Raw function."""
            return b + 1

        agent = Agent(name="test_agent", tools=[decorated, raw_function])

        assert len(agent.tools) == 2
        assert all(isinstance(t, Tool) for t in agent.tools)
        assert agent.tools[0].name == "decorated_tool"
        assert agent.tools[1].name == "raw_function"
        print("✓ Agent handles mixed Tools and functions")

    def test_agent_with_function_no_types(self):
        """Test Agent with function without type hints."""

        async def untyped_tool(x, y):
            """Tool without type hints."""
            return x + y

        agent = Agent(name="test_agent", tools=[untyped_tool])

        # Should still create a Tool, but with Any types
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], Tool)
        assert agent.tools[0].name == "untyped_tool"
        print("✓ Agent handles functions without type hints")

    def test_agent_with_function_no_docstring(self):
        """Test Agent with function without docstring."""

        async def no_doc(x: int) -> int:
            return x

        agent = Agent(name="test_agent", tools=[no_doc])

        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], Tool)
        # Should have generated description
        assert "no_doc" in agent.tools[0].description
        print("✓ Agent handles functions without docstrings")

    @pytest.mark.asyncio
    async def test_agent_auto_converted_tool_execution(self):
        """Test that auto-converted tools can be executed."""

        async def calculator(operation: str, a: int, b: int) -> int:
            """Perform a calculation."""
            if operation == "add":
                return a + b
            elif operation == "multiply":
                return a * b
            else:
                raise ValueError(f"Unknown operation: {operation}")

        agent = Agent(name="calc_agent", tools=[calculator])

        # Get the converted tool
        calc_tool = agent.tools[0]
        assert isinstance(calc_tool, Tool)

        # Execute it - Tool returns actual result
        result = await calc_tool(operation="add", a=10, b=5)
        assert result == 15

        result = await calc_tool(operation="multiply", a=10, b=5)
        assert result == 50

        print("✓ Auto-converted tool executes correctly")


class TestComplexScenarios:
    """Test complex tool scenarios."""

    def test_nested_toolsets_with_functions(self):
        """Test ToolSet with raw functions."""

        async def func1(x: int) -> int:
            """Function 1."""
            return x + 1

        async def func2(x: int) -> int:
            """Function 2."""
            return x * 2

        # ToolSet currently expects Tool objects, not raw functions
        # But Agent should handle them
        agent = Agent(name="test", tools=[func1, func2])

        assert len(agent.tools) == 2
        assert all(isinstance(t, Tool) for t in agent.tools)
        print("✓ Multiple functions auto-converted")

    def test_agent_with_no_tools(self):
        """Test Agent with empty tools list."""
        agent = Agent(name="empty_agent")
        assert agent.tools == []
        print("✓ Agent with no tools works")

    def test_type_preservation(self):
        """Test that type hints are preserved in schema."""

        @tool()
        async def typed_func(text: str, count: int, ratio: float, enabled: bool, items: list[str]) -> dict[str, Any]:
            """Function with various types."""
            return {"processed": True}

        # Check input schema has correct types
        input_fields = typed_func.input_schema.model_fields
        assert "text" in input_fields
        assert "count" in input_fields
        assert "ratio" in input_fields
        assert "enabled" in input_fields
        assert "items" in input_fields
        print("✓ Type hints preserved in schema")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
