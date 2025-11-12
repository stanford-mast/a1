"""
Tests for a1 core models.
"""

import pytest
from pydantic import BaseModel

from a1 import Agent, Tool, ToolSet, tool


class TestTool:
    """Test Tool class."""

    def test_tool_decorator_simple(self):
        """Test @tool decorator with simple function."""

        @tool(name="add", description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, Tool)
        assert add.name == "add"
        assert add.description == "Add two numbers"
        assert not add.is_terminal

    def test_tool_decorator_terminal(self):
        """Test @tool decorator with terminal flag."""

        @tool(name="done", description="Mark as done", is_terminal=True)
        async def done(result: str) -> str:
            return result

        assert done.is_terminal

    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution with validation."""

        @tool(name="add", description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b

        result = await add(a=2, b=3)
        assert result == 5


class TestAgent:
    """Test Agent class."""

    def test_agent_creation(self):
        """Test creating an agent."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            response: str

        @tool(name="test_tool")
        async def test_tool(x: int) -> int:
            return x * 2

        agent = Agent(
            name="test_agent", description="Test agent", input_schema=Input, output_schema=Output, tools=[test_tool]
        )

        assert agent.name == "test_agent"
        assert agent.description == "Test agent"
        assert len(agent.tools) == 1

    def test_get_all_tools(self):
        """Test getting all tools from agent including toolsets."""

        @tool(name="tool1")
        async def tool1(x: int) -> int:
            return x

        @tool(name="tool2")
        async def tool2(x: int) -> int:
            return x

        @tool(name="tool3")
        async def tool3(x: int) -> int:
            return x

        toolset = ToolSet(name="set1", description="Test toolset", tools=[tool2, tool3])

        class Input(BaseModel):
            x: int

        agent = Agent(name="test", description="Test", input_schema=Input, output_schema=Input, tools=[tool1, toolset])

        all_tools = agent.get_all_tools()
        assert len(all_tools) == 3
        assert {t.name for t in all_tools} == {"tool1", "tool2", "tool3"}

    def test_get_tool(self):
        """Test getting a specific tool by name."""

        @tool(name="target")
        async def target(x: int) -> int:
            return x

        class Input(BaseModel):
            x: int

        agent = Agent(name="test", description="Test", input_schema=Input, output_schema=Input, tools=[target])

        found = agent.get_tool("target")
        assert found is not None
        assert found.name == "target"

        not_found = agent.get_tool("nonexistent")
        assert not_found is None


class TestToolSet:
    """Test ToolSet class."""

    def test_toolset_creation(self):
        """Test creating a toolset."""

        @tool(name="tool1")
        async def tool1(x: int) -> int:
            return x

        @tool(name="tool2")
        async def tool2(x: int) -> int:
            return x

        toolset = ToolSet(name="test_set", description="Test toolset", tools=[tool1, tool2])

        assert toolset.name == "test_set"
        assert len(toolset.tools) == 2
