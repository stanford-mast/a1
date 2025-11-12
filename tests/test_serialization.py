"""
Tests for serialization/deserialization of a1 models.

Tests the ability to:
1. Serialize Agent, Tool, ToolSet, Skill, SkillSet to JSON
2. Deserialize them back from JSON
3. Save/load from files
4. Handle recursive Union types (Tool|ToolSet, Skill|SkillSet)
5. Preserve all important attributes through serialization
"""

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import BaseModel, Field

from a1 import (
    Agent,
    Skill,
    SkillSet,
    Tool,
    ToolSet,
    deserialize_agent,
    deserialize_skill,
    deserialize_tool,
    load_agent_from_file,
    load_skill_from_file,
    load_tool_from_file,
    save_agent_to_file,
    save_skill_to_file,
    save_tool_to_file,
    serialize_agent,
    serialize_skill,
    serialize_tool,
)
from a1.serialization import (
    deserialize_callable,
    deserialize_model_type,
    serialize_callable,
    serialize_model_type,
)


# Test schemas
class SimpleInput(BaseModel):
    query: str = Field(..., description="User query")


class SimpleOutput(BaseModel):
    response: str = Field(..., description="Agent response")


# Test functions
async def simple_func(query: str) -> str:
    """Simple test function."""
    return f"Processed: {query}"


async def another_func(query: str) -> str:
    """Another test function."""
    return f"Alternative: {query}"


class TestCallableSerialization:
    """Test serialization of callables."""

    def test_serialize_deserialize_regular_function(self):
        """Test serializing and deserializing a regular function."""
        # Serialize
        serialized = serialize_callable(simple_func)
        assert serialized["type"] == "function"
        assert serialized["name"] == "simple_func"

        # Deserialize
        deserialized = deserialize_callable(serialized)
        assert deserialized.__name__ == "simple_func"

    def test_serialize_none_callable(self):
        """Test serializing None."""
        serialized = serialize_callable(None)
        assert serialized["type"] == "none"

        deserialized = deserialize_callable(serialized)
        assert deserialized is None


class TestModelTypeSerialization:
    """Test serialization of Pydantic model types."""

    def test_serialize_deserialize_model_type(self):
        """Test serializing and deserializing a model type."""
        # Serialize
        serialized = serialize_model_type(SimpleInput)
        assert serialized["type"] == "reference"
        assert "module" in serialized
        assert "name" in serialized

        # Deserialize
        deserialized = deserialize_model_type(serialized)
        assert deserialized.__name__ == "SimpleInput"

        # Verify it works
        instance = deserialized(query="test")
        assert instance.query == "test"


class TestToolSerialization:
    """Test serialization of Tool objects."""

    def test_serialize_tool_to_json(self):
        """Test serializing a Tool to JSON string."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
            is_terminal=False,
        )

        # Serialize
        json_str = serialize_tool(tool)
        assert isinstance(json_str, str)

        # Verify it's valid JSON
        data = json.loads(json_str)
        assert data["_type"] == "Tool"
        assert data["name"] == "test_tool"
        assert data["description"] == "A test tool"
        assert data["is_terminal"] is False

    def test_deserialize_tool_from_json(self):
        """Test deserializing a Tool from JSON string."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
            is_terminal=False,
        )

        # Serialize and deserialize
        json_str = serialize_tool(tool)
        deserialized = deserialize_tool(json_str)

        assert isinstance(deserialized, Tool)
        assert deserialized.name == "test_tool"
        assert deserialized.description == "A test tool"
        assert deserialized.is_terminal is False


class TestToolSetSerialization:
    """Test serialization of ToolSet objects."""

    def test_serialize_toolset(self):
        """Test serializing a ToolSet."""
        tool1 = Tool(
            name="tool1",
            description="First tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
        )
        tool2 = Tool(
            name="tool2",
            description="Second tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=another_func,
        )

        toolset = ToolSet(
            name="test_toolset",
            description="A test toolset",
            tools=[tool1, tool2],
        )

        # Serialize
        json_str = serialize_tool(toolset)
        data = json.loads(json_str)
        assert data["_type"] == "ToolSet"
        assert data["name"] == "test_toolset"
        assert len(data["tools"]) == 2

    def test_serialize_nested_toolset(self):
        """Test serializing a ToolSet containing another ToolSet."""
        tool1 = Tool(
            name="tool1",
            description="First tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
        )

        inner_toolset = ToolSet(
            name="inner_toolset",
            description="Inner toolset",
            tools=[tool1],
        )

        outer_toolset = ToolSet(
            name="outer_toolset",
            description="Outer toolset",
            tools=[inner_toolset],
        )

        # Serialize
        json_str = serialize_tool(outer_toolset)
        assert isinstance(json_str, str)

        # Verify structure
        data = json.loads(json_str)
        assert data["_type"] == "ToolSet"
        assert len(data["tools"]) == 1
        assert data["tools"][0]["_type"] == "ToolSet"


class TestSkillSerialization:
    """Test serialization of Skill objects."""

    def test_serialize_skill(self):
        """Test serializing a Skill."""
        skill = Skill(
            name="python_basics",
            description="Python programming basics",
            content="# Python Basics\ndef hello():\n    print('Hello')",
            modules=["sys", "os"],
        )

        # Serialize
        json_str = serialize_skill(skill)
        data = json.loads(json_str)

        assert data["_type"] == "Skill"
        assert data["name"] == "python_basics"
        assert data["description"] == "Python programming basics"
        assert "def hello" in data["content"]
        assert data["modules"] == ["sys", "os"]

    def test_deserialize_skill(self):
        """Test deserializing a Skill."""
        skill = Skill(
            name="python_basics",
            description="Python programming basics",
            content="# Python Basics\ndef hello():\n    print('Hello')",
            modules=["sys", "os"],
        )

        # Serialize and deserialize
        json_str = serialize_skill(skill)
        deserialized = deserialize_skill(json_str)

        assert isinstance(deserialized, Skill)
        assert deserialized.name == "python_basics"
        assert deserialized.modules == ["sys", "os"]


class TestSkillSetSerialization:
    """Test serialization of SkillSet objects."""

    def test_serialize_skillset(self):
        """Test serializing a SkillSet."""
        skill1 = Skill(
            name="python_basics",
            description="Python basics",
            content="# Python",
            modules=["sys"],
        )
        skill2 = Skill(
            name="python_advanced",
            description="Python advanced",
            content="# Advanced Python",
            modules=["asyncio"],
        )

        skillset = SkillSet(
            name="python_skills",
            description="Python programming skills",
            skills=[skill1, skill2],
        )

        # Serialize
        json_str = serialize_skill(skillset)
        data = json.loads(json_str)

        assert data["_type"] == "SkillSet"
        assert len(data["skills"]) == 2


class TestAgentSerialization:
    """Test serialization of Agent objects."""

    def test_serialize_simple_agent(self):
        """Test serializing a simple Agent."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
        )

        agent = Agent(
            name="test_agent",
            description="A test agent",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            tools=[tool],
        )

        # Serialize
        json_str = serialize_agent(agent)
        data = json.loads(json_str)

        assert data["_type"] == "Agent"
        assert data["name"] == "test_agent"
        assert len(data["tools"]) == 1

    def test_deserialize_simple_agent(self):
        """Test deserializing a simple Agent."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
        )

        agent = Agent(
            name="test_agent",
            description="A test agent",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            tools=[tool],
        )

        # Serialize and deserialize
        json_str = serialize_agent(agent)
        deserialized = deserialize_agent(json_str)

        assert isinstance(deserialized, Agent)
        assert deserialized.name == "test_agent"
        assert len(deserialized.tools) == 1

    def test_serialize_agent_with_skills(self):
        """Test serializing an Agent with skills."""
        skill = Skill(
            name="python_basics",
            description="Python basics",
            content="# Python",
            modules=["sys"],
        )

        agent = Agent(
            name="test_agent",
            description="A test agent",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            skills=[skill],
        )

        # Serialize
        json_str = serialize_agent(agent)
        data = json.loads(json_str)

        assert data["_type"] == "Agent"
        assert len(data["skills"]) == 1
        assert data["skills"][0]["_type"] == "Skill"


class TestFileIO:
    """Test file I/O for serialization."""

    def test_save_and_load_agent(self):
        """Test saving and loading an Agent from file."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
        )

        agent = Agent(
            name="test_agent",
            description="A test agent",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            tools=[tool],
        )

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.json"

            save_agent_to_file(agent, path)
            assert path.exists()

            # Load back
            loaded = load_agent_from_file(path)
            assert isinstance(loaded, Agent)
            assert loaded.name == "test_agent"

    def test_save_and_load_tool(self):
        """Test saving and loading a Tool from file."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "tool.json"

            save_tool_to_file(tool, path)
            assert path.exists()

            loaded = load_tool_from_file(path)
            assert isinstance(loaded, Tool)
            assert loaded.name == "test_tool"

    def test_save_and_load_skill(self):
        """Test saving and loading a Skill from file."""
        skill = Skill(
            name="python_basics",
            description="Python basics",
            content="# Python",
            modules=["sys"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "skill.json"

            save_skill_to_file(skill, path)
            assert path.exists()

            loaded = load_skill_from_file(path)
            assert isinstance(loaded, Skill)
            assert loaded.name == "python_basics"

    def test_agent_save_load_methods(self):
        """Test Agent.save_to_file and Agent.load_from_file methods."""
        tool = Tool(
            name="test_tool",
            description="A test tool",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
        )

        agent = Agent(
            name="test_agent",
            description="A test agent",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            tools=[tool],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "agent.json"

            agent.save_to_file(path)
            assert path.exists()

            loaded = Agent.load_from_file(path)
            assert isinstance(loaded, Agent)
            assert loaded.name == "test_agent"


class TestRecursiveUnionTypes:
    """Test that recursive Union types are preserved during serialization."""

    def test_nested_toolset_preservation(self):
        """Test that nested ToolSet structure is preserved."""
        tool1 = Tool(
            name="tool1",
            description="Tool 1",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
        )

        inner_toolset = ToolSet(
            name="inner",
            description="Inner toolset",
            tools=[tool1],
        )

        outer_toolset = ToolSet(
            name="outer",
            description="Outer toolset",
            tools=[inner_toolset, tool1],  # Both ToolSet and Tool
        )

        json_str = serialize_tool(outer_toolset)
        deserialized = deserialize_tool(json_str)

        assert isinstance(deserialized, ToolSet)
        assert len(deserialized.tools) == 2
        assert isinstance(deserialized.tools[0], ToolSet)
        assert isinstance(deserialized.tools[1], Tool)

    def test_mixed_tools_and_toolsets_in_agent(self):
        """Test Agent with both Tool and ToolSet."""
        tool1 = Tool(
            name="tool1",
            description="Tool 1",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            execute=simple_func,
        )

        toolset = ToolSet(
            name="toolset1",
            description="Toolset 1",
            tools=[tool1],
        )

        agent = Agent(
            name="test_agent",
            description="Test agent",
            input_schema=SimpleInput,
            output_schema=SimpleOutput,
            tools=[tool1, toolset],  # Both Tool and ToolSet
        )

        json_str = serialize_agent(agent)
        deserialized = deserialize_agent(json_str)

        assert len(deserialized.tools) == 2
        assert isinstance(deserialized.tools[0], Tool)
        assert isinstance(deserialized.tools[1], ToolSet)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
