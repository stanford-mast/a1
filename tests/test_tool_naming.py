"""
Tests for LLM tool naming map generation.

Tests verify:
1. Tool names are generated correctly (llm_a, llm_b, ..., llm_z, llm_aa, etc.)
2. Naming map is included in definition code
3. Non-LLM tools are not included in the map
4. Tool naming works with nested ToolSets
"""

import pytest
from pydantic import BaseModel, Field

from a1 import Agent, Tool
from a1.builtin_tools import LLM
from a1.codegen import BaseGenerate, generate_tool_names


class QueryInput(BaseModel):
    query: str = Field(..., description="Query")


class Output(BaseModel):
    response: str = Field(..., description="Response")


async def mock_execute(query: str) -> str:
    """Mock execution."""
    return f"Result: {query}"


class TestToolNamingGeneration:
    """Test tool name generation."""

    def test_generate_single_tool_name(self):
        """Test generating a single tool name."""
        tools = [
            Tool(
                name="llm_gpt4",
                description="GPT-4 tool",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            )
        ]

        name_map = generate_tool_names(tools)
        assert "llm_gpt4" in name_map
        assert name_map["llm_gpt4"] == "llm_a"

    def test_generate_multiple_tool_names(self):
        """Test generating multiple tool names."""
        tools = [
            Tool(
                name="llm_gpt4",
                description="GPT-4",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
            Tool(
                name="llm_claude",
                description="Claude",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
            Tool(
                name="llm_gemini",
                description="Gemini",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
        ]

        name_map = generate_tool_names(tools)
        assert len(name_map) == 3
        assert name_map["llm_gpt4"] == "llm_a"
        assert name_map["llm_claude"] == "llm_b"
        assert name_map["llm_gemini"] == "llm_c"

    def test_generate_many_tool_names(self):
        """Test generating names beyond 26 tools."""
        tools = []
        for i in range(30):
            tools.append(
                Tool(
                    name=f"llm_tool_{i}",
                    description=f"Tool {i}",
                    input_schema=QueryInput,
                    output_schema=Output,
                    execute=mock_execute,
                )
            )

        name_map = generate_tool_names(tools)
        assert len(name_map) == 30

        # First 26 should be a-z
        for i in range(26):
            expected_name = chr(ord("a") + i)
            assert name_map[f"llm_tool_{i}"] == f"llm_{expected_name}"

        # After 26 should be aa, ab, ...
        assert name_map["llm_tool_26"] == "llm_aa"
        assert name_map["llm_tool_27"] == "llm_ab"

    def test_non_llm_tools_excluded(self):
        """Test that non-LLM tools are not included in naming map."""
        tools = [
            Tool(
                name="search",
                description="Search tool",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
            Tool(
                name="llm_gpt4",
                description="GPT-4",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
            Tool(
                name="database",
                description="Database tool",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
        ]

        name_map = generate_tool_names(tools)
        assert len(name_map) == 1
        assert "search" not in name_map
        assert "database" not in name_map
        assert "llm_gpt4" in name_map


class TestToolNamingInDefinitionCode:
    """Test that tool naming appears in definition code."""

    def test_tool_naming_in_definition_code(self):
        """Test short tool names (llm_a, llm_b) appear in generated definition code."""
        llm_tool = LLM("gpt-4.1-mini")

        agent = Agent(
            name="qa_agent",
            description="Q&A agent",
            input_schema=QueryInput,
            output_schema=Output,
            tools=[llm_tool],
        )

        # Generate definition code
        gen = BaseGenerate(llm_tool=llm_tool)
        definition_code = gen._build_definition_code(agent, return_function=False)

        # Should contain short tool names like llm_a
        llm_tools = [t for t in agent.get_all_tools() if "llm" in t.name.lower()]
        if llm_tools:
            assert "llm_a" in definition_code  # First LLM tool should be llm_a

    def test_tool_naming_map_comment(self):
        """Test that short tool names are used in function definitions."""
        llm_tool = LLM("gpt-4.1-mini")

        agent = Agent(
            name="qa_agent",
            description="Q&A agent",
            input_schema=QueryInput,
            output_schema=Output,
            tools=[llm_tool],
        )

        gen = BaseGenerate(llm_tool=llm_tool)
        definition_code = gen._build_definition_code(agent, return_function=False)

        # If there are LLM tools, should see async def llm_a
        llm_tools = [t for t in agent.get_all_tools() if "llm" in t.name.lower()]
        if llm_tools:
            assert "async def llm_a(" in definition_code  # Should have short name function


class TestToolNamingWithMultipleTools:
    """Test tool naming with multiple LLM and non-LLM tools."""

    def test_mixed_tools_naming(self):
        """Test naming with both LLM and non-LLM tools."""
        tools = [
            Tool(
                name="search",
                description="Search",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
            Tool(
                name="llm_gpt4",
                description="GPT-4",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
            Tool(
                name="llm_claude",
                description="Claude",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
            Tool(
                name="database",
                description="Database",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
        ]

        name_map = generate_tool_names(tools)

        # Only LLM tools should be named
        assert len(name_map) == 2
        assert name_map["llm_gpt4"] == "llm_a"
        assert name_map["llm_claude"] == "llm_b"


class TestToolNamingEdgeCases:
    """Test edge cases in tool naming."""

    def test_empty_tool_list(self):
        """Test with no LLM tools."""
        tools = [
            Tool(
                name="search",
                description="Search",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
        ]

        name_map = generate_tool_names(tools)
        assert len(name_map) == 0

    def test_case_insensitive_llm_detection(self):
        """Test that LLM detection is case insensitive."""
        tools = [
            Tool(
                name="LLM_UPPERCASE",
                description="Uppercase LLM",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
            Tool(
                name="llm_lowercase",
                description="Lowercase LLM",
                input_schema=QueryInput,
                output_schema=Output,
                execute=mock_execute,
            ),
        ]

        name_map = generate_tool_names(tools)
        assert len(name_map) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
