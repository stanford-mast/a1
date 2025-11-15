"""Agent model - composition of tools, skills, and behavior."""

from pathlib import Path
from typing import Any, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from .skill import Skill, SkillSet
from .strategy import Strategy
from .tool import Tool
from .toolset import ToolSet


class Agent(BaseModel):
    """
    An agent is a composition of tools, skills, and defined behavior.

    Attributes:
        name: Unique identifier for the agent (default: "agent")
        description: Human-readable description of agent's purpose (default: "")
        input_schema: Pydantic model for agent input (default: str wrapped in Input model)
        output_schema: Pydantic model for agent output (default: str wrapped in Output model)
        tools: List of tools or toolsets available to the agent
        skills: List of skills or skillsets available to the agent
    """

    name: str = "agent"
    description: str = ""
    input_schema: type[BaseModel] = Field(default_factory=lambda: create_model("Input", input=(str, ...)))
    output_schema: type[BaseModel] = Field(default_factory=lambda: create_model("Output", output=(str, ...)))
    tools: list[Tool | ToolSet] = Field(default_factory=list)
    skills: list[Skill | SkillSet] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    @classmethod
    def convert_functions_to_tools(cls, data: Any) -> Any:
        """Pre-validation hook to convert raw callable functions to Tool objects."""
        if isinstance(data, dict) and "tools" in data:
            data["tools"] = cls._convert_callables_to_tools(data["tools"])
        return data

    @classmethod
    def _convert_callables_to_tools(cls, tools_list: list) -> list[Tool | ToolSet]:
        """Convert raw callable functions to Tool objects."""
        from ..llm import LLM

        converted = []
        for item in tools_list:
            # If it's already a Tool or ToolSet, keep it as-is
            if isinstance(item, (Tool, ToolSet)):
                converted.append(item)
            # If it's an LLM instance, convert to its .tool property
            elif isinstance(item, LLM):
                converted.append(item.tool)
            # If it's a callable function, convert to Tool
            elif callable(item) and not isinstance(item, type):
                # Extract function metadata
                func_name = getattr(item, "__name__", "unnamed_tool")
                func_doc = getattr(item, "__doc__", "") or ""
                func_desc = func_doc.strip()

                # Get type hints
                try:
                    hints = get_type_hints(item)
                except Exception:
                    hints = {}

                return_type = hints.pop("return", Any)

                # Create input schema from parameters
                input_fields = {}
                for param_name, param_type in hints.items():
                    input_fields[param_name] = (param_type, ...)

                input_model = create_model(f"{func_name}_Input", **input_fields)

                # Create output schema from return type
                if return_type == Any or return_type is None:
                    output_model = create_model(f"{func_name}_Output", result=(Any, ...))
                elif isinstance(return_type, type) and issubclass(return_type, BaseModel):
                    output_model = return_type
                else:
                    output_model = create_model(f"{func_name}_Output", result=(return_type, ...))

                # Create Tool
                converted.append(
                    Tool(
                        name=func_name,
                        description=func_desc or f"Function: {func_name}",
                        input_schema=input_model,
                        output_schema=output_model,
                        execute=item,
                        is_terminal=False,
                    )
                )
            else:
                # Unknown type, keep as-is and let Pydantic validation catch it
                converted.append(item)

        return converted

    def get_all_tools(self) -> list[Tool]:
        """Flatten all tools from tools and toolsets."""
        from ..llm import LLM

        all_tools = []
        for item in self.tools:
            if isinstance(item, Tool):
                all_tools.append(item)
            elif isinstance(item, ToolSet):
                # Handle tools in ToolSet (could be Tool, ToolSet, or LLM)
                for tool_item in item.tools:
                    if isinstance(tool_item, Tool):
                        all_tools.append(tool_item)
                    elif isinstance(tool_item, ToolSet):
                        # Nested ToolSet - recursively process it
                        nested_agent = Agent(tools=[tool_item])
                        all_tools.extend(nested_agent.get_all_tools())
                    elif isinstance(tool_item, LLM):
                        all_tools.append(tool_item.tool)
            elif isinstance(item, LLM):
                all_tools.append(item.tool)
        return all_tools

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        for tool in self.get_all_tools():
            if tool.name == name:
                return tool
        return None

    def save_to_file(self, path: str | Path) -> None:
        """
        Save this agent to a JSON file.

        Args:
            path: File path to save to
        """
        from ..serialization import save_agent_to_file

        save_agent_to_file(self, path)

    @classmethod
    def load_from_file(cls, path: str | Path) -> "Agent":
        """
        Load an agent from a JSON file.

        Args:
            path: File path to load from

        Returns:
            Loaded Agent
        """
        from ..serialization import load_agent_from_file

        return load_agent_from_file(path)

    async def aot(self, cache: bool = True, strategy: Strategy | None = None) -> Tool:
        """
        Ahead-of-time compile this agent to a tool.

        Uses the global runtime to compile the agent. This is a thin wrapper
        around Runtime.aot() for convenience.

        Args:
            cache: Whether to use cached compilation (default: True)
            strategy: Optional Strategy for generation config (default: Strategy())

        Returns:
            Tool that executes the compiled agent
        """
        from ..runtime import get_runtime

        runtime = get_runtime()
        return await runtime.aot(self, cache=cache, strategy=strategy)

    async def jit(self, strategy: Strategy | None = None, **kwargs) -> Any:
        """
        Just-in-time execute this agent.

        Uses the global runtime to execute the agent. This is a thin wrapper
        around Runtime.jit() for convenience.

        Supports ergonomic string input: if the agent's input schema has exactly
        one string field, pass it as keyword argument with the field name, or
        use any kwarg name and it will be auto-mapped to the string field.

        Example:
            # With single string field 'query' in input schema
            result = await agent.jit(query="What is 2+2?")
            # Or with auto-mapping:
            result = await agent.jit(text="What is 2+2?")

        Args:
            strategy: Optional Strategy for generation config (default: Strategy())
            **kwargs: Input arguments matching this agent's input_schema

        Returns:
            Output from the agent
        """
        from ..runtime import get_runtime

        runtime = get_runtime()
        return await runtime.jit(self, strategy=strategy, **kwargs)

    @classmethod
    def from_langchain(cls, langchain_agent: Any) -> "Agent":
        """
        Convert a LangChain agent to an a1 Agent.

        Args:
            langchain_agent: A LangChain agent instance

        Returns:
            Equivalent a1 Agent
        """
        # Import here to avoid hard dependency
        try:
            from langchain.agents import AgentExecutor  # noqa: F401
        except ImportError:
            raise ImportError("langchain is required for from_langchain. Install with: pip install langchain")

        # Extract tools from LangChain agent
        tools = []
        if hasattr(langchain_agent, "tools"):
            for lc_tool in langchain_agent.tools:
                # Convert LangChain tool to a1 Tool
                # LangChain tools have name, description, func
                input_schema = create_model(
                    f"{lc_tool.name}_Input",
                    input=(str, ...),  # Simplified - LangChain tools often just take string input
                )
                output_schema = create_model(f"{lc_tool.name}_Output", result=(str, ...))

                async def execute_wrapper(input: str):
                    return {"result": lc_tool.func(input)}

                tools.append(
                    Tool(
                        name=lc_tool.name,
                        description=lc_tool.description or "",
                        input_schema=input_schema,
                        output_schema=output_schema,
                        execute=execute_wrapper,
                        is_terminal=False,
                    )
                )

        # Create agent
        return cls(
            name=getattr(langchain_agent, "name", "langchain_agent"),
            description=getattr(langchain_agent, "description", "Converted from LangChain"),
            input_schema=create_model("Input", query=(str, ...)),
            output_schema=create_model("Output", response=(str, ...)),
            tools=tools,
        )


__all__ = ["Agent"]
