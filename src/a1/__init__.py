"""
a1 - A modern agent compiler for building and executing LLM-powered agents.

This library provides:
- Agent, Tool, ToolSet: Core abstractions for defining agents
- Runtime: Execution environment with AOT/JIT compilation
- Built-in tools: LLM, Done
- RAG toolsets: FileSystemRAG, SQLRAG
- Strategies: Generate, Verify, Cost, Compact
- Observability: OpenTelemetry integration

Quick Start:
    from a1 import Agent, tool, LLM, Done, Runtime

    @tool(name="add", description="Add two numbers")
    async def add(a: int, b: int) -> int:
        return a + b

    agent = Agent(
        name="math_agent",
        description="Solves math problems",
        input_schema=MathInput,
        output_schema=MathOutput,
        tools=[add, LLM("gpt-4"), Done()]
    )

    # AOT compilation
    runtime = Runtime()
    compiled = await runtime.aot(agent)
    result = await compiled(problem="What is 2 + 2?")

    # JIT execution
    result = await runtime.jit(agent, "What is 2 + 2?")
"""

import a1.code_utils as code_utils
from .em import EM

from .builtin_tools import LLM, Done
from .context import BaseCompact, Compact, Context, no_history
from .executor import BaseExecutor, CodeOutput, Executor
from .llm import LLMInput, LLMOutput, no_context
from .models import Agent, Message, RetryStrategy, Skill, SkillSet, Strategy, Tool, ToolSet, tool
from .rag import RAG, Database, FileSystem
from .runtime import Runtime, get_context, get_runtime, new_context, set_runtime, set_strategy
from .serialization import (
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
from .strategies import (
    BaseCost,
    BaseGenerate,
    BaseVerify,
    Cost,
    Generate,
    IsLoop,
    QuantitativeCriteria,
    Verify,
)
from .extra_strategies import (
    ReduceAndGenerate,
    CheckOrdering,
)
from .extra_codecheck import (
    IsFunction,
    QualitativeCriteria,
)

__version__ = "0.1.5"

__all__ = [
    # Core models
    "Agent",
    "Tool",
    "ToolSet",
    "tool",
    "Message",
    "Strategy",
    "RetryStrategy",
    # Built-in tools
    "LLM",
    "LLMInput",
    "LLMOutput",
    "no_context",
    "Done",
    "EM",
    # Runtime
    "Runtime",
    "get_runtime",
    "set_runtime",
    "set_strategy",
    "get_context",
    "new_context",
    # Context management
    "Context",
    "no_history",
    "Compact",
    "BaseCompact",
    # Execution
    "Executor",
    "BaseExecutor",
    "CodeOutput",
    # Strategies
    "Generate",
    "BaseGenerate",
    "ReduceAndGenerate",
    "Verify",
    "BaseVerify",
    "CheckOrdering",
    "QualitativeCriteria",
    "IsLoop",
    "IsFunction",
    "Cost",
    "BaseCost",
    "QuantitativeCriteria",
    # RAG
    "FileSystem",
    "Database",
    "RAG",
    # Version
    "__version__",
]


# Convenience wrappers for top-level usage
# These use the global runtime


async def aot(agent: Agent, cache: bool = True) -> Tool:
    """
    Compile an agent ahead-of-time using the global runtime.

    Equivalent to: get_runtime().aot(agent, cache)
    """
    return await get_runtime().aot(agent, cache)


async def jit(agent: Agent, input_data: any) -> any:
    """
    Execute an agent just-in-time using the global runtime.

    Equivalent to: get_runtime().jit(agent, input_data)
    """
    return await get_runtime().jit(agent, input_data)


async def execute(tool: Tool, input_data: any) -> any:
    """
    Execute a tool using the global runtime.

    Equivalent to: get_runtime().execute(tool, input_data)
    """
    return await get_runtime().execute(tool, input_data)


# Add to __all__
__all__.extend(["aot", "jit", "execute"])
