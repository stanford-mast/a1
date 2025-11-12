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

from .models import Agent, Tool, ToolSet, Skill, SkillSet, tool, Message, Strategy, RetryStrategy
from .builtin_tools import LLM, Done
from .llm import LLMInput, LLMOutput, no_context
from .runtime import Runtime, get_runtime, set_runtime, set_strategy, get_context, new_context
from .context import Context, no_history, Compact, BaseCompact
from .executor import Executor, BaseExecutor, CodeOutput
from .strategies import (
    Generate, BaseGenerate,
    Verify, BaseVerify, QualitativeCriteria, IsLoop,
    Cost, BaseCost, QuantitativeCriteria
)
from .rag import FileSystem, Database, RAG
from .serialization import (
    serialize_agent, deserialize_agent,
    serialize_tool, deserialize_tool,
    serialize_skill, deserialize_skill,
    save_agent_to_file, load_agent_from_file,
    save_tool_to_file, load_tool_from_file,
    save_skill_to_file, load_skill_from_file,
)
import a1.code_utils as code_utils

__version__ = "0.1.1"

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
    "Verify",
    "BaseVerify",
    "QualitativeCriteria",
    "IsLoop",
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
