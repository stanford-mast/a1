"""
Models package for a1 agent compiler.

This package contains all core models split into separate modules to avoid circular imports:
- message: Message model
- strategy: Strategy classes (AttemptStrategy, RetryStrategy, Strategy, etc.)
- tool: Tool class and tool() decorator (most fundamental)
- toolset: ToolSet class
- skill: Skill and SkillSet classes
- agent: Agent class
"""

from .agent import Agent
from .message import Message
from .skill import Skill, SkillSet
from .strategy import AttemptStrategy, ParallelStrategy, RetryStrategy, Strategy
from .tool import Tool, tool
from .toolset import ToolSet

__all__ = [
    "Message",
    "AttemptStrategy",
    "ParallelStrategy",
    "RetryStrategy",
    "Strategy",
    "Tool",
    "tool",
    "ToolSet",
    "Skill",
    "SkillSet",
    "Agent",
]
