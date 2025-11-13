"""
Core models for a1 agent compiler.

This module defines the fundamental building blocks:
- Agent: Composed of tools and terminal conditions
- Tool: Executable function with schema validation
- ToolSet: Collection of related tools
"""

import inspect
from collections.abc import Callable
from datetime import datetime
from typing import Any, Optional, Union, get_type_hints
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, create_model


class Message(BaseModel):
    """Chat message in a context."""

    role: str = Field(..., description="Message role (user, assistant, system, tool)")
    content: str = Field(..., description="Message content")
    name: str | None = Field(None, description="Name of tool or function")
    tool_call_id: str | None = Field(None, description="ID of tool call")
    tool_calls: list[dict[str, Any]] | None = Field(None, description="Tool calls made")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message creation timestamp")
    message_id: str = Field(default_factory=lambda: uuid4().hex, description="Unique message ID for deduplication")


class AttemptStrategy(BaseModel):
    """
    Base strategy for operations that may require multiple attempts.
    
    Attributes:
        max_iterations: Maximum retry attempts (default: 3)
    """
    max_iterations: int = Field(default=3, description="Maximum retry attempts per operation")


class ParallelStrategy(AttemptStrategy):
    """
    Strategy for parallel batch processing with rate limit handling.
    
    Used for operations that process large datasets in chunks with parallel execution.
    Implements adaptive concurrency control with exponential backoff on rate limit errors.
    
    Attributes:
        max_iterations: Maximum retry attempts per chunk (default: 3)
        chunk_size: Number of items per chunk (default: 2048)
        max_parallel_chunks: Maximum chunks to process concurrently (default: 16)
    """
    chunk_size: int = Field(default=2048, description="Number of items per chunk/batch")
    max_parallel_chunks: int = Field(default=16, description="Maximum chunks to process in parallel")


class RetryStrategy(AttemptStrategy):
    """
    Retry strategy for LLM operations.
    
    Controls retry behavior and parallel execution for operations that may need
    multiple attempts to succeed (e.g., LLM calls with structured output validation).

    Attributes:
        max_iterations: Maximum retry attempts per operation (default: 3)
        num_candidates: Number of parallel attempts to execute (default: 1)
    """
    num_candidates: int = Field(default=1, description="Number of parallel attempts to execute")


class Strategy(RetryStrategy):
    """
    Configuration strategy for code generation (aot/jit).

    Extends RetryStrategy with additional parameters for cost-based selection,
    early stopping, and customizable generation/verification/cost pipelines.

    Attributes:
        max_iterations: Maximum refinement iterations per candidate (default: 3)
        num_candidates: Number of candidates to generate in parallel (default: 3)
        min_candidates_for_comparison: Minimum valid candidates before early comparison (default: 1)
        accept_cost_threshold: If set, immediately accept candidate below this cost (default: None)
        compare_cost_threshold: If set, compare early when min_candidates below this cost (default: None)
        generate: Custom code generation strategy (default: None, uses runtime's)
        verify: Custom verification strategy or list of strategies (default: None, uses runtime's)
        cost: Custom cost estimation strategy (default: None, uses runtime's)
        compact: Custom code compaction strategy (default: None, uses runtime's)
    """

    num_candidates: int = Field(default=3, description="Number of parallel attempts to execute")  # Override default
    min_candidates_for_comparison: int = Field(default=1, description="Minimum candidates before early comparison")
    accept_cost_threshold: float | None = Field(default=None, description="Immediately accept if cost below threshold")
    compare_cost_threshold: float | None = Field(
        default=None, description="Compare early when min_candidates below threshold"
    )
    generate: Any | None = Field(default=None, description="Custom code generation strategy")
    verify: Any | None = Field(default=None, description="Custom verification strategy or list")
    cost: Any | None = Field(default=None, description="Custom cost estimation strategy")
    compact: Any | None = Field(default=None, description="Custom code compaction strategy")

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Tool(BaseModel):
    """
    A tool is a callable function with schema validation.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description
        input_schema: Pydantic model for input validation
        output_schema: Pydantic model for output validation
        execute: Async function to execute
        is_terminal: Whether this tool ends execution
    """

    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    execute: Callable
    is_terminal: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        """
        Custom JSON schema for the Tool class.
        
        IMPORTANT: This is NOT used to get schemas for individual tools!
        Individual tool schemas come from: tool.input_schema.model_json_schema()
        
        Why this is needed:
        - Tool has `execute: Callable` which cannot be serialized to JSON schema
        - When generating schemas for models that contain Tool (e.g., LLMInput with tools: list[Tool]),
          Pydantic needs to generate Tool's class schema to understand the type structure
        - Without this, Pydantic fails with: "Cannot generate JsonSchema for CallableSchema"
        
        Solution:
        - Remove the `execute` field from the core schema before processing
        - Let Pydantic's handler process the rest normally (proper type checking!)
        - This preserves type validation for name, description, input_schema, output_schema, is_terminal
        
        Note: input_schema/output_schema are type[BaseModel] (class objects), so their schema
        just indicates "subclass of BaseModel". Actual schemas obtained via tool.input_schema.model_json_schema().
        """
        import copy
        
        # Create a copy and remove the execute field
        modified_schema = copy.deepcopy(core_schema)
        if 'schema' in modified_schema and modified_schema['schema'].get('type') == 'model-fields':
            fields = modified_schema['schema'].get('fields', {})
            if 'execute' in fields:
                del fields['execute']
        
        # Let Pydantic process the rest with proper type checking
        return handler(modified_schema)

    async def __call__(self, *args, **kwargs) -> Any:
        """Execute the tool with validation.

        Supports multiple styles:
        - Positional BaseModel: await tool(input_obj)  where input_obj is an InputSchema instance
        - Positional string: await tool(content_str)  where tool has 'content' as first parameter
        - Keyword: await tool(field1=value1, field2=value2, ...)
        """
        # If called with a single positional argument
        if len(args) == 1:
            arg = args[0]
            # If it's a BaseModel instance, unpack it
            if isinstance(arg, BaseModel):
                input_obj = arg
                # Extract all fields from the input object as kwargs without serializing
                # Use attribute access to preserve runtime objects (e.g., ToolWrapper)
                try:
                    # Pydantic V2: model_fields contains field names
                    field_names = list(getattr(input_obj.__class__, "model_fields", {}).keys())
                except Exception:
                    field_names = []

                if field_names:
                    kwargs_from_obj = {k: getattr(input_obj, k) for k in field_names}
                else:
                    # Fallback: attempt to use model_dump but keep raw attributes where possible
                    try:
                        dumped = input_obj.model_dump()
                        kwargs_from_obj = {}
                        for k, v in dumped.items():
                            # If the attribute exists on the object, prefer the raw attribute
                            if hasattr(input_obj, k):
                                kwargs_from_obj[k] = getattr(input_obj, k)
                            else:
                                kwargs_from_obj[k] = v
                    except Exception:
                        # Last resort: iterate input schema fields
                        kwargs_from_obj = {
                            k: getattr(input_obj, k) for k in getattr(self.input_schema, "model_fields", {})
                        }

                kwargs = {**kwargs_from_obj, **kwargs}
            # If it's a string and the first input field is 'content', use it for that field
            elif isinstance(arg, str) and "content" in self.input_schema.model_fields:
                kwargs["content"] = arg
            else:
                raise TypeError(f"Tool {self.name} does not accept positional arguments of type {type(arg).__name__}")
        elif args:
            raise TypeError(f"Tool {self.name} does not accept multiple positional arguments")

        # Separate schema fields from extra parameters
        schema_fields = set(self.input_schema.model_fields.keys())
        schema_kwargs = {k: v for k, v in kwargs.items() if k in schema_fields}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in schema_fields}

        # Validate input against schema
        validated_input = self.input_schema(**schema_kwargs)

        # Extract values without serialization to preserve Python objects
        exec_kwargs = {k: getattr(validated_input, k) for k in schema_fields}

        # Add extra parameters (like 'context') that aren't part of the schema
        exec_kwargs.update(extra_kwargs)

        # Execute
        if inspect.iscoroutinefunction(self.execute):
            result = await self.execute(**exec_kwargs)
        else:
            result = self.execute(**exec_kwargs)

        # Validate output
        if isinstance(result, dict):
            validated_output = self.output_schema(**result)
        else:
            # If result is already a Pydantic model or primitive
            validated_output = self.output_schema(result=result) if hasattr(self.output_schema, "result") else result

        return validated_output

    async def execute_with_runtime(self, **kwargs) -> Any:
        """
        Execute the tool and track in runtime context.

        This is a thin wrapper around Runtime.execute() for convenience.
        Uses the global runtime to execute and track this tool call.

        Args:
            **kwargs: Tool inputs

        Returns:
            Tool output
        """
        from .runtime import get_runtime

        runtime = get_runtime()
        return await runtime.execute(self, **kwargs)


def tool(name: str | None = None, description: str | None = None, is_terminal: bool = False):
    """
    Decorator to convert a Pydantic-typed function into a Tool.

    Example:
        @tool(name="add", description="Add two numbers")
        async def add(a: int, b: int) -> int:
            return a + b
    """

    def decorator(func: Callable) -> Tool:
        # Get function metadata
        func_name = name or func.__name__
        func_desc = description or (func.__doc__ or "").strip()

        # Get type hints
        hints = get_type_hints(func)
        return_type = hints.pop("return", Any)

        # Create input schema from parameters
        input_fields = {}
        for param_name, param_type in hints.items():
            input_fields[param_name] = (param_type, ...)

        InputModel = create_model(f"{func_name}_Input", **input_fields)

        # Create output schema from return type
        if return_type == Any or return_type is None:
            OutputModel = create_model(f"{func_name}_Output", result=(Any, ...))
        elif isinstance(return_type, type) and issubclass(return_type, BaseModel):
            OutputModel = return_type
        else:
            OutputModel = create_model(f"{func_name}_Output", result=(return_type, ...))

        return Tool(
            name=func_name,
            description=func_desc,
            input_schema=InputModel,
            output_schema=OutputModel,
            execute=func,
            is_terminal=is_terminal,
        )

    return decorator


class ToolSet(BaseModel):
    """
    A collection of related tools and/or ToolSets.

    Attributes:
        name: Unique identifier for the toolset
        description: Human-readable description
        tools: List of tools and/or nested ToolSets

    Note: Using List[Union[Tool, ToolSet]] allows ToolSet to define a tree structure
    of tools, enabling more advanced Generate implementations to selectively load
    different tools based on the task at hand. This hierarchical organization supports
    dynamic tool selection strategies.
    """

    name: str
    description: str
    tools: list[Union[Tool, "ToolSet"]]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    async def load_from_mcp(cls, server_config: dict[str, Any], name: str | None = None) -> "ToolSet":
        """
        Load tools from an MCP server.

        Args:
            server_config: MCP server configuration dict
            name: Optional name for the toolset (defaults to server name)

        Returns:
            ToolSet with tools from the MCP server
        """
        from mcp_use import MCPClient

        # Create client with server config
        client = MCPClient({"mcpServers": {name or "server": server_config}})
        await client.create_all_sessions()

        session = client.get_session(name or "server")
        mcp_tools = await session.list_tools()

        # Convert MCP tools to our Tool format
        tools = []
        for mcp_tool in mcp_tools:
            # Create input schema from MCP tool's JSON schema
            input_schema = _json_schema_to_pydantic(mcp_tool.inputSchema, f"{mcp_tool.name}_Input")

            # MCP tools return CallToolResult, we'll wrap it
            output_schema = create_model(f"{mcp_tool.name}_Output", content=(Any, ...), isError=(bool, False))

            # Create execute wrapper
            async def execute_wrapper(**kwargs):
                result = await session.call_tool(mcp_tool.name, kwargs)
                return {"content": result.content, "isError": result.isError}

            tools.append(
                Tool(
                    name=mcp_tool.name,
                    description=mcp_tool.description or "",
                    input_schema=input_schema,
                    output_schema=output_schema,
                    execute=execute_wrapper,
                    is_terminal=False,
                )
            )

        return cls(name=name or "mcp_toolset", description=f"Tools from MCP server: {name or 'server'}", tools=tools)


class Skill(BaseModel):
    """
    A reusable skill/knowledge unit with code and module dependencies.

    Skills encapsulate domain-specific knowledge, patterns, and best practices
    that can be selectively loaded into agents. Each skill includes content
    (code snippets, examples, instructions) and specifies required Python modules.

    Attributes:
        name: Unique identifier for the skill
        description: Human-readable description of what this skill provides
        content: The actual skill content (code, examples, documentation, instructions)
        modules: List of Python module names that this skill depends on
    """

    name: str
    description: str
    content: str
    modules: list[str] = Field(default_factory=list, description="Python modules required for this skill")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    async def from_url(
        cls,
        urls: str | list[str],
        name: str,
        description: str,
        chunk_size: int = 2000,
        llm: Any | None = None,
        instructions: str | None = None,
        modules: list[str] | None = None,
    ) -> "Skill":
        """
        Load skill content from one or more URLs using crawl4ai and LLM summarization.

        Uses crawl4ai to fetch and parse content from URLs, chunks the content,
        and uses an LLM to generate a concise skill summary and extract relevant
        Python modules. This enables creating skills from web documentation,
        articles, tutorials, and other online resources.

        Args:
            urls: Single URL or list of URLs to load content from
            name: Name for the generated skill
            description: Description of what the skill provides
            chunk_size: Size of content chunks for LLM processing (default: 2000 chars)
            llm: Optional LLM tool to use for summarization (uses default if not provided)
            instructions: Optional specific instructions for skill generation
            modules: Optional list of Python modules (auto-detected if not provided)

        Returns:
            Skill with content from the URLs

        Raises:
            ImportError: If crawl4ai is not installed
            ValueError: If URL loading fails

        Note:
            Requires: pip install crawl4ai
            The LLM used for summarization should be fast/cheap (e.g., gpt-3.5-turbo)
        """
        try:
            from crawl4ai import AsyncWebCrawler
        except ImportError:
            raise ImportError("crawl4ai is required for Skill.from_url. Install it with: pip install crawl4ai")

        # Normalize urls to list
        if isinstance(urls, str):
            urls = [urls]

        # Use default LLM if not provided
        if llm is None:
            from .builtin_tools import LLM

            llm = LLM("gpt-4.1-mini")

        # Fetch content from URLs using crawl4ai
        crawler = AsyncWebCrawler()
        all_content = []

        for url in urls:
            try:
                result = await crawler.arun(url)
                if result.success:
                    all_content.append(f"# From: {url}\n{result.markdown}")
                else:
                    raise ValueError(f"Failed to crawl {url}: {result.error}")
            except Exception as e:
                raise ValueError(f"Error crawling {url}: {str(e)}")

        full_content = "\n\n".join(all_content)

        # Chunk content for LLM processing
        chunks = [full_content[i : i + chunk_size] for i in range(0, len(full_content), chunk_size)]

        # Use LLM to generate skill content
        summarization_prompt = f"""
Given the following content from URL(s), create a concise skill summary that:
1. Captures the key concepts and best practices
2. Provides practical examples or code snippets
3. Is organized and easy to reference
4. Lists any Python modules that would be needed (if applicable)

Content to summarize:
{full_content[:5000]}  # Use first 5000 chars to save tokens

{f"Additional instructions: {instructions}" if instructions else ""}

Format the response as a markdown skill guide.
"""

        # Get LLM response (assuming llm is a Tool)
        summary_response = await llm(content=summarization_prompt)

        # Extract content from response
        if hasattr(summary_response, "content"):
            skill_content = summary_response.content
        else:
            skill_content = str(summary_response)

        # Auto-detect modules if not provided
        detected_modules = modules or []
        if not modules:
            # Simple heuristic: look for common module names in content
            common_modules = [
                "pandas",
                "numpy",
                "requests",
                "beautifulsoup4",
                "sqlalchemy",
                "flask",
                "django",
                "sklearn",
                "pytorch",
                "tensorflow",
                "matplotlib",
                "seaborn",
                "plotly",
                "asyncio",
                "aiohttp",
            ]
            for module in common_modules:
                if module.lower() in full_content.lower():
                    detected_modules.append(module)

        return cls(name=name, description=description, content=skill_content, modules=detected_modules)


class SkillSet(BaseModel):
    """
    A collection of related skills.

    SkillSets group multiple skills together for organizational purposes,
    allowing agents to have access to collections of domain-specific knowledge.

    Attributes:
        name: Unique identifier for the skillset
        description: Human-readable description of the skillset
        skills: List of skills in this collection
    """

    name: str
    description: str
    skills: list[Skill]

    model_config = ConfigDict(arbitrary_types_allowed=True)


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

    def get_all_tools(self) -> list[Tool]:
        """Flatten all tools from tools and toolsets."""
        all_tools = []
        for item in self.tools:
            if isinstance(item, Tool):
                all_tools.append(item)
            elif isinstance(item, ToolSet):
                all_tools.extend(item.tools)
        return all_tools

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        for tool in self.get_all_tools():
            if tool.name == name:
                return tool
        return None

    def save_to_file(self, path: Union[str, "Path"]) -> None:
        """
        Save this agent to a JSON file.

        Args:
            path: File path to save to
        """
        from .serialization import save_agent_to_file

        save_agent_to_file(self, path)

    @classmethod
    def load_from_file(cls, path: Union[str, "Path"]) -> "Agent":
        """
        Load an agent from a JSON file.

        Args:
            path: File path to load from

        Returns:
            Loaded Agent
        """
        from .serialization import load_agent_from_file

        return load_agent_from_file(path)

    async def aot(self, cache: bool = True, strategy: Optional["Strategy"] = None) -> Tool:
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
        from .runtime import get_runtime

        runtime = get_runtime()
        return await runtime.aot(self, cache=cache, strategy=strategy)

    async def jit(self, strategy: Optional["Strategy"] = None, **kwargs) -> Any:
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
        from .runtime import get_runtime

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
            from langchain.agents import AgentExecutor
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


def _json_schema_to_pydantic(schema: dict[str, Any], model_name: str) -> type[BaseModel]:
    """Convert JSON schema to Pydantic model."""
    if not schema or schema.get("type") != "object":
        # Fallback for simple schemas
        return create_model(model_name, **{"input": (Any, ...)})

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    fields = {}
    for name, prop in properties.items():
        field_type = _json_type_to_python(prop.get("type", "string"))
        default = ... if name in required else None
        fields[name] = (field_type, default)

    return create_model(model_name, **fields)


def _json_type_to_python(json_type: str) -> type:
    """Convert JSON schema type to Python type."""
    type_map = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(json_type, Any)


__all__ = [
    "Message",
    "RetryStrategy",
    "Strategy",
    "Tool",
    "tool",
    "ToolSet",
    "Skill",
    "SkillSet",
    "Agent",
]
