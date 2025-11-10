"""
Core models for a1 agent compiler.

This module defines the fundamental building blocks:
- Agent: Composed of tools and terminal conditions
- Tool: Executable function with schema validation
- ToolSet: Collection of related tools
"""

from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints
from enum import Enum
import inspect
from pydantic import BaseModel, Field, create_model, ConfigDict


class Message(BaseModel):
    """Chat message in a context."""
    role: str = Field(..., description="Message role (user, assistant, system, tool)")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(None, description="Name of tool or function")
    tool_call_id: Optional[str] = Field(None, description="ID of tool call")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(None, description="Tool calls made")


class Strategy(BaseModel):
    """
    Configuration strategy for code generation (aot/jit).
    
    Controls parallel generation, early stopping, and cost-based selection.
    
    Attributes:
        max_iterations: Maximum refinement iterations per candidate (default: 3)
        num_candidates: Number of candidates to generate in parallel (default: 1)
        min_candidates_for_comparison: Minimum valid candidates before early comparison (default: 1)
        accept_cost_threshold: If set, immediately accept candidate below this cost (default: None)
        compare_cost_threshold: If set, compare early when min_candidates below this cost (default: None)
    """
    max_iterations: int = Field(default=3, description="Maximum refinement iterations per candidate")
    num_candidates: int = Field(default=1, description="Number of candidates to generate in parallel")
    min_candidates_for_comparison: int = Field(default=1, description="Minimum candidates before early comparison")
    accept_cost_threshold: Optional[float] = Field(default=None, description="Immediately accept if cost below threshold")
    compare_cost_threshold: Optional[float] = Field(default=None, description="Compare early when min_candidates below threshold")


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
                # Extract all fields from the input object as kwargs
                if hasattr(input_obj, 'model_dump'):
                    kwargs = {**input_obj.model_dump(), **kwargs}
                else:
                    # Fallback for non-Pydantic objects
                    kwargs = {**{k: getattr(input_obj, k) for k in self.input_schema.model_fields}, **kwargs}
            # If it's a string and the first input field is 'content', use it for that field
            elif isinstance(arg, str) and 'content' in self.input_schema.model_fields:
                kwargs['content'] = arg
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
            validated_output = self.output_schema(result=result) if hasattr(self.output_schema, 'result') else result
        
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


def tool(
    name: Optional[str] = None,
    description: Optional[str] = None,
    is_terminal: bool = False
):
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
        return_type = hints.pop('return', Any)
        
        # Create input schema from parameters
        input_fields = {}
        for param_name, param_type in hints.items():
            input_fields[param_name] = (param_type, ...)
        
        InputModel = create_model(
            f"{func_name}_Input",
            **input_fields
        )
        
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
            is_terminal=is_terminal
        )
    
    return decorator


class ToolSet(BaseModel):
    """
    A collection of related tools.
    
    Attributes:
        name: Unique identifier for the toolset
        description: Human-readable description
        tools: List of tools in this set
    """
    name: str
    description: str
    tools: List[Tool]
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    @classmethod
    async def load_from_mcp(cls, server_config: Dict[str, Any], name: Optional[str] = None) -> "ToolSet":
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
            input_schema = _json_schema_to_pydantic(
                mcp_tool.inputSchema,
                f"{mcp_tool.name}_Input"
            )
            
            # MCP tools return CallToolResult, we'll wrap it
            output_schema = create_model(
                f"{mcp_tool.name}_Output",
                content=(Any, ...),
                isError=(bool, False)
            )
            
            # Create execute wrapper
            async def execute_wrapper(**kwargs):
                result = await session.call_tool(mcp_tool.name, kwargs)
                return {"content": result.content, "isError": result.isError}
            
            tools.append(Tool(
                name=mcp_tool.name,
                description=mcp_tool.description or "",
                input_schema=input_schema,
                output_schema=output_schema,
                execute=execute_wrapper,
                is_terminal=False
            ))
        
        return cls(
            name=name or "mcp_toolset",
            description=f"Tools from MCP server: {name or 'server'}",
            tools=tools
        )


class Agent(BaseModel):
    """
    An agent is a composition of tools with defined behavior.
    
    Attributes:
        name: Unique identifier for the agent (default: "agent")
        description: Human-readable description of agent's purpose (default: "")
        input_schema: Pydantic model for agent input (default: str wrapped in Input model)
        output_schema: Pydantic model for agent output (default: str wrapped in Output model)
        tools: List of tools or toolsets available to the agent
    """
    name: str = "agent"
    description: str = ""
    input_schema: type[BaseModel] = Field(default_factory=lambda: create_model("Input", input=(str, ...)))
    output_schema: type[BaseModel] = Field(default_factory=lambda: create_model("Output", output=(str, ...)))
    tools: List[Union[Tool, ToolSet]] = Field(default_factory=list)
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def get_all_tools(self) -> List[Tool]:
        """Flatten all tools from tools and toolsets."""
        all_tools = []
        for item in self.tools:
            if isinstance(item, Tool):
                all_tools.append(item)
            elif isinstance(item, ToolSet):
                all_tools.extend(item.tools)
        return all_tools
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        for tool in self.get_all_tools():
            if tool.name == name:
                return tool
        return None
    
    async def aot(self, cache: bool = True) -> Tool:
        """
        Ahead-of-time compile this agent to a tool.
        
        Uses the global runtime to compile the agent. This is a thin wrapper
        around Runtime.aot() for convenience.
        
        Args:
            cache: Whether to use cached compilation (default: True)
        
        Returns:
            Tool that executes the compiled agent
        """
        from .runtime import get_runtime
        runtime = get_runtime()
        return await runtime.aot(self, cache=cache)
    
    async def jit(self, **kwargs) -> Any:
        """
        Just-in-time execute this agent.
        
        Uses the global runtime to execute the agent. This is a thin wrapper
        around Runtime.jit() for convenience.
        
        Args:
            **kwargs: Input arguments matching this agent's input_schema
        
        Returns:
            Output from the agent
        """
        from .runtime import get_runtime
        runtime = get_runtime()
        return await runtime.jit(self, **kwargs)
    
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
        if hasattr(langchain_agent, 'tools'):
            for lc_tool in langchain_agent.tools:
                # Convert LangChain tool to a1 Tool
                # LangChain tools have name, description, func
                input_schema = create_model(
                    f"{lc_tool.name}_Input",
                    input=(str, ...)  # Simplified - LangChain tools often just take string input
                )
                output_schema = create_model(
                    f"{lc_tool.name}_Output",
                    result=(str, ...)
                )
                
                async def execute_wrapper(input: str):
                    return {"result": lc_tool.func(input)}
                
                tools.append(Tool(
                    name=lc_tool.name,
                    description=lc_tool.description or "",
                    input_schema=input_schema,
                    output_schema=output_schema,
                    execute=execute_wrapper,
                    is_terminal=False
                ))
        
        # Create agent
        return cls(
            name=getattr(langchain_agent, 'name', 'langchain_agent'),
            description=getattr(langchain_agent, 'description', 'Converted from LangChain'),
            input_schema=create_model("Input", query=(str, ...)),
            output_schema=create_model("Output", response=(str, ...)),
            tools=tools,
            terminal_tools=["done"]
        )


def _json_schema_to_pydantic(schema: Dict[str, Any], model_name: str) -> type[BaseModel]:
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
    "Strategy",
    "Tool",
    "tool",
    "ToolSet",
    "Agent",
]
