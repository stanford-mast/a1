"""ToolSet model - collection of related tools."""

from typing import Any, Union

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, create_model

from .tool import Tool


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


def _openapi_type_to_python(openapi_type: str) -> type:
    """Convert OpenAPI type to Python type."""
    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(openapi_type, Any)


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
        try:
            from mcp_use import MCPClient
        except ImportError:
            raise ImportError("mcp_use is required for ToolSet.load_from_mcp. Install it with: pip install mcp_use")

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

    @classmethod
    async def from_mcp_servers(cls, config: dict[str, Any]) -> "ToolSet":
        """
        Load tools from one or more MCP servers.

        Args:
            config: Full MCP configuration dict in format:
                {
                    "mcpServers": {
                        "server_name_1": {"command": "...", "args": [...], ...},
                        "server_name_2": {"command": "...", "args": [...], ...},
                        ...
                    }
                }

        Returns:
            ToolSet containing all tools from all configured MCP servers

        Example:
            >>> config = {
            ...     "mcpServers": {
            ...         "filesystem": {
            ...             "command": "npx",
            ...             "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
            ...         }
            ...     }
            ... }
            >>> toolset = await ToolSet.from_mcp_servers(config)
        """
        try:
            from mcp_use import MCPClient
        except ImportError:
            raise ImportError("mcp_use is required for ToolSet.from_mcp_servers. Install it with: pip install mcp_use")

        # Create client with full config
        client = MCPClient(config)
        await client.create_all_sessions()

        # Collect tools from all servers
        all_tools = []
        server_names = list(config.get("mcpServers", {}).keys())

        for server_name in server_names:
            session = client.get_session(server_name)
            mcp_tools = await session.list_tools()

            # Convert MCP tools to our Tool format
            for mcp_tool in mcp_tools:
                # Create input schema from MCP tool's JSON schema
                input_schema = _json_schema_to_pydantic(mcp_tool.inputSchema, f"{mcp_tool.name}_Input")

                # MCP tools return CallToolResult
                output_schema = create_model(f"{mcp_tool.name}_Output", content=(Any, ...), isError=(bool, False))

                # Create execute wrapper that captures the current tool/session
                def make_execute_wrapper(tool_name: str, sess):
                    async def execute_wrapper(**kwargs):
                        # Filter out None values - MCP servers don't accept null for optional params
                        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                        result = await sess.call_tool(tool_name, filtered_kwargs)
                        return {"content": result.content, "isError": result.isError}

                    return execute_wrapper

                execute_fn = make_execute_wrapper(mcp_tool.name, session)

                all_tools.append(
                    Tool(
                        name=mcp_tool.name,
                        description=mcp_tool.description or "",
                        input_schema=input_schema,
                        output_schema=output_schema,
                        execute=execute_fn,
                        is_terminal=False,
                    )
                )

        server_list = ", ".join(server_names) if server_names else "none"
        return cls(
            name="mcp_toolset",
            description=f"Tools from MCP servers: {server_list}",
            tools=all_tools,
        )

    @classmethod
    async def from_openapi(cls, endpoint: str, name: str | None = None) -> "ToolSet":
        """
        Load tools from an OpenAPI/Swagger specification.

        This method automatically detects and fetches the OpenAPI spec from common
        endpoint paths, then converts each API operation into a Tool.

        Args:
            endpoint: Base URL of the API server (e.g., "http://localhost:8000")
            name: Optional name for the toolset (defaults to API title from spec)

        Returns:
            ToolSet containing tools for each API operation

        Example:
            >>> # Load from FastAPI server
            >>> toolset = await ToolSet.from_openapi("http://localhost:8000")
            >>>
            >>> # Use the tools
            >>> for tool in toolset.tools:
            ...     print(f"{tool.name}: {tool.description}")
        """
        # Try common OpenAPI spec locations
        spec_paths = [
            "/openapi.json",
            "/openapi.yaml",
            "/swagger.json",
            "/swagger.yaml",
            "/docs/openapi.json",  # Alternative locations
            "/api/openapi.json",
        ]

        spec = None
        spec_url = None

        async with httpx.AsyncClient() as client:
            for path in spec_paths:
                try:
                    url = endpoint.rstrip("/") + path
                    response = await client.get(url, timeout=10.0)

                    if response.status_code == 200:
                        # Try parsing as JSON first
                        if path.endswith(".json"):
                            spec = response.json()
                        else:
                            # Parse YAML
                            spec = yaml.safe_load(response.text)

                        spec_url = url
                        break
                except Exception:
                    # Continue trying other paths
                    continue

        if not spec:
            raise ValueError(f"Could not find OpenAPI spec at {endpoint}. Tried: {', '.join(spec_paths)}")

        # Extract API info
        api_title = spec.get("info", {}).get("title", "API")
        api_version = spec.get("info", {}).get("version", "unknown")
        toolset_name = name or f"{api_title.lower().replace(' ', '_')}_api"

        # Convert OpenAPI operations to Tools
        tools = []
        paths = spec.get("paths", {})

        for path, path_item in paths.items():
            # Process each HTTP method
            for method in ["get", "post", "put", "patch", "delete", "options", "head"]:
                if method not in path_item:
                    continue

                operation = path_item[method]

                # Get operation details
                operation_id = operation.get("operationId", f"{method}_{path.replace('/', '_')}")
                summary = operation.get("summary", "")
                description = operation.get("description", summary)

                # Build input schema from parameters and requestBody
                input_fields = {}

                # Add path parameters
                parameters = operation.get("parameters", [])
                for param in parameters:
                    param_name = param.get("name")
                    param_schema = param.get("schema", {})
                    param_type = _openapi_type_to_python(param_schema.get("type", "string"))
                    param_required = param.get("required", False)

                    if param_required:
                        input_fields[param_name] = (param_type, ...)
                    else:
                        input_fields[param_name] = (param_type, None)

                # Add request body fields
                request_body = operation.get("requestBody")
                if request_body:
                    content = request_body.get("content", {})
                    json_content = content.get("application/json", {})
                    body_schema = json_content.get("schema", {})

                    if body_schema.get("type") == "object":
                        properties = body_schema.get("properties", {})
                        required = body_schema.get("required", [])

                        for prop_name, prop_schema in properties.items():
                            prop_type = _openapi_type_to_python(prop_schema.get("type", "string"))
                            if prop_name in required:
                                input_fields[prop_name] = (prop_type, ...)
                            else:
                                input_fields[prop_name] = (prop_type, None)

                # Create input schema
                if input_fields:
                    input_model = create_model(f"{operation_id}_Input", **input_fields)
                else:
                    input_model = create_model(f"{operation_id}_Input")

                # Create output schema (simplified - just returns response)
                output_model = create_model(
                    f"{operation_id}_Output",
                    status_code=(int, ...),
                    data=(Any, ...),
                )

                # Create execute function
                def make_execute_fn(url_path: str, http_method: str, base_url: str):
                    async def execute(**kwargs):
                        async with httpx.AsyncClient() as client:
                            # Build full URL with path parameters
                            full_path = url_path
                            for key, value in list(kwargs.items()):
                                if f"{{{key}}}" in full_path:
                                    full_path = full_path.replace(f"{{{key}}}", str(value))
                                    kwargs.pop(key)

                            url = base_url.rstrip("/") + full_path

                            # Make request
                            if http_method in ["get", "delete", "head", "options"]:
                                response = await client.request(http_method.upper(), url, params=kwargs)
                            else:
                                response = await client.request(http_method.upper(), url, json=kwargs)

                            # Return response
                            try:
                                data = response.json()
                            except Exception:
                                data = response.text

                            return {"status_code": response.status_code, "data": data}

                    return execute

                execute_fn = make_execute_fn(path, method, endpoint)

                tools.append(
                    Tool(
                        name=operation_id,
                        description=description or f"{method.upper()} {path}",
                        input_schema=input_model,
                        output_schema=output_model,
                        execute=execute_fn,
                        is_terminal=False,
                    )
                )

        return cls(
            name=toolset_name,
            description=f"Tools from {api_title} v{api_version} ({spec_url})",
            tools=tools,
        )


__all__ = ["ToolSet"]
