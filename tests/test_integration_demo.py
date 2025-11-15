"""
Live demonstration of MCP and OpenAPI tool loading and invocation.

This test suite demonstrates that the tools actually work end-to-end:
1. Load tools from real MCP filesystem server
2. Load tools from real OpenAPI server
3. Actually invoke the tools with real operations
"""

import tempfile
import time
from multiprocessing import Process
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from a1 import Agent, ToolSet, tool

# ============================================================================
# Test 1: @tool decorator and direct function passing
# ============================================================================


@tool(name="multiply", description="Multiply two numbers")
async def multiply(a: int, b: int) -> int:
    return a * b


async def divide(a: float, b: float) -> float:
    """Divide two numbers."""
    return a / b


@pytest.mark.asyncio
async def test_tool_decorator_and_function_passing():
    """Test that @tool decorator works and raw functions can be passed to Agent."""
    print("\n" + "=" * 70)
    print("TEST: @tool Decorator and Direct Function Passing")
    print("=" * 70)

    # Test direct function passing
    agent = Agent(
        name="math_agent",
        tools=[multiply, divide],  # Mix of @tool and raw function
    )

    print(f"\n✓ Created agent with {len(agent.tools)} tools")
    for tool in agent.tools:
        print(f"  - {tool.name}: {tool.description}")

    assert len(agent.tools) == 2
    assert any(t.name == "multiply" for t in agent.tools)
    assert any(t.name == "divide" for t in agent.tools)


# ============================================================================
# Test 2: MCP Server Integration
# ============================================================================


@pytest.mark.asyncio
async def test_mcp_integration():
    """Test MCP server integration with real filesystem server."""
    print("\n" + "=" * 70)
    print("TEST: MCP Server Integration (Real Filesystem Server)")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Resolve the real path (handles macOS symlinks like /var -> /private/var)
        tmpdir_resolved = str(Path(tmpdir).resolve())

        # Configure real MCP filesystem server
        config = {
            "mcpServers": {
                "filesystem": {
                    "command": "npx",
                    "args": [
                        "-y",
                        "@modelcontextprotocol/server-filesystem",
                        tmpdir_resolved,
                    ],
                }
            }
        }

        print(f"\n📁 Using temp directory: {tmpdir_resolved}")
        print("🔄 Loading tools from MCP filesystem server...")

        # Load tools from MCP server
        toolset = await ToolSet.from_mcp_servers(config)

        print(f"\n✓ Loaded {len(toolset.tools)} tools from MCP server:")
        for tool in toolset.tools[:5]:  # Show first 5
            print(f"  - {tool.name}")
        if len(toolset.tools) > 5:
            print(f"  ... and {len(toolset.tools) - 5} more")

        assert len(toolset.tools) > 0

        # Find and use the write_file tool
        write_tool = next((t for t in toolset.tools if t.name == "write_file"), None)
        read_tool = next((t for t in toolset.tools if t.name == "read_text_file"), None)

        assert write_tool is not None, "write_file tool not found"
        assert read_tool is not None, "read_text_file tool not found"

        print("\n📝 Testing MCP tools with actual file operations...")

        # Write a file (use resolved path)
        test_file = Path(tmpdir_resolved) / "demo.txt"
        test_content = "Hello from MCP integration test!"

        print(f"\n  Writing: '{test_content}'")
        write_result = await write_tool(path=str(test_file), content=test_content)
        print(f"  Write result: isError={write_result.isError}")
        assert write_result.isError is False

        # Read it back
        print(f"\n  Reading back from: {test_file.name}")
        read_result = await read_tool(path=str(test_file))
        print(f"  Read result: isError={read_result.isError}")
        print(f"  Content: {str(read_result.content)[:100]}...")
        assert read_result.isError is False

        # Verify
        actual_content = test_file.read_text()
        assert actual_content == test_content
        print("\n  ✅ Round-trip successful! File content matches.")


# ============================================================================
# Test 3: OpenAPI/FastAPI Integration
# ============================================================================


class CalculateRequest(BaseModel):
    operation: str
    a: int
    b: int


def create_demo_api():
    """Create a demo FastAPI server."""
    app = FastAPI(title="Demo Calculator API", version="1.0.0")

    @app.post("/calculate")
    async def calculate(req: CalculateRequest):
        """Perform arithmetic operations."""
        if req.operation == "add":
            return {"result": req.a + req.b}
        elif req.operation == "subtract":
            return {"result": req.a - req.b}
        elif req.operation == "multiply":
            return {"result": req.a * req.b}
        else:
            return {"error": "Unknown operation"}

    @app.get("/status")
    async def status():
        """Get API status."""
        return {"status": "running", "uptime": "demo"}

    return app


def run_demo_server():
    """Run the demo server."""
    app = create_demo_api()
    uvicorn.run(app, host="127.0.0.1", port=8767, log_level="error")


@pytest.mark.asyncio
async def test_openapi_integration():
    """Test OpenAPI integration with real FastAPI server."""
    print("\n" + "=" * 70)
    print("TEST: OpenAPI Integration (Real FastAPI Server)")
    print("=" * 70)

    # Start server in background
    server_process = Process(target=run_demo_server, daemon=True)
    server_process.start()

    try:
        # Wait for server to start
        print("\n🚀 Starting FastAPI server...")
        time.sleep(2)

        server_url = "http://127.0.0.1:8767"
        print(f"📡 Loading tools from OpenAPI server at {server_url}...")

        # Load tools from OpenAPI spec
        toolset = await ToolSet.from_openapi(server_url)

        print(f"\n✓ Loaded {len(toolset.tools)} tools from OpenAPI server:")
        for tool in toolset.tools:
            print(f"  - {tool.name}")

        assert len(toolset.tools) > 0

        # Find and use the calculate tool
        calc_tool = next((t for t in toolset.tools if "calculate" in t.name.lower()), None)
        assert calc_tool is not None, "calculate tool not found"

        print("\n🧮 Testing OpenAPI tool with actual HTTP requests...")

        # Test addition
        print("\n  Calling: calculate(operation='add', a=15, b=27)")
        result1 = await calc_tool(operation="add", a=15, b=27)
        print(f"  Response: status_code={result1.status_code}")
        print(f"  Result: {result1.data['result']}")
        assert result1.data["result"] == 42

        # Test multiplication
        print("\n  Calling: calculate(operation='multiply', a=6, b=7)")
        result2 = await calc_tool(operation="multiply", a=6, b=7)
        print(f"  Response: status_code={result2.status_code}")
        print(f"  Result: {result2.data['result']}")
        assert result2.data["result"] == 42

        print("\n  ✅ Both calculations successful!")

    finally:
        # Cleanup
        server_process.terminate()
        server_process.join(timeout=5)
