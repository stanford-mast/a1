"""
Test OpenAPI integration with a REAL running HTTP server.

This test starts an actual HTTP server and tests ToolSet.from_openapi()
with real HTTP requests.
"""

import time
from multiprocessing import Process

import pytest
import uvicorn

pytest.importorskip("fastapi", reason="FastAPI not installed")

from fastapi import FastAPI
from pydantic import BaseModel

from a1 import ToolSet


class CalculateRequest(BaseModel):
    """Request model for calculation."""

    operation: str
    a: int
    b: int


def create_test_server():
    """Create a FastAPI server for testing."""
    app = FastAPI(
        title="Test Calculator API",
        version="1.0.0",
        description="A test API for OpenAPI integration",
    )

    @app.post("/calculate")
    async def calculate(req: CalculateRequest):
        """Perform a calculation."""
        if req.operation == "add":
            return {"result": req.a + req.b}
        elif req.operation == "multiply":
            return {"result": req.a * req.b}
        else:
            return {"error": "Unknown operation"}

    @app.get("/status")
    async def get_status():
        """Get API status."""
        return {"status": "healthy", "version": "1.0.0"}

    @app.get("/users/{user_id}")
    async def get_user(user_id: int):
        """Get user information by ID."""
        return {"id": user_id, "name": f"User{user_id}", "active": True}

    return app


def run_server():
    """Run the test server."""
    app = create_test_server()
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="error")


@pytest.fixture(scope="module")
def test_server():
    """Start a real HTTP server for testing."""
    # Start server in background process
    server_process = Process(target=run_server, daemon=True)
    server_process.start()

    # Wait for server to start
    time.sleep(2)

    yield "http://127.0.0.1:8765"

    # Cleanup
    server_process.terminate()
    server_process.join(timeout=5)


class TestOpenAPILiveIntegration:
    """Test OpenAPI integration with a live server."""

    @pytest.mark.asyncio
    async def test_load_tools_from_live_openapi_server(self, test_server):
        """Test loading tools from a real OpenAPI server."""
        # Load tools from the live server
        toolset = await ToolSet.from_openapi(test_server)

        assert toolset is not None
        assert len(toolset.tools) > 0

        tool_names = [tool.name for tool in toolset.tools]
        print(f"\n✓ Loaded {len(toolset.tools)} tools from OpenAPI server")
        print(f"  Tools: {tool_names}")

        # Verify we have the expected endpoints
        assert any("calculate" in name.lower() for name in tool_names)
        assert any("status" in name.lower() for name in tool_names)
        assert any("user" in name.lower() for name in tool_names)

    @pytest.mark.asyncio
    async def test_invoke_openapi_tool_post_request(self, test_server):
        """Test invoking a POST endpoint tool."""
        toolset = await ToolSet.from_openapi(test_server)

        # Find the calculate tool (POST /calculate)
        calc_tool = None
        for tool in toolset.tools:
            if "calculate" in tool.name.lower():
                calc_tool = tool
                break

        assert calc_tool is not None, "Calculate tool not found"
        print(f"\n✓ Found calculate tool: {calc_tool.name}")

        # Invoke the tool with actual HTTP request
        result = await calc_tool(operation="add", a=10, b=5)

        print("  Request: add(10, 5)")
        print(f"  Response: {result}")

        assert result is not None
        # Result is a Pydantic model
        assert hasattr(result, "status_code")
        assert hasattr(result, "data")
        assert result.status_code == 200
        assert result.data["result"] == 15

        print(f"✓ POST tool execution works! Result: {result.data['result']}")

    @pytest.mark.asyncio
    async def test_invoke_openapi_tool_get_request(self, test_server):
        """Test invoking a GET endpoint tool."""
        toolset = await ToolSet.from_openapi(test_server)

        # Find the status tool (GET /status)
        status_tool = None
        for tool in toolset.tools:
            if "status" in tool.name.lower():
                status_tool = tool
                break

        assert status_tool is not None, "Status tool not found"
        print(f"\n✓ Found status tool: {status_tool.name}")

        # Invoke the tool
        result = await status_tool()

        print(f"  Response: {result}")

        assert result is not None
        assert result.status_code == 200
        assert result.data["status"] == "healthy"

        print(f"✓ GET tool execution works! Status: {result.data['status']}")

    @pytest.mark.asyncio
    async def test_invoke_openapi_tool_with_path_params(self, test_server):
        """Test invoking a tool with path parameters."""
        toolset = await ToolSet.from_openapi(test_server)

        # Find the get_user tool (GET /users/{user_id})
        user_tool = None
        for tool in toolset.tools:
            if "user" in tool.name.lower() and "{" not in tool.name:
                user_tool = tool
                break

        assert user_tool is not None, "User tool not found"
        print(f"\n✓ Found user tool: {user_tool.name}")

        # Invoke with path parameter
        result = await user_tool(user_id=42)

        print("  Request: get_user(user_id=42)")
        print(f"  Response: {result}")

        assert result is not None
        assert result.status_code == 200
        assert result.data["id"] == 42
        assert result.data["name"] == "User42"

        print(f"✓ Path parameter handling works! User: {result.data['name']}")

    @pytest.mark.asyncio
    async def test_invoke_multiple_operations(self, test_server):
        """Test invoking multiple different operations."""
        toolset = await ToolSet.from_openapi(test_server)

        # Find calculate tool
        calc_tool = next((t for t in toolset.tools if "calculate" in t.name.lower()), None)
        assert calc_tool is not None

        # Test multiple operations
        result1 = await calc_tool(operation="add", a=100, b=50)
        assert result1.data["result"] == 150

        result2 = await calc_tool(operation="multiply", a=7, b=8)
        assert result2.data["result"] == 56

        print("\n✓ Multiple operations work!")
        print(f"  add(100, 50) = {result1.data['result']}")
        print(f"  multiply(7, 8) = {result2.data['result']}")

    @pytest.mark.asyncio
    async def test_toolset_metadata_from_openapi(self, test_server):
        """Test that toolset has correct metadata from OpenAPI spec."""
        toolset = await ToolSet.from_openapi(test_server)

        # Check toolset name and description
        assert "test_calculator_api" in toolset.name.lower()
        assert "Test Calculator API" in toolset.description
        assert "1.0.0" in toolset.description

        print("\n✓ Toolset metadata:")
        print(f"  Name: {toolset.name}")
        print(f"  Description: {toolset.description}")

        # Check tool metadata
        for tool in toolset.tools:
            assert tool.name is not None
            assert tool.description is not None
            assert tool.input_schema is not None
            assert tool.output_schema is not None
            print(f"  - {tool.name}: {tool.description[:50]}...")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
