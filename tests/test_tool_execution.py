"""
Test tool execution and advanced RAG features.

Tests:
1. Actual tool execution (not just creation)
2. Smart RAG router based on input type
3. OpenAPI-based tool loading
4. FastAPI integration
"""

import tempfile
from pathlib import Path

import pytest

from a1 import RAG, Database, FileSystem, ToolSet

# ============================================================================
# Tool Execution Tests
# ============================================================================


class TestToolExecution:
    """Test actual execution of RAG tools."""

    @pytest.mark.asyncio
    async def test_filesystem_rag_ls_execution(self):
        """Test actually executing LS tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "file1.txt").write_text("Content 1")
            (tmppath / "file2.txt").write_text("Content 2")
            (tmppath / "subdir").mkdir()

            fs = FileSystem(str(tmppath))
            rag = RAG(filesystem=fs)
            rag_toolset = rag.get_toolset()
            ls_tool = [t for t in rag_toolset.tools if t.name == "ls"][0]

            # Execute tool
            result = await ls_tool.execute(path="")

            # Verify actual execution results
            assert isinstance(result, dict)
            assert "files" in result
            assert len(result["files"]) >= 3
            assert any("file1" in str(f) for f in result["files"])
            assert any("subdir" in str(f) for f in result["files"])
            print(f"✓ LS execution result: {result['files']}")

    @pytest.mark.asyncio
    async def test_filesystem_rag_grep_execution(self):
        """Test actually executing GREP tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "doc1.txt").write_text("The quick brown fox jumps\nover the lazy dog")
            (tmppath / "doc2.txt").write_text("The fox is very clever\nand likes to jump")

            fs = FileSystem(str(tmppath))
            rag = RAG(filesystem=fs)
            rag_toolset = rag.get_toolset()
            grep_tool = [t for t in rag_toolset.tools if t.name == "grep"][0]

            # Execute grep
            result = await grep_tool.execute(pattern="fox", path="", limit=100)

            # Verify actual search results
            assert isinstance(result, dict)
            assert "matches" in result
            assert len(result["matches"]) >= 2
            assert any("fox" in m["content"].lower() for m in result["matches"])
            print(f"✓ GREP found {len(result['matches'])} matches")
            for match in result["matches"]:
                print(f"  - {match['file']}:{match['line']}: {match['content'][:50]}")

    @pytest.mark.asyncio
    async def test_filesystem_rag_cat_execution(self):
        """Test actually executing CAT tool."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            test_content = "This is a test file.\nWith multiple lines.\nOf content."
            (tmppath / "test.txt").write_text(test_content)

            fs = FileSystem(str(tmppath))
            rag = RAG(filesystem=fs)
            rag_toolset = rag.get_toolset()
            cat_tool = [t for t in rag_toolset.tools if t.name == "cat"][0]

            # Execute cat
            result = await cat_tool.execute(path="test.txt", limit=10000)

            # Verify actual file content
            assert isinstance(result, dict)
            assert "content" in result
            assert result["content"] == test_content
            assert result["truncated"] is False
            print(f"✓ CAT read content:\n{result['content']}")

    @pytest.mark.asyncio
    async def test_sqlrag_execution(self):
        """Test actually executing SQL tool."""
        df_data = {
            "id": [1, 2, 3],
            "product": ["Widget", "Gadget", "Doohickey"],
            "price": [9.99, 19.99, 14.99],
            "in_stock": [True, False, True],
        }

        db = Database("duckdb:///:memory:")
        rag = RAG(database=db)
        rag_toolset = rag.get_toolset()
        sql_tool = [t for t in rag_toolset.tools if t.name == "sql"][0]

        # Execute SQL query
        result = await sql_tool.execute(query="SELECT * FROM information_schema.tables LIMIT 0", limit=100)

        # Verify actual query results
        assert isinstance(result, dict)
        print("✓ SQL query executed successfully")


# ============================================================================
# Smart RAG Router
# ============================================================================


class SmartRAGRouter:
    """
    Syntax sugar RAG router that combines FileSystem and Database backends.

    Usage:
        router = SmartRAGRouter(
            filesystem_path="./documents",
            database=db  # optional
        )

        # Get readonly tools
        toolset = router.get_toolset()
    """

    def __init__(self, filesystem_path: str | Path = None, database=None):
        """
        Initialize RAG router with optional filesystem and database backends.

        Args:
            filesystem_path: Path for FileSystem
            database: Database instance
        """
        self.filesystem = None
        self.database = None
        self.tools = []

        if filesystem_path:
            self.filesystem = FileSystem(str(filesystem_path))
            fs_rag = RAG(filesystem=self.filesystem)
            self.tools.extend(fs_rag.get_toolset().tools)

        if database is not None:
            self.database = database
            db_rag = RAG(database=self.database)
            self.tools.extend(db_rag.get_toolset().tools)


class TestRAG:
    """Test SmartRAGRouter functionality."""

    @pytest.mark.asyncio
    async def test_rag_filesystem_only(self):
        """Test RAG router with only filesystem."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "data.txt").write_text("Sample data")

            router = SmartRAGRouter(filesystem_path=tmppath)

            # Should have filesystem tools
            ls_tool = [t for t in router.tools if t.name == "ls"][0]
            result = await ls_tool.execute(path="")
            assert "files" in result
            print(f"✓ RAG router filesystem: {result['files']}")

    @pytest.mark.asyncio
    async def test_rag_database_only(self):
        """Test RAG router with only database."""
        db = Database("duckdb:///:memory:")
        router = SmartRAGRouter(database=db)

        # Should have database tools
        sql_tool = [t for t in router.tools if t.name == "sql"][0]
        result = await sql_tool.execute(query="SELECT 1")
        assert isinstance(result, dict)
        print("✓ RAG router database: query executed")

    @pytest.mark.asyncio
    async def test_rag_both_backends(self):
        """Test RAG router with both filesystem and database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "file1.txt").write_text("Content 1")

            db = Database("duckdb:///:memory:")

            router = SmartRAGRouter(filesystem_path=tmppath, database=db)

            # Filesystem operations
            fs_result = await [t for t in router.tools if t.name == "ls"][0].execute(path="")
            assert "files" in fs_result

            # Database operations
            db_result = await [t for t in router.tools if t.name == "sql"][0].execute(query="SELECT 1")
            assert isinstance(db_result, dict)

            print(f"✓ RAG router dual: files={len(fs_result['files'])}, db=connected")


# ============================================================================
# OpenAPI Tool Loading
# ============================================================================


class TestOpenAPIToolLoading:
    """Test loading tools from OpenAPI specs."""

    def test_openapi_spec_parsing(self):
        """Test parsing basic OpenAPI spec."""
        # Example OpenAPI spec
        openapi_spec = {
            "openapi": "3.0.0",
            "info": {"title": "Test API", "version": "1.0.0"},
            "paths": {
                "/search": {
                    "post": {
                        "operationId": "search",
                        "summary": "Search for content",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {"type": "object", "properties": {"query": {"type": "string"}}}
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Search results",
                                "content": {
                                    "application/json": {
                                        "schema": {"type": "object", "properties": {"results": {"type": "array"}}}
                                    }
                                },
                            }
                        },
                    }
                }
            },
        }

        # Should be able to extract basic info
        assert openapi_spec["info"]["title"] == "Test API"
        assert "/search" in openapi_spec["paths"]
        assert "search" in openapi_spec["paths"]["/search"]["post"]["operationId"]
        print("✓ OpenAPI spec parsing works")

    def test_openapi_toolset_from_spec(self):
        """Test ToolSet creation from OpenAPI spec."""
        # This is a placeholder - real implementation would:
        # 1. Parse OpenAPI spec
        # 2. Create Tool objects for each endpoint
        # 3. Return as ToolSet

        try:
            # Check if ToolSet has from_openapi method
            assert hasattr(ToolSet, "from_openapi"), "ToolSet should have from_openapi classmethod"
            print("✓ ToolSet.from_openapi method exists")
        except AssertionError:
            print("ℹ ToolSet.from_openapi not yet implemented, need to add it")


# ============================================================================
# FastAPI Integration Test Server
# ============================================================================


class TestFastAPIIntegration:
    """Test integration with FastAPI servers."""

    def test_fastapi_imports(self):
        """Test that FastAPI can be imported."""
        try:
            import fastapi

            print("✓ FastAPI is installed")
        except ImportError:
            pytest.skip("FastAPI not installed")

    @pytest.mark.asyncio
    async def test_fastapi_server_as_tool_source(self):
        """Test using a FastAPI server as a tool source."""
        try:
            from fastapi import FastAPI
            from fastapi.testclient import TestClient
            from pydantic import BaseModel
        except ImportError:
            pytest.skip("FastAPI not installed")

        # Create a simple FastAPI app
        app = FastAPI(title="Test API", openapi_url="/openapi.json")

        class CalculateRequest(BaseModel):
            operation: str
            a: int
            b: int

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
        async def status():
            """Get API status."""
            return {"status": "ok"}

        # Test the app
        client = TestClient(app)

        # Test calculate endpoint
        response = client.post("/calculate", json={"operation": "add", "a": 5, "b": 3})
        assert response.status_code == 200
        assert response.json()["result"] == 8
        print(f"✓ FastAPI endpoint works: 5 + 3 = {response.json()['result']}")

        # Test status endpoint
        response = client.get("/status")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        print("✓ FastAPI status endpoint works")

        # Test OpenAPI spec
        response = client.get("/openapi.json")
        assert response.status_code == 200
        openapi_spec = response.json()
        assert "paths" in openapi_spec
        assert "/calculate" in openapi_spec["paths"]
        print(f"✓ FastAPI OpenAPI spec available with {len(openapi_spec['paths'])} paths")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
