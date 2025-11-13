"""
Test FileSystem, Database, and RAG classes.

Tests:
1. FileSystem class with write_file, delete_file tools
2. Database class with insert, update, delete tools
3. RAG wrapper for readonly access to FileSystem and Database
"""

import tempfile
from pathlib import Path

import pytest

from a1 import RAG, Database, FileSystem

# Check if duckdb-engine is available
try:
    import duckdb_engine
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False


class TestFileSystemReadWrite:
    """Test FileSystem class with read and write operations."""

    @pytest.mark.asyncio
    async def test_filesystem_write_file(self):
        """Test writing to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(tmpdir)
            toolset = fs.get_toolset()

            # Find write_file tool
            write_tool = [t for t in toolset.tools if t.name == "write_file"][0]

            # Write file
            result = await write_tool.execute(path="test.txt", content="Hello, World!", mode="w")

            assert result["success"] is True
            assert result["bytes_written"] == len("Hello, World!")

            # Verify file exists
            assert (Path(tmpdir) / "test.txt").exists()
            assert (Path(tmpdir) / "test.txt").read_text() == "Hello, World!"
            print(f"✓ Write file: {result}")

    @pytest.mark.asyncio
    async def test_filesystem_append_file(self):
        """Test appending to a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(tmpdir)
            toolset = fs.get_toolset()
            write_tool = [t for t in toolset.tools if t.name == "write_file"][0]

            # Write initial content
            await write_tool.execute(path="log.txt", content="Line 1\n", mode="w")

            # Append content
            result = await write_tool.execute(path="log.txt", content="Line 2\n", mode="a")

            assert result["success"] is True

            # Verify appended content
            content = (Path(tmpdir) / "log.txt").read_text()
            assert "Line 1" in content
            assert "Line 2" in content
            print(f"✓ Append file: {content.splitlines()}")

    @pytest.mark.asyncio
    async def test_filesystem_delete_file(self):
        """Test deleting a file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file first
            test_file = Path(tmpdir) / "delete_me.txt"
            test_file.write_text("Content")

            fs = FileSystem(tmpdir)
            toolset = fs.get_toolset()
            delete_tool = [t for t in toolset.tools if t.name == "delete_file"][0]

            # Delete file
            result = await delete_tool.execute(path="delete_me.txt")

            assert result["success"] is True
            assert not test_file.exists()
            print(f"✓ Delete file: {result}")

    @pytest.mark.asyncio
    async def test_filesystem_all_tools(self):
        """Test that FileSystem provides all expected tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fs = FileSystem(tmpdir)
            toolset = fs.get_toolset()

            tool_names = {t.name for t in toolset.tools}
            expected = {"ls", "grep", "cat", "write_file", "delete_file"}

            assert tool_names == expected
            print(f"✓ FileSystem tools: {tool_names}")


class TestDatabaseReadWrite:
    """Test Database class with DuckDB connection strings."""

    @pytest.mark.asyncio
    @pytest.mark.skipif(not HAS_DUCKDB, reason="duckdb-engine not installed")
    async def test_database_with_duckdb_file(self):
        """Test Database with DuckDB file-based connection string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            # DuckDB connection string format
            connection = f"duckdb:///{db_path}"

            db = Database(connection)
            toolset = db.get_toolset()

            # Create table and insert data
            sql_tool = [t for t in toolset.tools if t.name == "sql"][0]
            create_result = await sql_tool.execute(query="CREATE TABLE users (id INTEGER, name VARCHAR)")

            # Verify table creation succeeded (no error)
            assert "error" not in create_result or create_result["row_count"] == 0
            print(f"✓ DuckDB table created: {connection}")

    @pytest.mark.asyncio
    async def test_database_insert(self):
        """Test inserting rows into database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            connection = f"duckdb:///{db_path}"

            db = Database(connection)
            toolset = db.get_toolset()

            sql_tool = [t for t in toolset.tools if t.name == "sql"][0]
            insert_tool = [t for t in toolset.tools if t.name == "insert"][0]

            # Create table
            await sql_tool.execute(query="CREATE TABLE products (id INTEGER, name VARCHAR, price FLOAT)")

            # Insert rows
            result = await insert_tool.execute(
                table="products",
                data=[
                    {"id": 1, "name": "Widget", "price": 9.99},
                    {"id": 2, "name": "Gadget", "price": 19.99},
                ],
            )

            assert result["success"] is True
            assert result["rows_affected"] == 2
            print(f"✓ Insert rows: {result['rows_affected']} rows")

            # Verify rows were inserted
            query_result = await sql_tool.execute(query="SELECT * FROM products")
            assert query_result["row_count"] == 2
            print(f"✓ Query result: {query_result['row_count']} rows")

    @pytest.mark.asyncio
    async def test_database_update(self):
        """Test updating rows in database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            connection = f"duckdb:///{db_path}"

            db = Database(connection)
            toolset = db.get_toolset()

            sql_tool = [t for t in toolset.tools if t.name == "sql"][0]
            insert_tool = [t for t in toolset.tools if t.name == "insert"][0]
            update_tool = [t for t in toolset.tools if t.name == "update"][0]

            # Setup
            await sql_tool.execute(query="CREATE TABLE inventory (id INTEGER, stock INTEGER)")
            await insert_tool.execute(table="inventory", data=[{"id": 1, "stock": 100}])

            # Update
            result = await update_tool.execute(table="inventory", where="id = 1", updates={"stock": 50})

            assert result["success"] is True
            print(f"✓ Update rows: {result}")

            # Verify update
            query_result = await sql_tool.execute(query="SELECT stock FROM inventory WHERE id = 1")
            assert query_result["rows"][0]["stock"] == 50

    @pytest.mark.asyncio
    async def test_database_delete(self):
        """Test deleting rows from database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            connection = f"duckdb:///{db_path}"

            db = Database(connection)
            toolset = db.get_toolset()

            sql_tool = [t for t in toolset.tools if t.name == "sql"][0]
            insert_tool = [t for t in toolset.tools if t.name == "insert"][0]
            delete_tool = [t for t in toolset.tools if t.name == "delete"][0]

            # Setup
            await sql_tool.execute(query="CREATE TABLE temp_data (id INTEGER, value VARCHAR)")
            await insert_tool.execute(
                table="temp_data",
                data=[
                    {"id": 1, "value": "keep"},
                    {"id": 2, "value": "delete"},
                ],
            )

            # Delete
            result = await delete_tool.execute(table="temp_data", where="id = 2")

            assert result["success"] is True
            print(f"✓ Delete rows: {result}")

            # Verify delete
            query_result = await sql_tool.execute(query="SELECT * FROM temp_data")
            assert query_result["row_count"] == 1
            assert query_result["rows"][0]["value"] == "keep"

    @pytest.mark.asyncio
    async def test_database_all_tools(self):
        """Test that Database provides all expected tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            connection = f"duckdb:///{db_path}"

            db = Database(connection)
            toolset = db.get_toolset()

            tool_names = {t.name for t in toolset.tools}
            expected = {"sql", "insert", "update", "delete"}

            assert tool_names == expected
            print(f"✓ Database tools: {tool_names}")


class TestFileSystemRAGReadonly:
    """Test RAG readonly wrapper for FileSystem."""

    @pytest.mark.asyncio
    async def test_rag_filesystem_readonly_tools_only(self):
        """Test that RAG with FileSystem only has readonly tools."""
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.txt").write_text("test content")

            fs = FileSystem(tmpdir)
            rag = RAG(filesystem=fs)
            toolset = rag.get_toolset()

            tool_names = {t.name for t in toolset.tools}
            expected = {"ls", "grep", "cat"}

            assert tool_names == expected
            assert "write_file" not in tool_names
            assert "delete_file" not in tool_names
            print(f"✓ RAG FileSystem tools: {tool_names}")

    @pytest.mark.asyncio
    async def test_rag_filesystem_cat_works(self):
        """Test that RAG with FileSystem cat tool works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = "Hello from RAG"
            Path(tmpdir, "readme.txt").write_text(content)

            fs = FileSystem(tmpdir)
            rag = RAG(filesystem=fs)
            toolset = rag.get_toolset()
            cat_tool = [t for t in toolset.tools if t.name == "cat"][0]

            result = await cat_tool.execute(path="readme.txt")

            assert result["content"] == content
            print(f"✓ RAG FileSystem cat: {result['content']}")


class TestRAGDatabaseReadonly:
    """Test RAG readonly wrapper for Database with DuckDB."""

    @pytest.mark.asyncio
    async def test_rag_database_select_only(self):
        """Test that RAG with Database only accepts SELECT queries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            connection = f"duckdb:///{db_path}"

            # Setup database with data
            db = Database(connection)
            toolset = db.get_toolset()
            sql_tool = [t for t in toolset.tools if t.name == "sql"][0]
            insert_tool = [t for t in toolset.tools if t.name == "insert"][0]

            await sql_tool.execute(query="CREATE TABLE test_data (id INTEGER, value VARCHAR)")
            await insert_tool.execute(table="test_data", data=[{"id": 1, "value": "test"}])

            # Now test RAG
            rag = RAG(database=db)
            toolset = rag.get_toolset()
            rag_sql_tool = [t for t in toolset.tools if t.name == "sql"][0]

            # SELECT should work
            select_result = await rag_sql_tool.execute(query="SELECT * FROM test_data")
            assert select_result["row_count"] == 1
            print(f"✓ RAG Database SELECT works: {select_result['row_count']} rows")

            # INSERT should fail
            insert_result = await rag_sql_tool.execute(query="INSERT INTO test_data (id, value) VALUES (2, 'fail')")
            assert "error" in insert_result
            assert "Only SELECT queries are allowed" in insert_result["error"]
            print(f"✓ RAG Database INSERT blocked: {insert_result['error']}")

    @pytest.mark.asyncio
    async def test_rag_database_only_has_sql_tool(self):
        """Test that RAG with Database only has sql tool (no insert/update/delete)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.duckdb"
            connection = f"duckdb:///{db_path}"

            db = Database(connection)
            rag = RAG(database=db)
            toolset = rag.get_toolset()

            tool_names = {t.name for t in toolset.tools}
            expected = {"sql"}

            assert tool_names == expected
            assert "insert" not in tool_names
            assert "update" not in tool_names
            assert "delete" not in tool_names
            print(f"✓ RAG Database tools: {tool_names}")

    @pytest.mark.asyncio
    async def test_rag_with_duckdb_connection_string(self):
        """Test RAG with actual DuckDB connection string."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "production.duckdb"
            connection = f"duckdb:///{db_path}"

            # Setup: Create database with data
            db = Database(connection)
            db_toolset = db.get_toolset()
            sql_tool = [t for t in db_toolset.tools if t.name == "sql"][0]
            insert_tool = [t for t in db_toolset.tools if t.name == "insert"][0]

            await sql_tool.execute(query="CREATE TABLE employees (id INTEGER, name VARCHAR, salary FLOAT)")
            await insert_tool.execute(
                table="employees",
                data=[
                    {"id": 1, "name": "Alice", "salary": 100000.0},
                    {"id": 2, "name": "Bob", "salary": 85000.0},
                ],
            )

            # Now use RAG for readonly queries
            rag = RAG(database=db)
            toolset = rag.get_toolset()
            rag_sql_tool = [t for t in toolset.tools if t.name == "sql"][0]

            result = await rag_sql_tool.execute(query="SELECT * FROM employees WHERE salary > 90000")

            assert result["row_count"] == 1
            assert result["rows"][0]["name"] == "Alice"
            print(f"✓ RAG Database DuckDB query: {result}")
