"""
FileSystem and Database abstractions with read/write capabilities and RAG wrapper for readonly access.

Provides:
- FileSystem: Read/write file operations (ls, grep, cat, write_file, delete_file)
- Database: Read/write database operations (sql, insert, update, delete)
- RAG: Readonly wrapper that filters FileSystem/Database tools to readonly subsets
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .models import Tool, ToolSet

# ============================================================================
# Input/Output Schemas - FileSystem
# ============================================================================


class LSInput(BaseModel):
    """Input for ls tool."""

    path: str = Field(..., description="Directory path to list")
    limit: int = Field(1000, description="Maximum number of entries to return")


class LSOutput(BaseModel):
    """Output from ls tool."""

    files: list[str] = Field(..., description="List of files and directories")
    truncated: bool = Field(False, description="Whether the listing was truncated due to limit")


class GrepInput(BaseModel):
    """Input for grep tool."""

    pattern: str = Field(..., description="Pattern to search for")
    path: str = Field(..., description="File or directory to search in")
    limit: int = Field(100, description="Maximum number of results")


class GrepOutput(BaseModel):
    """Output from grep tool."""

    matches: list[dict[str, Any]] = Field(..., description="List of matches with file and line info")


class CatInput(BaseModel):
    """Input for cat tool."""

    path: str = Field(..., description="File path to read")
    limit: int = Field(10000, description="Maximum characters to read")


class CatOutput(BaseModel):
    """Output from cat tool."""

    content: str = Field(..., description="File content")
    truncated: bool = Field(False, description="Whether content was truncated")


class WriteFileInput(BaseModel):
    """Input for write_file tool."""

    path: str = Field(..., description="File path to write")
    content: str = Field(..., description="File content to write")
    mode: str = Field("w", description="Write mode: 'w' (overwrite) or 'a' (append)")


class WriteFileOutput(BaseModel):
    """Output from write_file tool."""

    success: bool = Field(..., description="Whether write succeeded")
    path: str = Field(..., description="Path that was written")
    bytes_written: int = Field(..., description="Number of bytes written")


class DeleteFileInput(BaseModel):
    """Input for delete_file tool."""

    path: str = Field(..., description="File path to delete")


class DeleteFileOutput(BaseModel):
    """Output from delete_file tool."""

    success: bool = Field(..., description="Whether delete succeeded")
    path: str = Field(..., description="Path that was deleted")


# ============================================================================
# Input/Output Schemas - Database
# ============================================================================


class SQLQueryInput(BaseModel):
    """Input for SQL query tool."""

    query: str = Field(..., description="SQL query to execute")
    limit: int = Field(100, description="Maximum rows to return")


class SQLQueryOutput(BaseModel):
    """Output from SQL query."""

    rows: list[dict[str, Any]] = Field(..., description="Query results as list of dicts")
    columns: list[str] = Field(..., description="Column names")
    row_count: int = Field(..., description="Number of rows returned")


class SQLInsertInput(BaseModel):
    """Input for insert tool."""

    table: str = Field(..., description="Table name")
    data: list[dict[str, Any]] = Field(..., description="List of rows to insert")


class SQLInsertOutput(BaseModel):
    """Output from insert tool."""

    success: bool = Field(..., description="Whether insert succeeded")
    rows_affected: int = Field(..., description="Number of rows inserted")


class SQLUpdateInput(BaseModel):
    """Input for update tool."""

    table: str = Field(..., description="Table name")
    where: str = Field(..., description="WHERE clause (e.g., 'id = 1')")
    updates: dict[str, Any] = Field(..., description="Columns to update")


class SQLUpdateOutput(BaseModel):
    """Output from update tool."""

    success: bool = Field(..., description="Whether update succeeded")
    rows_affected: int = Field(..., description="Number of rows updated")


class SQLDeleteInput(BaseModel):
    """Input for delete tool."""

    table: str = Field(..., description="Table name")
    where: str = Field(..., description="WHERE clause (e.g., 'id = 1')")


class SQLDeleteOutput(BaseModel):
    """Output from delete tool."""

    success: bool = Field(..., description="Whether delete succeeded")
    rows_affected: int = Field(..., description="Number of rows deleted")


# ============================================================================
# FileSystem Class (Read/Write)
# ============================================================================


class FileSystem:
    """
    FileSystem abstraction for read/write file operations.

    Uses fsspec for flexible filesystem access (local, s3, gcs, etc.)
    """

    def __init__(self, filepath: str | Path):
        """
        Initialize FileSystem.

        Args:
            filepath: Base path or fsspec URL (e.g., "s3://bucket/prefix", "/local/path")
        """
        import fsspec

        self.filepath = filepath
        self.fs, self.base_path = fsspec.core.url_to_fs(str(filepath))

    def get_toolset(self) -> ToolSet:
        """Get all read/write tools as a ToolSet."""
        return ToolSet(
            name=f"filesystem_{Path(self.filepath).name}",
            description=f"File operations on {self.filepath}",
            tools=[
                self._ls_tool(),
                self._grep_tool(),
                self._cat_tool(),
                self._write_file_tool(),
                self._delete_file_tool(),
            ],
        )

    def _ls_tool(self) -> Tool:
        """List directory contents."""

        async def execute(path: str, limit: int = 1000) -> dict[str, Any]:
            full_path = f"{self.base_path}/{path}".rstrip("/")
            try:
                files = self.fs.ls(full_path, detail=False)
                rel_files = [f.replace(self.base_path + "/", "") for f in files]
                truncated = False
                if len(rel_files) > limit:
                    rel_files = rel_files[:limit]
                    truncated = True
                return {"files": rel_files, "truncated": truncated}
            except Exception as e:
                return {"files": [], "truncated": False, "error": str(e)}

        return Tool(
            name="ls",
            description="List files and directories",
            input_schema=LSInput,
            output_schema=LSOutput,
            execute=execute,
        )

    def _grep_tool(self) -> Tool:
        """Search for pattern in files."""

        async def execute(pattern: str, path: str, limit: int = 100) -> dict[str, Any]:
            import re

            full_path = f"{self.base_path}/{path}".rstrip("/")
            matches = []

            try:
                if self.fs.isfile(full_path):
                    files = [full_path]
                else:
                    files = self.fs.find(full_path)

                pattern_re = re.compile(pattern, re.IGNORECASE)

                for file in files:
                    if len(matches) >= limit:
                        break

                    try:
                        content = self.fs.cat_file(file).decode("utf-8", errors="ignore")
                        for i, line in enumerate(content.split("\n"), 1):
                            if pattern_re.search(line):
                                rel_file = file.replace(self.base_path + "/", "")
                                matches.append({"file": rel_file, "line": i, "content": line.strip()})
                                if len(matches) >= limit:
                                    break
                    except Exception:
                        continue

                return {"matches": matches}
            except Exception as e:
                return {"matches": [], "error": str(e)}

        return Tool(
            name="grep",
            description="Search for pattern in files",
            input_schema=GrepInput,
            output_schema=GrepOutput,
            execute=execute,
        )

    def _cat_tool(self) -> Tool:
        """Read file contents."""

        async def execute(path: str, limit: int = 10000) -> dict[str, Any]:
            full_path = f"{self.base_path}/{path}".rstrip("/")

            try:
                content = self.fs.cat_file(full_path).decode("utf-8", errors="ignore")
                truncated = len(content) > limit
                if truncated:
                    content = content[:limit]

                return {"content": content, "truncated": truncated}
            except Exception as e:
                return {"content": "", "truncated": False, "error": str(e)}

        return Tool(
            name="cat",
            description="Read file contents",
            input_schema=CatInput,
            output_schema=CatOutput,
            execute=execute,
        )

    def _write_file_tool(self) -> Tool:
        """Write to file."""

        async def execute(path: str, content: str, mode: str = "w") -> dict[str, Any]:
            full_path = f"{self.base_path}/{path}".rstrip("/")

            try:
                if mode not in ("w", "a"):
                    return {"success": False, "path": path, "bytes_written": 0, "error": "Invalid mode"}

                if mode == "w":
                    self.fs.pipe_file(full_path, content.encode("utf-8"))
                else:  # append
                    existing = ""
                    try:
                        existing = self.fs.cat_file(full_path).decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                    self.fs.pipe_file(full_path, (existing + content).encode("utf-8"))

                return {"success": True, "path": path, "bytes_written": len(content.encode("utf-8"))}
            except Exception as e:
                return {"success": False, "path": path, "bytes_written": 0, "error": str(e)}

        return Tool(
            name="write_file",
            description="Write content to file (can overwrite or append)",
            input_schema=WriteFileInput,
            output_schema=WriteFileOutput,
            execute=execute,
        )

    def _delete_file_tool(self) -> Tool:
        """Delete file."""

        async def execute(path: str) -> dict[str, Any]:
            full_path = f"{self.base_path}/{path}".rstrip("/")

            try:
                self.fs.rm_file(full_path)
                return {"success": True, "path": path}
            except Exception as e:
                return {"success": False, "path": path, "error": str(e)}

        return Tool(
            name="delete_file",
            description="Delete a file",
            input_schema=DeleteFileInput,
            output_schema=DeleteFileOutput,
            execute=execute,
        )


# ============================================================================
# Database Class (Read/Write)
# ============================================================================


class Database:
    """
    Database abstraction for read/write SQL operations.

    Supports SQLAlchemy connection strings (PostgreSQL, MySQL, SQLite, DuckDB, etc.)
    """

    def __init__(self, connection: str, schema: str | None = None):
        """
        Initialize Database.

        Args:
            connection: SQLAlchemy connection string (e.g., "duckdb:///file.db", "sqlite:///db.db")
            schema: Optional default schema
        """
        from sqlalchemy import create_engine

        self.connection_str = connection
        self.schema = schema
        self.engine = create_engine(connection)

    def get_toolset(self) -> ToolSet:
        """Get all read/write tools as a ToolSet."""
        return ToolSet(
            name="database",
            description="Database operations (read/write)",
            tools=[
                self._sql_tool(),
                self._insert_tool(),
                self._update_tool(),
                self._delete_tool(),
            ],
        )

    def _sql_tool(self) -> Tool:
        """Execute SQL query."""

        async def execute(query: str, limit: int = 100) -> dict[str, Any]:
            from sqlalchemy import text

            try:
                query_upper = query.strip().upper()
                is_select = query_upper.startswith("SELECT")

                if is_select:
                    # For SELECT, use regular connect
                    with self.engine.connect() as conn:
                        result = conn.execute(text(query))
                        rows = [dict(row._mapping) for row in result]
                        columns = list(result.keys()) if result.returns_rows else []

                        return {"rows": rows, "columns": columns, "row_count": len(rows)}
                else:
                    # For DDL/DML, use begin() for proper transaction
                    with self.engine.begin() as conn:
                        conn.execute(text(query))

                    return {"rows": [], "columns": [], "row_count": 0}
            except Exception as e:
                return {"rows": [], "columns": [], "row_count": 0, "error": str(e)}

        return Tool(
            name="sql",
            description="Execute SQL query (any type)",
            input_schema=SQLQueryInput,
            output_schema=SQLQueryOutput,
            execute=execute,
        )

    def _insert_tool(self) -> Tool:
        """Insert rows into table."""

        async def execute(table: str, data: list[dict[str, Any]]) -> dict[str, Any]:
            from sqlalchemy import text

            try:
                if not data:
                    return {"success": False, "rows_affected": 0, "error": "No data to insert"}

                with self.engine.begin() as conn:
                    # Use raw SQL with named parameters for each row
                    for i, row_data in enumerate(data):
                        cols = list(row_data.keys())
                        placeholders = ",".join([f":{col}" for col in cols])
                        col_names = ",".join(cols)
                        sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"
                        conn.execute(text(sql), row_data)

                return {"success": True, "rows_affected": len(data)}
            except Exception as e:
                return {"success": False, "rows_affected": 0, "error": str(e)}
            except Exception as e:
                return {"success": False, "rows_affected": 0, "error": str(e)}

        return Tool(
            name="insert",
            description="Insert rows into table",
            input_schema=SQLInsertInput,
            output_schema=SQLInsertOutput,
            execute=execute,
        )

    def _update_tool(self) -> Tool:
        """Update rows in table."""

        async def execute(table: str, where: str, updates: dict[str, Any]) -> dict[str, Any]:
            from sqlalchemy import text

            try:
                if not updates:
                    return {"success": False, "rows_affected": 0, "error": "No updates provided"}

                # Build UPDATE query
                set_clause = ", ".join([f"{col} = :{col}" for col in updates.keys()])
                update_sql = f"UPDATE {table} SET {set_clause} WHERE {where}"

                with self.engine.connect() as conn:
                    result = conn.execute(text(update_sql), updates)
                    conn.commit()
                    rows_affected = result.rowcount if hasattr(result, "rowcount") else 0

                return {"success": True, "rows_affected": rows_affected}
            except Exception as e:
                return {"success": False, "rows_affected": 0, "error": str(e)}

        return Tool(
            name="update",
            description="Update rows in table",
            input_schema=SQLUpdateInput,
            output_schema=SQLUpdateOutput,
            execute=execute,
        )

    def _delete_tool(self) -> Tool:
        """Delete rows from table."""

        async def execute(table: str, where: str) -> dict[str, Any]:
            from sqlalchemy import text

            try:
                delete_sql = f"DELETE FROM {table} WHERE {where}"

                with self.engine.connect() as conn:
                    result = conn.execute(text(delete_sql))
                    conn.commit()
                    rows_affected = result.rowcount if hasattr(result, "rowcount") else 0

                return {"success": True, "rows_affected": rows_affected}
            except Exception as e:
                return {"success": False, "rows_affected": 0, "error": str(e)}

        return Tool(
            name="delete",
            description="Delete rows from table",
            input_schema=SQLDeleteInput,
            output_schema=SQLDeleteOutput,
            execute=execute,
        )


# ============================================================================
# RAG - Readonly wrapper
# ============================================================================


class RAG:
    """
    RAG (Retrieval-Augmented Generation) wrapper providing readonly access to FileSystem or Database.

    Filters tools to readonly subsets:
    - FileSystem: ls, grep, cat (no write/delete)
    - Database: sql SELECT only (no insert/update/delete)
    """

    def __init__(self, filesystem: FileSystem | None = None, database: Database | None = None):
        """
        Initialize RAG with optional FileSystem and Database backends.

        Args:
            filesystem: FileSystem instance
            database: Database instance
        """
        self.filesystem = filesystem
        self.database = database
        self.tools = []

        if filesystem:
            fs_toolset = filesystem.get_toolset()
            readonly_tools = [t for t in fs_toolset.tools if t.name in ("ls", "grep", "cat")]
            self.tools.extend(readonly_tools)

        if database:
            # For database, wrap the sql tool to enforce SELECT-only
            db_toolset = database.get_toolset()
            sql_tool = [t for t in db_toolset.tools if t.name == "sql"][0]
            self.tools.append(self._wrap_sql_readonly(sql_tool))

    @staticmethod
    def _wrap_sql_readonly(sql_tool: Tool) -> Tool:
        """Wrap sql tool to enforce SELECT-only queries."""
        original_execute = sql_tool.execute

        async def execute_readonly(query: str, limit: int = 100) -> dict[str, Any]:
            # Security: Only allow SELECT queries
            query_upper = query.strip().upper()
            if not query_upper.startswith("SELECT"):
                return {"rows": [], "columns": [], "row_count": 0, "error": "Only SELECT queries are allowed"}

            # Add LIMIT if not present
            if "LIMIT" not in query_upper:
                query = f"{query.rstrip(';')} LIMIT {limit}"

            return await original_execute(query, limit)

        return Tool(
            name="sql",
            description="Execute SELECT query (readonly)",
            input_schema=sql_tool.input_schema,
            output_schema=sql_tool.output_schema,
            execute=execute_readonly,
        )

    def get_toolset(self, name: str = "rag") -> ToolSet:
        """Get tools as a ToolSet for use in agents."""
        return ToolSet(
            name=name, description="Readonly RAG toolset (file search and database queries)", tools=self.tools
        )


__all__ = [
    "FileSystem",
    "Database",
    "RAG",
]
