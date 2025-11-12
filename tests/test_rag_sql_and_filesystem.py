import asyncio

from a1.rag import RAG, Database, FileSystem


def test_rag_database_with_connection_string(tmp_path):
    # Create a sqlite file and populate
    db_file = tmp_path / "test.db"
    db_url = f"sqlite:///{db_file}"

    from sqlalchemy import create_engine, text

    engine = create_engine(db_url)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE data (id INTEGER PRIMARY KEY, name TEXT, value INTEGER);"))
        conn.execute(text("INSERT INTO data (name, value) VALUES ('a', 1), ('b', 2), ('c', 3);"))

    # Initialize RAG with Database (readonly access)
    db = Database(connection=db_url)
    rag = RAG(database=db)
    toolset = rag.get_toolset()

    # Execute a simple select
    async def run_query():
        tool = [t for t in toolset.tools if t.name == "sql"][0]
        result = await tool.execute(query="SELECT * FROM data", limit=10)
        assert "rows" in result
        assert result["row_count"] == 3
        assert len(result["rows"]) == 3
        assert result["columns"] == ["id", "name", "value"]

        # Non-SELECT should be rejected
        bad = await tool.execute(query="DROP TABLE data", limit=10)
        assert bad.get("error") is not None

    asyncio.run(run_query())


def test_rag_filesystem_limits(tmp_path):
    # Create many files
    base = tmp_path / "docs"
    base.mkdir()
    for i in range(50):
        (base / f"file_{i}.txt").write_text("line1\nline2\nmatch me\n")

    fs = FileSystem(str(base))
    rag = RAG(filesystem=fs)
    toolset = rag.get_toolset()

    async def run_fs_tests():
        ls_tool = [t for t in toolset.tools if t.name == "ls"][0]
        grep_tool = [t for t in toolset.tools if t.name == "grep"][0]
        cat_tool = [t for t in toolset.tools if t.name == "cat"][0]

        # Default ls should return up to default limit (1000) and truncated False
        res = await ls_tool.execute(path="", limit=1000)
        assert "files" in res
        assert res["truncated"] is False
        assert len(res["files"]) == 50

        # ls with small limit should be truncated
        res2 = await ls_tool.execute(path="", limit=10)
        assert res2["truncated"] is True
        assert len(res2["files"]) == 10

        # grep default limit should enforce
        g = await grep_tool.execute(pattern="match", path="", limit=5)
        assert "matches" in g
        assert len(g["matches"]) <= 5

        # cat should truncate large file content
        big_file = base / "big.txt"
        big_file.write_text("A" * 20000)
        c = await cat_tool.execute(path="big.txt")
        assert "content" in c
        assert c["truncated"] is True
        assert len(c["content"]) == 10000

    asyncio.run(run_fs_tests())
