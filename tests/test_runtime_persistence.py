"""Tests for Runtime persistence with from_file() and keep_updated."""

import json
import tempfile
from pathlib import Path

from a1 import Runtime, get_context


def test_runtime_from_file_creates_new():
    """Test from_file creates new runtime if file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "runtime.json"

        with Runtime.from_file(str(path)) as runtime:
            assert runtime.file_path == path
            assert runtime.keep_updated is False
            assert len(runtime.CTX) == 0


def test_runtime_from_file_loads_existing():
    """Test from_file loads existing runtime state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "runtime.json"

        # Create file with runtime state
        data = {
            "cache_dir": ".a1",
            "contexts": {"main": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}]},
        }
        with open(path, "w") as f:
            json.dump(data, f)

        # Load runtime
        with Runtime.from_file(str(path)) as runtime:
            assert "main" in runtime.CTX
            ctx = runtime.CTX["main"]
            assert len(ctx) == 2
            assert ctx.messages[0].content == "Hello"
            assert ctx.messages[1].content == "Hi"


def test_runtime_keep_updated_saves_on_context_change():
    """Test keep_updated auto-saves when context changes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "runtime.json"

        with Runtime.from_file(str(path), keep_updated=True) as runtime:
            ctx = get_context("main")
            ctx.user("Hello")

        # Verify file was saved
        assert path.exists()
        with open(path) as f:
            data = json.load(f)

        assert "main" in data["contexts"]
        assert len(data["contexts"]["main"]) == 1
        assert data["contexts"]["main"][0]["content"] == "Hello"


def test_runtime_reload_session():
    """Test saving and reloading a session."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "session.json"

        # Create session
        with Runtime.from_file(str(path), keep_updated=True) as runtime:
            ctx = get_context("main")
            ctx.user("Hello")
            ctx.assistant("Hi there")
            ctx.user("How are you?")

        # Reload session
        with Runtime.from_file(str(path)) as runtime:
            ctx = get_context("main")
            assert len(ctx.messages) == 3
            assert ctx.messages[0].content == "Hello"
            assert ctx.messages[2].content == "How are you?"


def test_runtime_multiple_contexts():
    """Test persistence with multiple contexts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "runtime.json"

        with Runtime.from_file(str(path), keep_updated=True) as runtime:
            ctx1 = get_context("main")
            ctx1.user("Main message")

            ctx2 = get_context("secondary")
            ctx2.user("Secondary message")

        # Reload and verify both contexts
        with Runtime.from_file(str(path)) as runtime:
            assert "main" in runtime.CTX
            assert "secondary" in runtime.CTX
            assert runtime.CTX["main"].messages[0].content == "Main message"
            assert runtime.CTX["secondary"].messages[0].content == "Secondary message"


def test_runtime_preserves_cache_dir():
    """Test cache_dir is preserved across saves."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "runtime.json"
        cache = tmpdir + "/.custom_cache"

        # Create with custom cache_dir
        with Runtime.from_file(str(path), keep_updated=True, cache_dir=cache) as runtime:
            ctx = get_context("main")
            ctx.user("Test")

        # Reload and verify cache_dir
        with Runtime.from_file(str(path)) as runtime:
            assert str(runtime.cache_dir) == cache


def test_runtime_without_keep_updated():
    """Test runtime without keep_updated doesn't auto-save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "runtime.json"

        with Runtime.from_file(str(path), keep_updated=False) as runtime:
            ctx = get_context("main")
            ctx.user("Hello")

        # File shouldn't exist since keep_updated is False
        assert not path.exists()


def test_runtime_manual_save():
    """Test manual save works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "runtime.json"

        with Runtime.from_file(str(path), keep_updated=False) as runtime:
            ctx = get_context("main")
            ctx.user("Hello")

            # Manual save
            runtime._save()

        # Verify saved
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert len(data["contexts"]["main"]) == 1


def test_runtime_creates_parent_directories():
    """Test from_file creates parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "nested" / "runtime.json"

        with Runtime.from_file(str(path), keep_updated=True) as runtime:
            ctx = get_context("main")
            ctx.user("Hello")

        assert path.exists()
        assert path.parent.exists()


def test_runtime_persistence_with_tool_calls():
    """Test persistence works with tool calls in context."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "runtime.json"

        with Runtime.from_file(str(path), keep_updated=True) as runtime:
            ctx = get_context("main")
            ctx.assistant("", tool_calls=[{"id": "1", "name": "calc", "arguments": "{}"}])
            ctx.tool("42", name="calc", tool_call_id="1")

        # Reload and verify
        with Runtime.from_file(str(path)) as runtime:
            ctx = get_context("main")
            assert len(ctx.messages) == 2
            assert ctx.messages[0].tool_calls is not None
            assert ctx.messages[1].name == "calc"
