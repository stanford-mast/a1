"""Tests for Context persistence with from_file() and keep_updated."""

import json
import tempfile
from pathlib import Path

from a1 import Context
from a1.models import Message


def test_context_from_file_creates_new():
    """Test from_file creates new context if file doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        ctx = Context.from_file(str(path))

        assert len(ctx) == 0
        assert ctx.file_path == path
        assert ctx.keep_updated is False


def test_context_from_file_loads_existing():
    """Test from_file loads existing messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        # Create file with messages
        messages = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
        with open(path, "w") as f:
            json.dump(messages, f)

        # Load context
        ctx = Context.from_file(str(path))

        assert len(ctx) == 2
        assert ctx.messages[0].role == "user"
        assert ctx.messages[0].content == "Hello"
        assert ctx.messages[1].role == "assistant"
        assert ctx.messages[1].content == "Hi there"


def test_context_keep_updated_saves_on_append():
    """Test keep_updated auto-saves when messages are appended."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        # Create context with auto-update
        ctx = Context.from_file(str(path), keep_updated=True)
        ctx.user("Hello")

        # Verify file was created and contains message
        assert path.exists()
        with open(path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["role"] == "user"
        assert data[0]["content"] == "Hello"


def test_context_keep_updated_saves_multiple_messages():
    """Test keep_updated saves multiple messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        ctx = Context.from_file(str(path), keep_updated=True)
        ctx.user("Hello")
        ctx.assistant("Hi there")
        ctx.user("How are you?")

        # Load file and verify all messages saved
        with open(path) as f:
            data = json.load(f)

        assert len(data) == 3
        assert data[0]["content"] == "Hello"
        assert data[1]["content"] == "Hi there"
        assert data[2]["content"] == "How are you?"


def test_context_keep_updated_saves_on_extend():
    """Test keep_updated saves when messages are extended."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        ctx = Context.from_file(str(path), keep_updated=True)
        messages = [Message(role="user", content="Hello"), Message(role="assistant", content="Hi")]
        ctx.extend(messages)

        # Verify saved
        with open(path) as f:
            data = json.load(f)

        assert len(data) == 2


def test_context_keep_updated_saves_on_clear():
    """Test keep_updated saves when context is cleared."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        ctx = Context.from_file(str(path), keep_updated=True)
        ctx.user("Hello")
        ctx.assistant("Hi")

        # Clear and verify
        ctx.clear()

        with open(path) as f:
            data = json.load(f)

        assert len(data) == 0


def test_context_without_keep_updated_does_not_save():
    """Test context without keep_updated doesn't auto-save."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        ctx = Context.from_file(str(path), keep_updated=False)
        ctx.user("Hello")

        # File should be empty since keep_updated is False
        # (file created but empty on init)
        assert not path.exists()


def test_context_manual_save():
    """Test manual save works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        ctx = Context(file_path=path)
        ctx.user("Hello")
        ctx._save()

        # Verify saved
        with open(path) as f:
            data = json.load(f)

        assert len(data) == 1
        assert data[0]["content"] == "Hello"


def test_context_persistence_with_tool_calls():
    """Test persistence works with tool calls."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        ctx = Context.from_file(str(path), keep_updated=True)
        ctx.assistant("", tool_calls=[{"id": "1", "name": "calc", "arguments": "{}"}])
        ctx.tool("42", name="calc", tool_call_id="1")

        # Load and verify
        with open(path) as f:
            data = json.load(f)

        assert len(data) == 2
        assert data[0]["tool_calls"] is not None
        assert data[1]["name"] == "calc"


def test_context_reload_from_updated_file():
    """Test loading context that was updated by another process."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        # First context writes
        ctx1 = Context.from_file(str(path), keep_updated=True)
        ctx1.user("Hello")

        # Second context loads
        ctx2 = Context.from_file(str(path))

        assert len(ctx2) == 1
        assert ctx2.messages[0].content == "Hello"


def test_context_creates_parent_directories():
    """Test from_file creates parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "subdir" / "nested" / "test.json"

        ctx = Context.from_file(str(path), keep_updated=True)
        ctx.user("Hello")

        assert path.exists()
        assert path.parent.exists()
