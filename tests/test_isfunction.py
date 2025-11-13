"""Tests for IsFunction verifier."""

from a1.extra_codecheck import IsFunction


class TestIsFunction:
    """Test IsFunction verifier."""

    def test_valid_async_function(self):
        """Test valid async function passes."""
        verifier = IsFunction()
        code = """
async def my_task(x: int) -> str:
    result = await tool_a(x)
    return f"Result: {result}"
"""
        is_valid, error = verifier.verify(code)
        assert is_valid
        assert error is None

    def test_missing_function(self):
        """Test code without function fails."""
        verifier = IsFunction()
        code = """
x = await tool_a(42)
result = f"Result: {x}"
"""
        is_valid, error = verifier.verify(code)
        assert not is_valid
        assert "No async function implementation found" in error

    def test_multiple_functions(self):
        """Test multiple functions fail."""
        verifier = IsFunction()
        code = """
async def func1(x: int) -> str:
    return str(x)

async def func2(y: int) -> str:
    return str(y)
"""
        is_valid, error = verifier.verify(code)
        assert not is_valid
        assert "Multiple function definitions" in error

    def test_sync_function_fails(self):
        """Test non-async function fails."""
        verifier = IsFunction()
        code = """
def my_task(x: int) -> str:
    return str(x)
"""
        is_valid, error = verifier.verify(code)
        assert not is_valid
        # Should fail because it's not async

    def test_syntax_error(self):
        """Test syntax error is caught."""
        verifier = IsFunction()
        code = "async def my_task(x: int) -> str"  # Missing colon
        is_valid, error = verifier.verify(code)
        assert not is_valid
        assert "Syntax error" in error
