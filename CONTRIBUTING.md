# Contributing to A1

Thank you for your interest in contributing to A1! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/A1
   cd A1
   ```

2. **Install uv (if not already installed):**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Create a virtual environment and install dependencies:**
   ```bash
   uv sync --dev
   ```

4. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On Linux/macOS
   # or
   .venv\Scripts\activate  # On Windows
   ```

5. **Install pre-commit hooks:**
   ```bash
   uv run pre-commit install
   ```
   
   This will automatically run `ruff` linting and formatting on staged files before each commit.

## Development Workflow

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=A1 --cov-report=html

# Run specific test file
uv run pytest tests/test_models.py

# Run specific test
uv run pytest tests/test_models.py::TestTool::test_tool_decorator_simple
```

### Code Quality

We use `ruff` for linting and formatting. Pre-commit hooks are configured to automatically run ruff on staged files before each commit.

**Automatic checks (via pre-commit):**
- Pre-commit hooks automatically run `ruff check --fix` and `ruff format` on staged Python files before each commit
- If checks fail, the commit will be blocked until issues are resolved

**Manual checks:**
```bash
# Format code
uv run ruff format .

# Check linting
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .
```

We use `mypy` for type checking:

```bash
uv run mypy src/A1
```

### Running Examples

```bash
uv run python examples/simple_agent.py
```

## Making Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes and write tests**

3. **Ensure all tests pass and code is formatted:**
   ```bash
   uv run ruff format .
   uv run ruff check --fix .
   uv run pytest
   ```

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

5. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public classes and functions
- Keep line length to 120 characters
- Use descriptive variable names

### Example:

```python
from typing import Optional

async def my_function(param1: str, param2: int = 10) -> Optional[str]:
    """
    Brief description of what the function does.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 10)
    
    Returns:
        Description of return value
    """
    # Implementation
    return result
```

## Testing Guidelines

- Write tests for all new features
- Maintain or improve code coverage
- Use descriptive test names: `test_<what_is_being_tested>`
- Use pytest fixtures for common setup
- Test both success and failure cases

### Example:

```python
import pytest
from A1 import tool

class TestTool:
    """Test Tool functionality."""
    
    def test_tool_creation_with_decorator(self):
        """Test creating a tool using @tool decorator."""
        @tool(name="test", description="Test tool")
        async def test_func(x: int) -> int:
            return x * 2
        
        assert test_func.name == "test"
        assert test_func.description == "Test tool"
    
    @pytest.mark.asyncio
    async def test_tool_execution(self):
        """Test tool execution with validation."""
        @tool(name="add")
        async def add(a: int, b: int) -> int:
            return a + b
        
        result = await add(a=2, b=3)
        assert result == 5
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new classes and functions
- Update examples if adding new features
- Keep documentation clear and concise

## Commit Messages

Use clear, descriptive commit messages:

```
feat: add support for custom verification strategies
fix: correct type hints in Runtime.aot method
docs: update README with MCP integration examples
test: add tests for FileSystemRAG toolset
refactor: simplify code generation logic
```

Prefixes:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `test`: Test additions or changes
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `chore`: Build process or tooling changes

## Release Process

Maintainers will handle releases:

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a git tag
4. Build and publish to PyPI

## Questions?

- Open an issue for bug reports or feature requests
- Start a discussion for questions or ideas
- Check existing issues and PRs before creating new ones

## License

By contributing to A1, you agree that your contributions will be licensed under the MIT License.
