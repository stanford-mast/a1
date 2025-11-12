"""
Tests for code normalization utilities.

Tests the normalize_generated_code function which handles various
LLM output formats and normalizes them to clean function bodies.
"""

from a1.code_utils import normalize_generated_code


class TestNormalizeGeneratedCode:
    """Test cases for normalize_generated_code function."""

    def test_just_body_ideal_case(self):
        """Test Case 1: Just body (ideal case) - should remain unchanged."""
        input_code = "result = await llm(...)\nreturn result"
        expected = "result = await llm(...)\nreturn result"
        result = normalize_generated_code(input_code)
        assert result == expected

    def test_body_with_extra_indentation(self):
        """Test Case 2: Body with extra indentation - should remove common indent."""
        input_code = "    result = await llm(...)\n    return result"
        expected = "result = await llm(...)\nreturn result"
        result = normalize_generated_code(input_code)
        assert result == expected

    def test_full_function_definition(self):
        """Test Case 3: Full function definition - should extract only body."""
        input_code = """async def parser_agent(problem: str) -> ParsedProblem:
    result = await llm(..., output_schema=ParsedProblem)
    return result"""
        expected = "result = await llm(..., output_schema=ParsedProblem)\nreturn result"
        result = normalize_generated_code(input_code)
        assert result == expected

    def test_function_with_docstring(self):
        """Test Case 4: Function with docstring - should skip docstring, extract body."""
        input_code = """async def parser_agent(problem: str) -> ParsedProblem:
    \"\"\"Parse the problem.\"\"\"
    result = await llm(..., output_schema=ParsedProblem)
    return result"""
        expected = "result = await llm(..., output_schema=ParsedProblem)\nreturn result"
        result = normalize_generated_code(input_code)
        assert result == expected

    def test_with_markdown_fences(self):
        """Test Case 5: Code with markdown fences - should remove fences."""
        input_code = """```python
result = await llm(...)
return result
```"""
        expected = "result = await llm(...)\nreturn result"
        result = normalize_generated_code(input_code)
        assert result == expected

    def test_with_comments(self):
        """Test Case 6: Code with comments - should remove comments."""
        input_code = """# Build the prompt
result = await llm(...)
# Return the result
return result"""
        expected = "result = await llm(...)\nreturn result"
        result = normalize_generated_code(input_code)
        assert result == expected

    def test_empty_code(self):
        """Test empty/None code - should handle gracefully."""
        assert normalize_generated_code("") == ""
        assert normalize_generated_code(None) == None

    def test_complex_multiline_body(self):
        """Test more complex multiline function body."""
        input_code = """async def complex_func(x: str) -> Output:
    \"\"\"Complex function.\"\"\"
    # Parse input
    parsed = json.loads(x)
    # Call LLM
    result = await llm(parsed)
    # Return output
    return Output(result=result)"""

        # Should extract body, remove comments
        result = normalize_generated_code(input_code)
        assert "parsed = json.loads(x)" in result
        assert "result = await llm(parsed)" in result
        assert "return Output(result=result)" in result
        assert "# Parse input" not in result
        assert "# Call LLM" not in result
        assert "# Return output" not in result

    def test_markdown_fences_with_py(self):
        """Test markdown fences with ```py instead of ```python."""
        input_code = """```py
result = await llm(...)
return result
```"""
        expected = "result = await llm(...)\nreturn result"
        result = normalize_generated_code(input_code)
        assert result == expected

    def test_markdown_fences_plain(self):
        """Test markdown fences with just ``` (no language)."""
        input_code = """```
result = await llm(...)
return result
```"""
        expected = "result = await llm(...)\nreturn result"
        result = normalize_generated_code(input_code)
        assert result == expected

    def test_mixed_indentation_and_fences(self):
        """Test combination of markdown fences and extra indentation."""
        input_code = """```python
    result = await llm(...)
    return result
```"""
        expected = "result = await llm(...)\nreturn result"
        result = normalize_generated_code(input_code)
        assert result == expected

    def test_preserves_internal_indentation(self):
        """Test that internal indentation (like in if statements) is preserved."""
        input_code = """    if condition:
        result = await llm(...)
    else:
        result = default
    return result"""

        # Should remove common 4-space indent but preserve internal structure
        result = normalize_generated_code(input_code)
        lines = result.split("\n")
        assert lines[0] == "if condition:"
        assert lines[1] == "    result = await llm(...)"  # Still indented relative to if
        assert lines[2] == "else:"
        assert lines[3] == "    result = default"
        assert lines[4] == "return result"
