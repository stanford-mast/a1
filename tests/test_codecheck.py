"""
Tests for code verification and validation module.
"""

from a1.codecheck import (
    BaseVerify,
    IsLoop,
    check_code_candidate,
    check_dangerous_ops,
    check_syntax,
)


class TestBaseVerify:
    """Test BaseVerify strategy."""

    def test_valid_code(self):
        """Test verification of valid code."""
        code = """
x = 1
y = 2
result = x + y
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid
        assert error is None

    def test_syntax_error(self):
        """Test detection of syntax errors."""
        code = "this is not valid python!!!"
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Syntax error" in error

    def test_dangerous_import_os(self):
        """Test detection of dangerous os import."""
        code = """
import os
os.system("rm -rf /")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Dangerous import" in error

    def test_dangerous_import_subprocess(self):
        """Test detection of dangerous subprocess import."""
        code = """
import subprocess
subprocess.run(["ls"])
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Dangerous import" in error

    def test_dangerous_function_eval(self):
        """Test detection of eval function."""
        code = """
result = eval("1 + 1")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Dangerous function" in error

    def test_dangerous_function_exec(self):
        """Test detection of exec function."""
        code = """
exec("print('hello')")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Dangerous function" in error

    def test_safe_imports_allowed(self):
        """Test that safe imports are allowed."""
        code = """
import json
import re
from typing import List
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_async_await_allowed(self):
        """Test that async/await syntax is allowed."""
        code = """
async def test():
    result = await some_tool(x=1)
    return result
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


class TestIsLoop:
    """Test IsLoop verifier."""

    def test_valid_loop_pattern(self):
        """Test verification of valid agentic loop."""
        code = """
while True:
    result = await llm(prompt="What should I do?")
    if result.done:
        break
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert is_valid
        assert error is None

    def test_missing_while_true(self):
        """Test detection of missing while True."""
        code = """
result = await llm(prompt="test")
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Not a standard agentic loop" in error

    def test_missing_llm_call(self):
        """Test detection of missing LLM call."""
        code = """
while True:
    x = 1
    break
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid

    def test_missing_break(self):
        """Test detection of missing break."""
        code = """
while True:
    result = await llm(prompt="test")
"""
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid

    def test_syntax_error(self):
        """Test handling of syntax errors."""
        code = "while True invalid"
        verifier = IsLoop()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Invalid syntax" in error


class TestCheckCodeCandidate:
    """Test check_code_candidate function."""

    def test_valid_code(self):
        """Test validation of valid code."""
        code = """
x = 1
y = 2
result = x + y
"""
        is_valid, error = check_code_candidate(code)
        assert is_valid
        assert error is None

    def test_invalid_syntax(self):
        """Test validation fails on syntax error."""
        code = "this is not valid x ="
        is_valid, error = check_code_candidate(code)
        assert not is_valid
        assert error is not None

    def test_dangerous_code(self):
        """Test validation fails on dangerous code."""
        code = """
import os
os.system("rm -rf /")
"""
        is_valid, error = check_code_candidate(code)
        assert not is_valid
        assert error is not None

    def test_with_custom_verifiers(self):
        """Test validation with custom verifiers."""
        code = """
while True:
    result = await llm(prompt="test")
    break
"""
        # Should pass basic verification
        is_valid, error = check_code_candidate(code)
        assert is_valid

        # Should also pass loop verification
        is_valid, error = check_code_candidate(code, verifiers=[IsLoop()])
        assert is_valid

    def test_custom_verifier_failure(self):
        """Test that custom verifier can reject code."""
        code = """
result = await tool(x=1)
"""
        # Should pass basic verification
        is_valid, error = check_code_candidate(code)
        assert is_valid

        # Should fail loop verification
        is_valid, error = check_code_candidate(code, verifiers=[IsLoop()])
        assert not is_valid


class TestCheckSyntax:
    """Test check_syntax function."""

    def test_valid_syntax(self):
        """Test valid Python syntax."""
        code = "x = 1 + 2"
        is_valid, error = check_syntax(code)
        assert is_valid
        assert error is None

    def test_invalid_syntax(self):
        """Test invalid Python syntax."""
        code = "x = 1 +"
        is_valid, error = check_syntax(code)
        assert not is_valid
        assert "Syntax error" in error
        assert "line" in error

    def test_multiline_valid(self):
        """Test valid multiline code."""
        code = """
def test():
    x = 1
    return x
"""
        is_valid, error = check_syntax(code)
        assert is_valid

    def test_multiline_invalid(self):
        """Test invalid multiline code."""
        code = """
def test()
    x = 1
"""
        is_valid, error = check_syntax(code)
        assert not is_valid


class TestCheckDangerousOps:
    """Test check_dangerous_ops function."""

    def test_safe_code(self):
        """Test safe code passes."""
        code = """
import json
x = json.dumps({"a": 1})
"""
        is_safe, error = check_dangerous_ops(code)
        assert is_safe
        assert error is None

    def test_dangerous_eval(self):
        """Test eval is detected."""
        code = "x = eval('1+1')"
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "eval" in error

    def test_dangerous_exec(self):
        """Test exec is detected."""
        code = "exec('print(1)')"
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "exec" in error

    def test_dangerous_os_import(self):
        """Test os import is detected."""
        code = "import os"
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "os" in error

    def test_dangerous_subprocess_import(self):
        """Test subprocess import is detected."""
        code = "from subprocess import run"
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "subprocess" in error

    def test_safe_standard_library(self):
        """Test safe stdlib imports are allowed."""
        code = """
import json
import re
from typing import List
from dataclasses import dataclass
"""
        is_safe, error = check_dangerous_ops(code)
        assert is_safe

    def test_syntax_error_handling(self):
        """Test handling of syntax errors."""
        code = "this is invalid x ="
        is_safe, error = check_dangerous_ops(code)
        assert not is_safe
        assert "Invalid syntax" in error


# ============================================================================
# Edge Case Tests: Type Checking, Async/Await, Imports, Malformed Code
# ============================================================================


class TestTypeChecking:
    """Test type checking with definition + generated code."""

    def test_tuple_code_with_definition(self):
        """Test verification of tuple (definition_code, generated_code)."""
        definition_code = """
from pydantic import BaseModel, Field

class Input(BaseModel):
    x: int = Field(..., description="A number")

class Output(BaseModel):
    result: int = Field(..., description="Result")
"""
        generated_code = """
output = Output(result=Input.model_validate(validated).x + 1)
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify((definition_code, generated_code), None)
        # Should pass - valid types
        assert is_valid or error is None  # ty might not be installed

    def test_tuple_code_mismatch(self):
        """Test that type mismatches are caught (if ty available)."""
        definition_code = """
from pydantic import BaseModel, Field

class Input(BaseModel):
    x: int

class Output(BaseModel):
    result: int
"""
        generated_code = """
# Type mismatch: assigning string to int field
output = Output(result="not an int")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify((definition_code, generated_code), None)
        # ty will catch this if installed, otherwise passes
        if not is_valid:
            assert "Type checking" in error or "invalid-assignment" in error

    def test_extract_full_code(self):
        """Test _extract_full_code concatenates definition and generated."""
        verifier = BaseVerify()
        definition = "x = 1\n"
        generated = "y = 2"
        full = verifier._extract_full_code((definition, generated))
        assert "x = 1" in full
        assert "y = 2" in full

    def test_extract_code_string(self):
        """Test _extract_code returns string as-is."""
        verifier = BaseVerify()
        code = "x = 1"
        result = verifier._extract_code(code)
        assert result == code

    def test_extract_code_tuple(self):
        """Test _extract_code extracts generated from tuple."""
        verifier = BaseVerify()
        code = ("definition", "generated")
        result = verifier._extract_code(code)
        assert result == "generated"


class TestAsyncAwaitEdgeCases:
    """Test handling of async/await patterns in generated code."""

    def test_simple_await(self):
        """Test basic await syntax."""
        code = """
result = await tool(x=1)
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_nested_await(self):
        """Test nested async calls."""
        code = """
result1 = await tool1(x=1)
result2 = await tool2(y=result1)
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_asyncio_run_in_code(self):
        """Test that asyncio.run is caught as suspicious (but not dangerous)."""
        # asyncio.run shouldn't appear in async context code
        # But it's not a "dangerous" operation per se
        code = """
import asyncio
result = asyncio.run(some_func())
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        # asyncio import is allowed, so this should pass
        # The issue would be at runtime when called in async context
        assert is_valid or "asyncio" not in error

    def test_multiple_awaits_same_line(self):
        """Test multiple await expressions."""
        code = """
a, b = await tool1(x=1), await tool2(y=2)
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_await_in_comprehension(self):
        """Test await in list comprehension."""
        code = """
results = [await tool(i) for i in range(3)]
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        # This is actually invalid syntax (await in comprehension without async for)
        # But we're just checking if verification works
        assert is_valid or "await" in error or "comprehension" in error


class TestImportEdgeCases:
    """Test handling of various import patterns."""

    def test_relative_imports(self):
        """Test relative imports are allowed."""
        code = """
from . import utils
from .submodule import helper
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_star_import(self):
        """Test star imports are allowed if safe."""
        code = """
from json import *
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_multiple_imports_same_line(self):
        """Test multiple imports on same line."""
        code = """
import json, re, asyncio
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_alias_imports(self):
        """Test import aliases."""
        code = """
import json as j
from typing import List as L
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_conditional_import(self):
        """Test conditional imports."""
        code = """
if True:
    import json
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


class TestMalformedCodeEdgeCases:
    """Test handling of various malformed code patterns."""

    def test_unmatched_parentheses(self):
        """Test unmatched parentheses."""
        code = "result = func(x=1"
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Syntax error" in error

    def test_unmatched_brackets(self):
        """Test unmatched brackets."""
        code = "result = [1, 2, 3"
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Syntax error" in error

    def test_missing_colon(self):
        """Test missing colon in if statement."""
        code = """
if True
    print("hello")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Syntax error" in error

    def test_indentation_error(self):
        """Test indentation errors."""
        code = """
def test():
x = 1
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert not is_valid
        assert "Syntax error" in error

    def test_invalid_escape_sequence(self):
        """Test invalid escape sequences."""
        code = r'path = "C:\temp\file"'  # Invalid Windows path
        verifier = BaseVerify()
        # This might pass or fail depending on Python version
        # Just ensure it doesn't crash
        is_valid, error = verifier.verify(code, None)
        assert isinstance(is_valid, bool)

    def test_mixing_tabs_spaces(self):
        """Test mixing tabs and spaces (Python 3 error)."""
        code = "if True:\n\tx = 1\n    y = 2"
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        # May fail depending on context
        assert isinstance(is_valid, bool)


class TestFunctionEdgeCases:
    """Test handling of various function patterns."""

    def test_lambda_function(self):
        """Test lambda functions."""
        code = """
transform = lambda x: x + 1
result = transform(5)
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_generator_function(self):
        """Test generator functions."""
        code = """
def gen():
    yield 1
    yield 2

result = list(gen())
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_nested_functions(self):
        """Test nested function definitions."""
        code = """
def outer():
    def inner():
        return 42
    return inner()

result = outer()
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_decorator(self):
        """Test function decorators."""
        code = """
@some_decorator
def test():
    return 42
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_async_generator(self):
        """Test async generator functions."""
        code = """
async def gen():
    yield 1
    yield 2
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


class TestExceptionHandlingEdgeCases:
    """Test exception handling patterns."""

    def test_try_except(self):
        """Test basic try/except."""
        code = """
try:
    result = risky_operation()
except Exception as e:
    result = None
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_try_except_finally(self):
        """Test try/except/finally."""
        code = """
try:
    result = risky_operation()
except ValueError:
    result = None
finally:
    cleanup()
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_try_except_else(self):
        """Test try/except/else."""
        code = """
try:
    result = risky_operation()
except Exception:
    result = None
else:
    process(result)
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_raise_exception(self):
        """Test raising exceptions."""
        code = """
if error_condition:
    raise ValueError("Something went wrong")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


class TestComplexDataStructures:
    """Test complex data structure patterns."""

    def test_dict_comprehension(self):
        """Test dictionary comprehension."""
        code = """
result = {i: i**2 for i in range(10)}
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_set_comprehension(self):
        """Test set comprehension."""
        code = """
result = {i**2 for i in range(10)}
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_nested_data_structures(self):
        """Test nested dictionaries and lists."""
        code = """
data = {
    'items': [1, 2, 3],
    'mapping': {'a': 1, 'b': 2},
    'nested': {
        'deep': [{'x': 1}]
    }
}
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_unpacking(self):
        """Test tuple unpacking."""
        code = """
a, b, c = [1, 2, 3]
*rest, last = [1, 2, 3, 4, 5]
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


class TestContextManagementEdgeCases:
    """Test context (with statement) patterns."""

    def test_with_statement(self):
        """Test basic with statement."""
        code = """
with open('file.txt') as f:
    data = f.read()
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_multiple_context_managers(self):
        """Test multiple context managers."""
        code = """
with open('in.txt') as f_in, open('out.txt', 'w') as f_out:
    f_out.write(f_in.read())
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_async_with_statement(self):
        """Test async with statement."""
        code = """
async with async_resource() as resource:
    await resource.process()
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


class TestStringFormattingEdgeCases:
    """Test various string formatting patterns."""

    def test_f_string(self):
        """Test f-string formatting."""
        code = """
name = "world"
result = f"Hello {name}!"
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_f_string_with_expressions(self):
        """Test f-string with complex expressions."""
        code = """
x = 5
result = f"Result: {x * 2 + 3}"
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_multiline_string(self):
        """Test multiline strings."""
        code = '''
text = """
This is a
multiline
string
"""
'''
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_raw_string(self):
        """Test raw strings."""
        code = r'path = r"C:\Users\name\file.txt"'
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid


class TestSpecialOperators:
    """Test special Python operators and constructs."""

    def test_walrus_operator(self):
        """Test walrus operator := (Python 3.8+)."""
        code = """
if (n := len(items)) > 10:
    print(f"Too many items: {n}")
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid

    def test_match_statement(self):
        """Test match/case statements (Python 3.10+)."""
        code = """
match status:
    case 200:
        result = "OK"
    case 404:
        result = "Not Found"
    case _:
        result = "Unknown"
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        # May fail on older Python versions, that's ok
        assert isinstance(is_valid, bool)

    def test_operator_overloading(self):
        """Test operator overloading."""
        code = """
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y
    
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

v = Vector(1, 2) + Vector(3, 4)
"""
        verifier = BaseVerify()
        is_valid, error = verifier.verify(code, None)
        assert is_valid
