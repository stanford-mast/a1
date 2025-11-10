"""
Code validation and verification for generated Python code.

Provides functions to check code syntax and safety before execution.
Includes CFG-based validation to ensure tool ordering respects preconditions.
Includes type checking for full type safety (Rust-based ty, extremely fast ~9ms startup).
"""

import ast
import logging
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Type Checking with ty (Astral's Rust-based type checker)
# ============================================================================

def check_ty_types(full_code: str) -> Tuple[bool, Optional[str]]:
    """
    Run ty type checking on the full code (definition + generated).
    
    Creates a temporary file with the code and runs ty (Astral's Rust-based type checker) on it.
    ty is extremely fast (~9ms startup) and has excellent Pydantic support.
    
    Args:
        full_code: Complete Python code including definitions and generated code
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Create a temporary directory with both the code file and a config file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            code_path = temp_dir_path / "code.py"
            pyproject_path = temp_dir_path / "pyproject.toml"
            
            # Write the code
            code_path.write_text(full_code)
            
            # Write ty config to balance strictness with LLM-generated code
            # Key goals:
            # 1. Catch wrong attribute access (e.g., details.price when price is nested)
            # 2. Catch dict access on Pydantic models (e.g., details["items"] instead of details.items)
            # 3. Catch type mismatches (e.g., int assigned to str field)
            # 4. Allow possibly-unresolved references in LLM patterns (e.g., Optional narrowing in comprehensions)
            # 5. Allow possibly-missing attributes/imports (LLM code often has conditional definitions)
            #
            # Rules configuration:
            # - Error level: unresolved-attribute (wrong attrs), invalid-key (dict access on Pydantic),
            #                invalid-assignment (type mismatches), invalid-argument-type
            # - Ignore level: possibly-* rules (too strict for LLM code with conditionals)
            #                 invalid-return-type (gives false positives on code with early returns)
            config_content = """
[tool.ty.rules]
# Error on definite type errors
unresolved-attribute = "error"
invalid-key = "error"
invalid-assignment = "error"
invalid-argument-type = "error"
non-subscriptable = "error"
call-non-callable = "error"
missing-argument = "error"
unknown-argument = "error"

# Ignore return type errors (false positives with early returns)
invalid-return-type = "ignore"

# Ignore "possibly" errors (too strict for LLM code patterns)
possibly-unresolved-reference = "ignore"
possibly-missing-attribute = "ignore"
possibly-missing-import = "ignore"
possibly-missing-implicit-call = "ignore"
"""
            pyproject_path.write_text(config_content)
            
            # Run ty with concise output for easier parsing
            # Use --project to specify directory (ty will find pyproject.toml)
            result = subprocess.run(
                ["ty", "check", "--project", str(temp_dir_path), "--output-format", "concise", str(code_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # ty exits with code 1 if there are errors
            if result.returncode != 0:
                # Parse concise output (format: file:line:col: [rule] message)
                error_lines = []
                for line in result.stdout.strip().split('\n'):
                    if line and ':' in line:
                        # Extract line number and message from concise format
                        parts = line.split(':', 3)
                        if len(parts) >= 4:
                            line_num = parts[1]
                            message = parts[3].strip()
                            error_lines.append(f"Line {line_num}: {message}")
                        else:
                            error_lines.append(line)
                
                if error_lines:
                    return False, "\n".join(error_lines)
                else:
                    # Fallback if parsing fails - just return stdout
                    return False, result.stdout.strip() or "Type checking failed"
            
            # No errors found
            return True, None
            
    except subprocess.TimeoutExpired:
        return False, "Type checking timed out (>10s)"
    except FileNotFoundError:
        logger.debug("ty not found, skipping type checking")
        return True, None  # Skip silently if not installed
    except Exception as e:
        logger.warning(f"Type checking failed with exception: {e}")
        return True, None  # Don't fail validation if ty check fails


# ============================================================================
# Verify Base Class and Implementations
# ============================================================================

class Verify:
    """
    Base class for code verification strategies.
    
    Verifies that generated code is valid and safe to execute.
    """
    
    def verify(self, code, agent: Any) -> Tuple[bool, Optional[str]]:
        """
        Verify generated code.
        
        Args:
            code: Generated Python code (str) or tuple of (definition_code, generated_code)
            agent: Agent specification
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        raise NotImplementedError
    
    def _extract_code(self, code):
        """Extract generated_code from code (str or tuple)."""
        if isinstance(code, tuple):
            # If tuple, use only the generated_code part (second element)
            return code[1] if len(code) > 1 else code[0]
        return code
    
    def _extract_full_code(self, code):
        """Extract full code for verification (including definitions if available)."""
        if isinstance(code, tuple):
            # Concatenate definition_code and generated_code
            definition_code, generated_code = code[0], code[1] if len(code) > 1 else ""
            return (definition_code + "\n" + generated_code) if definition_code else generated_code
        return code


class BaseVerify(Verify):
    """
    Base verification implementation that checks:
    - Syntax validity
    - No dangerous operations (eval, exec, subprocess, etc)
    - Type safety using ty (Rust-based type checker)
    - Imports are allowed (they work fine in exec with proper state setup)
    """
    
    DANGEROUS_MODULES = {
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        'socket', 'urllib', 'requests', 'http',
        '__import__', 'eval', 'exec', 'compile'
    }
    
    # Allowed imports in generated code
    ALLOWED_IMPORTS = {
        'asyncio', 're', 'json', 'datetime', 'time',
        'pydantic', 'typing', 'collections', 'itertools',
        'functools', 'operator', 'math', 'random',
        'decimal', 'fractions', 'statistics',
    }
    
    def verify(self, code, agent: Any) -> Tuple[bool, Optional[str]]:
        """Verify code is syntactically valid, safe, and type-correct."""
        # Extract just the generated code for syntax checking
        generated_code = self._extract_code(code)
        
        # Check syntax
        try:
            tree = ast.parse(generated_code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check for dangerous operations
        for node in ast.walk(tree):
            # Check imports - allow them, they work fine in exec()
            # The executor provides all necessary imports and tools in its state
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom):
                    module_name = node.module
                else:
                    # For "import X" statements, get the first name
                    module_name = node.names[0].name if node.names else None
                
                if module_name:
                    # Only block dangerous modules
                    if any(dangerous in module_name for dangerous in self.DANGEROUS_MODULES):
                        return False, f"Dangerous import detected: {module_name}"
            
            # Check function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    if func_name in ['eval', 'exec', 'compile', '__import__']:
                        return False, f"Dangerous function detected: {func_name}"
        
        # Run type checking on full code (definition + generated) if available
        full_code = self._extract_full_code(code)
        if full_code != generated_code:  # Only run if we have definition code
            is_valid, type_error = check_ty_types(full_code)
            if not is_valid:
                return False, f"Type checking failed: {type_error}"
        
        return True, None


class IsLoop(Verify):
    """
    Verifies that code follows standard agentic loop pattern.
    
    When this verifier is present, the Runtime.aot() method can skip
    LLM generation and use a templated loop instead.
    
    Expected pattern:
    ```python
    while <condition>:
        result = await llm(...)
        if <terminal_condition>:
            break
    ```
    """
    
    def verify(self, code, agent: Any) -> Tuple[bool, Optional[str]]:
        """Check if code is a standard agentic loop."""
        # Extract just the generated code
        generated_code = self._extract_code(code)
        
        try:
            tree = ast.parse(generated_code)
        except SyntaxError:
            return False, "Invalid syntax"
        
        # Look for while loop with LLM call and break condition
        has_while_loop = False
        has_llm_call = False
        has_break = False
        
        for node in ast.walk(tree):
            # Check for while loop (any condition, not just True)
            if isinstance(node, ast.While):
                has_while_loop = True
            
            # Check for LLM/tool calls (check both function name and variable name)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id.lower()
                    if 'llm' in func_name or 'gpt' in func_name or 'openai' in func_name:
                        has_llm_call = True
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr.lower()
                    if 'llm' in attr_name or 'gpt' in attr_name:
                        has_llm_call = True
            
            # Check for break
            if isinstance(node, ast.Break):
                has_break = True
        
        if has_while_loop and has_llm_call and has_break:
            return True, None
        else:
            missing = []
            if not has_while_loop:
                missing.append("while loop")
            if not has_llm_call:
                missing.append("LLM call")
            if not has_break:
                missing.append("break statement")
            return False, f"Not a standard agentic loop pattern (missing: {', '.join(missing)})"


# ============================================================================
# Helper Functions for Code Checking
# ============================================================================

def check_code_candidate(
    code: str,
    agent: Optional[Any] = None,
    verifiers: Optional[list] = None
) -> Tuple[bool, Optional[str]]:
    """
    Check if generated code candidate is valid.
    
    Performs:
    1. Syntax validation
    2. Safety checks (no dangerous imports/functions)
    3. Optional custom verifiers
    
    Args:
        code: Generated Python code
        agent: Optional agent object for context
        verifiers: Optional list of Verify instances to run
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Always run basic verification
    basic_verify = BaseVerify()
    is_valid, error = basic_verify.verify(code, agent)
    if not is_valid:
        return False, error
    
    # Run custom verifiers if provided
    if verifiers:
        for verifier in verifiers:
            is_valid, error = verifier.verify(code, agent)
            if not is_valid:
                return False, error
    
    return True, None


def check_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check if code has valid Python syntax.
    
    Args:
        code: Python code to check
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def check_dangerous_ops(code: str) -> Tuple[bool, Optional[str]]:
    """
    Check for dangerous operations in code.
    
    Args:
        code: Python code to check
    
    Returns:
        Tuple of (is_safe, error_message)
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False, "Invalid syntax"
    
    dangerous_funcs = {'eval', 'exec', 'compile', '__import__'}
    dangerous_modules = {
        'os', 'sys', 'subprocess', 'shutil',
        'socket', 'urllib', 'requests', 'http'
    }
    
    for node in ast.walk(tree):
        # Check function calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in dangerous_funcs:
                return False, f"Dangerous function: {node.func.id}"
        
        # Check imports
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
            if module and any(d in module for d in dangerous_modules):
                return False, f"Dangerous import: {module}"
    
    return True, None


class IsFunction(Verify):
    """
    Verify that code contains at least one async function definition that's not a stub.
    
    Used by AOT mode to ensure generated code is a proper function.
    Extracts function metadata (name, args, return type) for validation.
    Ignores stub functions that raise NotImplementedError.
    """
    
    def verify(self, code, **kwargs) -> tuple[bool, Optional[str]]:
        """Check if code has at least one async function (not a stub) and extract metadata."""
        # Extract just the generated code (not definition code)
        generated_code = self._extract_code(code)
        
        # Parse code
        try:
            tree = ast.parse(generated_code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Find async function definitions at module level, excluding stubs
        func_defs = []
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef):
                # Check if it's a stub (raises NotImplementedError)
                is_stub = False
                for stmt in node.body:
                    if isinstance(stmt, ast.Raise):
                        if isinstance(stmt.exc, ast.Call):
                            if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                                is_stub = True
                                break
                
                # Only count non-stub functions
                if not is_stub:
                    func_defs.append(node)
        
        if len(func_defs) == 0:
            return False, "No async function implementation found. AOT mode requires a function (stubs don't count)."
        
        # Check that the function is async (should always be true here)
        func_def = func_defs[0]  # Use the first non-stub function
        if not isinstance(func_def, ast.AsyncFunctionDef):
            return False, "Function must be async (use 'async def')"
        
        # Extract metadata for validation
        func_name = func_def.name
        func_args = [arg.arg for arg in func_def.args.args]
        
        # Get return annotation if present
        return_annotation = None
        if func_def.returns:
            if isinstance(func_def.returns, ast.Name):
                return_annotation = func_def.returns.id
            elif isinstance(func_def.returns, ast.Constant):
                return_annotation = func_def.returns.value
        
        # Log extracted metadata
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Function metadata - name: {func_name}, args: {func_args}, return: {return_annotation}")
        
        return True, None


class QualitativeCriteria(Verify):
    """
    LLM-based verification using natural language criteria.
    
    Prompts an LLM to evaluate code against qualitative criteria (returns boolean).
    Supports multiple samples with parallel execution and majority voting.
    
    Args:
        expression: Natural language criteria description (e.g., "Code is readable and well-structured")
        llm: Tool instance for LLM calls (e.g., LLM("gpt-4o-mini"))
        num_samples: Number of parallel LLM calls to make (default: 1)
        min_samples_for_aggregation: Minimum successful samples needed (default: 1)
        min_pass: Minimum samples that must return True for overall True (default: 1)
    """
    
    def __init__(
        self,
        expression: str,
        llm: Any,  # Tool
        num_samples: int = 1,
        min_samples_for_aggregation: int = 1,
        min_pass: int = 1
    ):
        self.expression = expression
        self.llm = llm
        self.num_samples = num_samples
        self.min_samples_for_aggregation = min_samples_for_aggregation
        self.min_pass = min_pass
    
    def verify(self, code: str, agent: Any) -> Tuple[bool, Optional[str]]:
        """
        Verify code using LLM-based qualitative criteria with optional sampling.
        
        Returns:
            Tuple of (passes_criteria, error_message)
        """
        import asyncio
        
        async def _verify_async():
            # Create evaluation prompt
            prompt = f"""Evaluate the following Python code against this criteria:

Criteria: {self.expression}

Code:
```python
{code}
```

Return ONLY "true" or "false" (lowercase, no quotes, no explanation).
"""
            
            # Run multiple samples in parallel if requested
            if self.num_samples <= 1:
                # Single sample
                result = await self.llm(content=prompt)
                # Handle both string and LLMOutput responses
                if isinstance(result, str):
                    response = result.strip().lower()
                elif hasattr(result, "content"):
                    response = (result.content or "").strip().lower()
                else:
                    response = result.get("content", "").strip().lower()
                
                if response == "true":
                    return True, None
                elif response == "false":
                    return False, f"Code does not meet criteria: {self.expression}"
                else:
                    return False, f"Invalid LLM response: {response}"
            
            # Multiple samples - run in parallel
            tasks = [self.llm(content=prompt) for _ in range(self.num_samples)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Parse results
            valid_responses = []
            for result in results:
                if isinstance(result, Exception):
                    continue
                # Handle both string and LLMOutput responses
                if isinstance(result, str):
                    response = result.strip().lower()
                elif hasattr(result, "content"):
                    response = (result.content or "").strip().lower()
                else:
                    response = result.get("content", "").strip().lower()
                if response in ("true", "false"):
                    valid_responses.append(response == "true")
            
            # Check if we have enough valid samples
            if len(valid_responses) < self.min_samples_for_aggregation:
                return False, f"Insufficient valid LLM responses: {len(valid_responses)}/{self.min_samples_for_aggregation}"
            
            # Count how many passed
            num_passed = sum(valid_responses)
            
            if num_passed >= self.min_pass:
                return True, None
            else:
                return False, f"Code failed criteria ({num_passed}/{len(valid_responses)} samples passed, need {self.min_pass}): {self.expression}"
        
        # Run async verification - handle both sync and async contexts
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # We are in an async context, so we need to await directly
            # Create a new task and wait for it
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _verify_async())
                return future.result()
        except RuntimeError:
            # No event loop running, use asyncio.run
            return asyncio.run(_verify_async())


__all__ = [
    "Verify",
    "BaseVerify",
    "QualitativeCriteria",
    "IsLoop",
    "IsFunction",
    "check_code_candidate",
    "check_syntax",
    "check_dangerous_ops",
]
