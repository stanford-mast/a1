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
from pathlib import Path
from typing import Any, Tuple, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Type Checking with ty (Astral's Rust-based type checker)
# ============================================================================


def check_ty_types(full_code: str) -> tuple[bool, str | None]:
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
                timeout=10,
            )

            # ty exits with code 1 if there are errors
            if result.returncode != 0:
                # Parse concise output (format: file:line:col: [rule] message)
                error_lines = []
                for line in result.stdout.strip().split("\n"):
                    if line and ":" in line:
                        # Extract line number and message from concise format
                        parts = line.split(":", 3)
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

    def verify(self, code, agent: Any) -> tuple[bool, str | None]:
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
    - Field constraint validation on constant tool inputs (compile-time validation)
    - Imports are allowed (they work fine in exec with proper state setup)
    """

    DANGEROUS_MODULES = {
        "os",
        "sys",
        "subprocess",
        "shutil",
        "pathlib",
        "socket",
        "urllib",
        "requests",
        "http",
        "__import__",
        "eval",
        "exec",
        "compile",
    }

    # Allowed imports in generated code
    ALLOWED_IMPORTS = {
        "asyncio",
        "re",
        "json",
        "datetime",
        "time",
        "pydantic",
        "typing",
        "collections",
        "itertools",
        "functools",
        "operator",
        "math",
        "random",
        "decimal",
        "fractions",
        "statistics",
    }

    def verify(self, code, agent: Any) -> tuple[bool, str | None]:
        """Verify code is syntactically valid, safe, type-correct, and respects Field constraints."""
        # Extract just the generated code for syntax checking
        generated_code = self._extract_code(code)

        # Check syntax
        try:
            tree = ast.parse(generated_code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check for large enums requiring EM tool
        has_large_enum, enum_error = self._check_large_enums(agent)
        if has_large_enum:
            return False, enum_error
        
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
                    if func_name in ["eval", "exec", "compile", "__import__"]:
                        return False, f"Dangerous function detected: {func_name}"

        # Validate constant tool inputs against Field constraints (compile-time validation)
        if agent and hasattr(agent, 'tools'):
            is_valid, constraint_error = self._check_field_constraints(tree, agent)
            if not is_valid:
                return False, constraint_error

        # Run type checking on full code (definition + generated) if available
        full_code = self._extract_full_code(code)
        if full_code != generated_code:  # Only run if we have definition code
            is_valid, type_error = check_ty_types(full_code)
            if not is_valid:
                return False, f"Type checking failed: {type_error}"

        return True, None
    
    def _check_field_constraints(self, tree: ast.AST, agent: Any) -> Tuple[bool, Optional[str]]:
        """
        Validate constant inputs to tool calls against Pydantic Field constraints.
        
        Checks compile-time knowable values (constants, attribute access on input schema, etc.)
        against the tool's input schema Field validators (pattern, ge, le, min_length, etc.).
        
        Args:
            tree: AST of generated code
            agent: Agent instance with tools
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        from pydantic import ValidationError
        from .cfg_builder import ConstantExtractor
        
        # Build map of tool names to tool instances
        tool_map = {}
        if hasattr(agent, 'tools'):
            for tool in agent.tools:
                if hasattr(tool, 'name'):
                    tool_map[tool.name] = tool
        
        # Create constant extractor with agent input schema
        input_schema = getattr(agent, 'input_schema', None) if agent else None
        extractor = ConstantExtractor(tree, input_schema)
        
        # Find all function calls in the code
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            
            # Get function name
            func_name = None
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
            
            if not func_name or func_name not in tool_map:
                continue
            
            tool = tool_map[func_name]
            
            # Skip if tool has no input schema
            if not hasattr(tool, 'input_schema') or not tool.input_schema:
                continue
            
            # Extract constant keyword arguments
            const_kwargs = {}
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue  # **kwargs
                
                # Try to extract constant value
                const_value, is_const = extractor.extract_constant(keyword.value)
                if is_const and not isinstance(const_value, tuple):
                    # Skip marker tuples like ('__INPUT__', ...)
                    const_kwargs[keyword.arg] = const_value
            
            # Also handle positional arguments if we can map them to parameter names
            if hasattr(tool.input_schema, 'model_fields'):
                param_names = list(tool.input_schema.model_fields.keys())
                for i, arg in enumerate(node.args):
                    if i < len(param_names):
                        const_value, is_const = extractor.extract_constant(arg)
                        if is_const and not isinstance(const_value, tuple):
                            const_kwargs[param_names[i]] = const_value
            
            # If we have any constant arguments, validate them
            if const_kwargs:
                try:
                    # Create partial instance to validate just the constant fields
                    # This will trigger Field validators (pattern, ge, le, etc.)
                    tool.input_schema(**const_kwargs)
                except ValidationError as e:
                    # Format error nicely
                    errors = []
                    for error in e.errors():
                        field_path = '.'.join(str(x) for x in error['loc'])
                        msg = error['msg']
                        errors.append(f"{field_path}: {msg}")
                    
                    return False, (
                        f"Tool '{func_name}' constant input validation failed:\n  " + 
                        "\n  ".join(errors)
                    )
                except Exception as e:
                    # Other validation errors (e.g., type mismatches)
                    return False, f"Tool '{func_name}' constant input validation failed: {e}"
        
        return True, None
    
    def _check_large_enums(self, agent: Any) -> Tuple[bool, Optional[str]]:
        """
        Check if agent has large enums (>100 values) and verify EM tool availability.
        
        Large enums require the EM (Embedding) tool for semantic reduction.
        
        Returns:
            Tuple of (has_error, error_message)
        """
        from .schema_utils import detect_large_enums
        
        # Check agent input schema
        if hasattr(agent, 'input_schema') and hasattr(agent.input_schema, 'model_json_schema'):
            try:
                input_schema = agent.input_schema.model_json_schema()
                large_enums = detect_large_enums(input_schema, threshold=100)
                if large_enums:
                    # Check if EM tool is available in agent tools
                    has_em = self._has_em_tool(agent)
                    if not has_em:
                        enum_info = ", ".join([f"{path} ({size} values)" for path, size in large_enums])
                        return True, (
                            f"Agent input schema contains large enums requiring EM tool: {enum_info}. "
                            f"Add EM() to agent tools or reduce enum size to ≤100 values."
                        )
            except Exception:
                pass  # Ignore schema extraction errors
        
        # Check agent output schema
        if hasattr(agent, 'output_schema') and hasattr(agent.output_schema, 'model_json_schema'):
            try:
                output_schema = agent.output_schema.model_json_schema()
                large_enums = detect_large_enums(output_schema, threshold=100)
                if large_enums:
                    has_em = self._has_em_tool(agent)
                    if not has_em:
                        enum_info = ", ".join([f"{path} ({size} values)" for path, size in large_enums])
                        return True, (
                            f"Agent output schema contains large enums requiring EM tool: {enum_info}. "
                            f"Add EM() to agent tools or reduce enum size to ≤100 values."
                        )
            except Exception:
                pass
        
        # Check tool schemas
        if hasattr(agent, 'tools'):
            for tool in agent.tools:
                # Check tool input schema
                if hasattr(tool, 'input_schema') and hasattr(tool.input_schema, 'model_json_schema'):
                    try:
                        tool_input_schema = tool.input_schema.model_json_schema()
                        large_enums = detect_large_enums(tool_input_schema, threshold=100)
                        if large_enums:
                            has_em = self._has_em_tool(agent)
                            if not has_em:
                                tool_name = getattr(tool, 'name', 'unknown')
                                enum_info = ", ".join([f"{path} ({size} values)" for path, size in large_enums])
                                return True, (
                                    f"Tool '{tool_name}' input schema contains large enums requiring EM tool: {enum_info}. "
                                    f"Add EM() to agent tools or reduce enum size to ≤100 values."
                                )
                    except Exception:
                        pass
                
                # Check tool output schema
                if hasattr(tool, 'output_schema') and hasattr(tool.output_schema, 'model_json_schema'):
                    try:
                        tool_output_schema = tool.output_schema.model_json_schema()
                        large_enums = detect_large_enums(tool_output_schema, threshold=100)
                        if large_enums:
                            has_em = self._has_em_tool(agent)
                            if not has_em:
                                tool_name = getattr(tool, 'name', 'unknown')
                                enum_info = ", ".join([f"{path} ({size} values)" for path, size in large_enums])
                                return True, (
                                    f"Tool '{tool_name}' output schema contains large enums requiring EM tool: {enum_info}. "
                                    f"Add EM() to agent tools or reduce enum size to ≤100 values."
                                )
                    except Exception:
                        pass
        
        return False, None
    
    def _has_em_tool(self, agent: Any) -> bool:
        """Check if agent has EM tool available."""
        if not hasattr(agent, 'tools'):
            return False
        
        for tool in agent.tools:
            tool_name = getattr(tool, 'name', '').lower()
            if tool_name == 'em' or 'embedding' in tool_name:
                return True
        
        return False


class IsLoop(Verify):
    """
    Verifies that code follows standard agentic loop pattern.

    When this verifier is present, the Runtime.aot() method can skip
    LLM generation and use a templated loop instead.

    Accepts two patterns:

    1. Legacy pattern (explicit while loop):
    ```python
    while <condition>:
        result = await llm(...)
        if <terminal_condition>:
            break
    ```

    2. New pattern (LLM handles looping internally):
    ```python
    output = await llm(..., output_schema=...)
    ```
    The LLM tool internally loops until a terminal tool is called.
    """

    def verify(self, code, agent: Any) -> tuple[bool, str | None]:
        """Check if code is a standard agentic loop."""
        # Extract just the generated code
        generated_code = self._extract_code(code)

        try:
            tree = ast.parse(generated_code)
        except SyntaxError:
            return False, "Invalid syntax"

        # Check for either pattern:
        # 1. Legacy: while loop with LLM call and break
        # 2. New: LLM call with output_schema parameter (LLM handles looping)

        has_while_loop = False
        has_llm_call = False
        has_break = False
        has_llm_with_output_schema = False

        for node in ast.walk(tree):
            # Check for while loop (any condition, not just True)
            if isinstance(node, ast.While):
                has_while_loop = True

            # Check for LLM/tool calls (check both function name and variable name)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id.lower()
                    is_llm_call = "llm" in func_name or "gpt" in func_name or "openai" in func_name
                elif isinstance(node.func, ast.Attribute):
                    attr_name = node.func.attr.lower()
                    is_llm_call = "llm" in attr_name or "gpt" in attr_name
                else:
                    is_llm_call = False

                if is_llm_call:
                    has_llm_call = True
                    # Check if this LLM call has output_schema keyword argument
                    for keyword in node.keywords:
                        if keyword.arg == "output_schema":
                            has_llm_with_output_schema = True
                            break

            # Check for break
            if isinstance(node, ast.Break):
                has_break = True

        # Accept if:
        # 1. New pattern: LLM call with output_schema (LLM handles looping internally)
        # 2. Legacy pattern: while loop with LLM call and break
        if has_llm_with_output_schema:
            return True, None
        elif has_while_loop and has_llm_call and has_break:
            return True, None
        else:
            missing = []
            if not has_llm_call:
                missing.append("LLM call")
            elif not has_llm_with_output_schema and not has_break:
                missing.append("break statement (legacy pattern) or output_schema parameter (new pattern)")
            return False, f"Not a standard agentic loop pattern (missing: {', '.join(missing)})"


# ============================================================================
# Helper Functions for Code Checking
# ============================================================================


def check_code_candidate(code: str, agent: Any | None = None, verifiers: list | None = None) -> tuple[bool, str | None]:
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


def check_syntax(code: str) -> tuple[bool, str | None]:
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


def check_dangerous_ops(code: str) -> tuple[bool, str | None]:
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

    dangerous_funcs = {"eval", "exec", "compile", "__import__"}
    dangerous_modules = {"os", "sys", "subprocess", "shutil", "socket", "urllib", "requests", "http"}

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


__all__ = [
    "Verify",
    "BaseVerify",
    "IsLoop",
    "check_code_candidate",
    "check_syntax",
    "check_dangerous_ops",
]
