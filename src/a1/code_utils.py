"""
Utilities for code analysis, generation, and transformation.

Provides helpers for:
- Extracting code structure (functions, imports, etc)
- Code wrapping and indentation
- Result extraction from execution
- Function detection and analysis
- Schema management for code execution

Similar to schema_utils.py but for code rather than schemas.
"""

import ast
import asyncio
import logging
import re
import textwrap
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================================
# Nested Schema Generation (for code generation with arbitrarily nested schemas)
# ============================================================================


def generate_nested_pydantic_classes(schema: dict[str, Any], class_prefix: str, lines: list[str]) -> str:
    """
    Recursively generate nested Pydantic classes from a JSON schema.
    Returns the type hint for the schema.

    Handles:
    - Simple types (string, number, integer, boolean, array, object)
    - Nullable types (["string", "null"])
    - Required vs optional fields
    - Nested objects (full recursive support)
    - Arrays of nested objects (List[NestedModel])
    - Enums (Literal types)

    Args:
        schema: JSON schema (can be object, array, or primitive type)
        class_prefix: Prefix for generated class names
        lines: List to append generated class definitions to

    Returns:
        Type hint string (e.g., "str", "List[MenuItem]", "Optional[Restaurant]")

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "items": {
        ...             "type": "array",
        ...             "items": {
        ...                 "type": "object",
        ...                 "properties": {
        ...                     "name": {"type": "string"},
        ...                     "price": {"type": "number"}
        ...                 },
        ...                 "required": ["name"]
        ...             }
        ...         }
        ...     },
        ...     "required": ["items"]
        ... }
        >>> lines = []
        >>> type_hint = generate_nested_pydantic_classes(schema, "Output", lines)
        >>> print("\\n".join(lines))
    """

    schema_type = schema.get("type")

    # Handle enums - use Literal for type safety
    if "enum" in schema:
        enum_values = schema["enum"]
        non_null_values = [v for v in enum_values if v is not None]
        has_null = None in enum_values

        if non_null_values:
            formatted_values = []
            for v in non_null_values:
                if isinstance(v, str):
                    formatted_values.append(f"'{v}'")
                else:
                    formatted_values.append(str(v))
            literal_str = f"Literal[{', '.join(formatted_values)}]"
            return f"Optional[{literal_str}]" if has_null else literal_str
        elif has_null:
            return "Optional[Any]"

    # Handle type unions (e.g., ["string", "null"])
    if isinstance(schema_type, list):
        non_null_types = [t for t in schema_type if t != "null"]
        is_nullable = "null" in schema_type

        if len(non_null_types) == 1:
            temp_schema = dict(schema)
            temp_schema["type"] = non_null_types[0]
            inner_type = generate_nested_pydantic_classes(temp_schema, class_prefix, lines)
            return f"Optional[{inner_type}]" if is_nullable else inner_type
        else:
            type_hints = []
            for t in non_null_types:
                temp_schema = dict(schema)
                temp_schema["type"] = t
                type_hints.append(generate_nested_pydantic_classes(temp_schema, class_prefix, lines))
            union_str = f"Union[{', '.join(type_hints)}]"
            return f"Optional[{union_str}]" if is_nullable else union_str

    # Handle arrays
    if schema_type == "array":
        items_schema = schema.get("items", {})
        if not items_schema:
            return "List[Any]"

        item_type = generate_nested_pydantic_classes(items_schema, f"{class_prefix}Item", lines)
        return f"List[{item_type}]"

    # Handle objects - generate a new Pydantic class
    if schema_type == "object":
        properties = schema.get("properties", {})
        if not properties:
            return "Dict[str, Any]"

        # First, recursively generate nested classes for all properties
        prop_type_hints = {}
        for prop_name, prop_schema in properties.items():
            # Convert snake_case to PascalCase for class names
            pascal_prop = "".join(word.capitalize() for word in prop_name.split("_"))
            prop_type_hint = generate_nested_pydantic_classes(prop_schema, f"{class_prefix}{pascal_prop}", lines)
            prop_type_hints[prop_name] = prop_type_hint

        # Now generate the class definition for this object
        lines.append(f"class {class_prefix}(BaseModel):")

        required = schema.get("required", [])
        has_fields = False

        for prop_name, prop_schema in properties.items():
            has_fields = True
            prop_desc = prop_schema.get("description", "")
            is_required = prop_name in required
            prop_type_hint = prop_type_hints[prop_name]

            # Build Field() arguments with validators from JSON schema
            field_args = []
            default_value = "..." if is_required else "None"
            field_args.append(default_value)
            
            # Add description
            if prop_desc:
                # Escape quotes in description
                escaped_desc = prop_desc.replace('"', '\\"')
                field_args.append(f'description="{escaped_desc}"')
            
            # Add validators from JSON schema
            if "pattern" in prop_schema:
                pattern = prop_schema["pattern"]
                # Escape backslashes and quotes for raw string
                escaped_pattern = pattern.replace("\\", "\\\\")
                field_args.append(f"pattern=r'{escaped_pattern}'")
            
            if "minimum" in prop_schema:
                field_args.append(f"ge={prop_schema['minimum']}")
            
            if "maximum" in prop_schema:
                field_args.append(f"le={prop_schema['maximum']}")
            
            if "exclusiveMinimum" in prop_schema:
                field_args.append(f"gt={prop_schema['exclusiveMinimum']}")
            
            if "exclusiveMaximum" in prop_schema:
                field_args.append(f"lt={prop_schema['exclusiveMaximum']}")
            
            if "minLength" in prop_schema:
                field_args.append(f"min_length={prop_schema['minLength']}")
            
            if "maxLength" in prop_schema:
                field_args.append(f"max_length={prop_schema['maxLength']}")
            
            if "minItems" in prop_schema:
                field_args.append(f"min_length={prop_schema['minItems']}")
            
            if "maxItems" in prop_schema:
                field_args.append(f"max_length={prop_schema['maxItems']}")
            
            # Generate field definition
            if field_args:
                field_def = f"Field({', '.join(field_args)})"
                lines.append(f"    {prop_name}: {prop_type_hint} = {field_def}")
            else:
                # No validators, use simple default
                if is_required:
                    lines.append(f"    {prop_name}: {prop_type_hint}")
                else:
                    if "Optional" in prop_type_hint or prop_type_hint.startswith("Union"):
                        lines.append(f"    {prop_name}: {prop_type_hint} = None")
                    else:
                        lines.append(f"    {prop_name}: Optional[{prop_type_hint}] = None")

        if not has_fields:
            lines.append("    pass")

        lines.append("")
        return class_prefix

    # Helper function for primitive types
    def json_type_to_python(json_type: str) -> str:
        type_map = {
            "string": "str",
            "number": "float",
            "integer": "int",
            "boolean": "bool",
            "array": "List[Any]",
            "object": "Dict[str, Any]",
        }
        return type_map.get(json_type, "Any")

    # Primitive types
    return json_type_to_python(schema_type) if schema_type else "Any"


# ============================================================================
# Code Structure Analysis
# ============================================================================


def extract_async_functions(code: str) -> list[tuple[str, ast.AsyncFunctionDef]]:
    """
    Extract all async function definitions from code.

    Args:
        code: Python code string

    Returns:
        List of (function_name, ast_node) tuples

    Example:
        >>> code = "async def foo(): pass\\nasync def bar(): pass"
        >>> funcs = extract_async_functions(code)
        >>> [name for name, _ in funcs]
        ['foo', 'bar']
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    funcs = []
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef):
            funcs.append((node.name, node))
    return funcs


def extract_non_stub_async_functions(code: str) -> list[tuple[str, ast.AsyncFunctionDef]]:
    """
    Extract async function definitions that aren't stubs (NotImplementedError).

    Args:
        code: Python code string

    Returns:
        List of (function_name, ast_node) tuples for non-stub functions

    Example:
        >>> code = '''
        ... async def stub():
        ...     raise NotImplementedError
        ... async def real():
        ...     return 42
        ... '''
        >>> funcs = extract_non_stub_async_functions(code)
        >>> len(funcs)
        1
        >>> funcs[0][0]
        'real'
    """
    all_funcs = extract_async_functions(code)
    non_stubs = []

    for func_name, func_node in all_funcs:
        is_stub = _is_stub_function(func_node)
        if not is_stub:
            non_stubs.append((func_name, func_node))

    return non_stubs


def _is_stub_function(func_node: ast.AsyncFunctionDef) -> bool:
    """Check if function only raises NotImplementedError."""
    for stmt in func_node.body:
        if isinstance(stmt, ast.Raise):
            if isinstance(stmt.exc, ast.Call):
                if isinstance(stmt.exc.func, ast.Name) and stmt.exc.func.id == "NotImplementedError":
                    return True
    return False


def get_function_signature(func_node: ast.AsyncFunctionDef) -> tuple[str, list[str], str | None]:
    """
    Extract function metadata from AST node.

    Args:
        func_node: AsyncFunctionDef AST node

    Returns:
        Tuple of (name, arg_names, return_type_annotation)

    Example:
        >>> code = "async def greet(name: str) -> str: pass"
        >>> tree = ast.parse(code)
        >>> func = tree.body[0]
        >>> name, args, ret = get_function_signature(func)
        >>> name
        'greet'
        >>> args
        ['name']
    """
    func_name = func_node.name
    func_args = [arg.arg for arg in func_node.args.args]

    return_annotation = None
    if func_node.returns:
        if isinstance(func_node.returns, ast.Name):
            return_annotation = func_node.returns.id
        elif isinstance(func_node.returns, ast.Constant):
            return_annotation = str(func_node.returns.value)

    return func_name, func_args, return_annotation


def has_code_structure(code: str, structure_type: str) -> bool:
    """
    Check if code contains a specific structure type.

    Args:
        code: Python code string
        structure_type: 'while_loop', 'for_loop', 'if_statement', 'function'

    Returns:
        True if structure found

    Example:
        >>> code = "while True: pass"
        >>> has_code_structure(code, 'while_loop')
        True
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    type_map = {
        "while_loop": ast.While,
        "for_loop": ast.For,
        "if_statement": ast.If,
        "function": ast.AsyncFunctionDef,
    }

    target_type = type_map.get(structure_type)
    if not target_type:
        return False

    for node in ast.walk(tree):
        if isinstance(node, target_type):
            return True

    return False


# ============================================================================
# Code Wrapping and Transformation
# ============================================================================


def extract_future_imports(code: str) -> tuple[list[str], str]:
    """
    Extract __future__ imports from code (must stay at file beginning).

    Python requires __future__ imports to be at the very start of the file,
    before any other code. This extracts them so they can be placed correctly
    when wrapping code in functions.

    Args:
        code: Python code string

    Returns:
        Tuple of (future_imports_list, remaining_code)

    Example:
        >>> code = "from __future__ import annotations\\nprint('hi')"
        >>> futures, rest = extract_future_imports(code)
        >>> futures
        ['from __future__ import annotations']
        >>> rest.strip()
        "print('hi')"
    """
    future_imports = []
    remaining_lines = []

    for line in code.split("\n"):
        if line.strip().startswith("from __future__"):
            future_imports.append(line)
        else:
            remaining_lines.append(line)

    return future_imports, "\n".join(remaining_lines)


def extract_nested_models(schema_class: Any) -> dict[str, type]:
    """
    Extract all nested Pydantic model classes from a schema.

    Given a Pydantic BaseModel, recursively extract all nested model
    types referenced in its fields and return them as a dict mapping
    class names to class objects.

    Args:
        schema_class: A Pydantic BaseModel class

    Returns:
        Dict mapping class name to class object for all nested models

    Example:
        >>> class Address(BaseModel):
        ...     street: str
        ...     city: str
        >>> class Contact(BaseModel):
        ...     address: Address
        >>> models = extract_nested_models(Contact)
        >>> 'Address' in models
        True
    """
    from pydantic import BaseModel

    result = {}
    visited = set()

    def extract_recursive(model_class):
        if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
            return

        if model_class in visited or not hasattr(model_class, "model_fields"):
            return

        visited.add(model_class)

        for field_name, field_info in model_class.model_fields.items():
            field_type = field_info.annotation

            # Handle Optional[X], Union[X, ...], etc.
            if hasattr(field_type, "__origin__"):
                if hasattr(field_type, "__args__"):
                    # For Optional[X], try to extract the non-None type
                    for arg in field_type.__args__:
                        if arg is not type(None):
                            field_type = arg
                            break

            # Handle List[X], Dict[str, X], etc.
            if hasattr(field_type, "__origin__") and hasattr(field_type, "__args__"):
                field_type = field_type.__args__[0]

            # If it's a Pydantic model, add it and recurse
            if isinstance(field_type, type) and issubclass(field_type, BaseModel):
                result[field_type.__name__] = field_type
                extract_recursive(field_type)

    extract_recursive(schema_class)
    return result


def wrap_code_in_async_function(code: str) -> str:
    """
    Wrap code in async function, respecting __future__ import constraints.

    This wraps arbitrary Python code in an async function so it can be
    executed with top-level await. It properly handles __future__ imports
    which must appear at the file beginning.

    Args:
        code: Python code to wrap

    Returns:
        Code wrapped in async def __exec_wrapper()

    Example:
        >>> code = "x = 42\\nprint(x)"
        >>> wrapped = wrap_code_in_async_function(code)
        >>> "async def __exec_wrapper" in wrapped
        True
        >>> "return locals()" in wrapped
        True
    """
    future_imports, remaining_code = extract_future_imports(code)

    wrapped = ""
    if future_imports:
        wrapped += "\n".join(future_imports) + "\n\n"

    wrapped += "async def __exec_wrapper():\n"
    for line in remaining_code.split("\n"):
        wrapped += f"    {line}\n"
    wrapped += "    return locals()\n"

    return wrapped


def wrap_code_body_as_function(
    code: str, function_name: str, input_schema: Any, output_schema_name: str = "Output"
) -> str:
    """
    If code is a function body, wrap it with proper async function signature.

    Converts code like:
        x = await tool(...)
        result = process(x)

    Into:
        async def my_function(input_param: str) -> Output:
            x = await tool(...)
            result = process(x)

    Args:
        code: Python code (function body)
        function_name: Name for generated function
        input_schema: Pydantic model with model_fields for parameters
        output_schema_name: Name of output schema class (default: "Output")

    Returns:
        Code with function signature wrapper

    Example:
        >>> from pydantic import BaseModel
        >>> class Input(BaseModel):
        ...     query: str
        >>> code = "result = query.upper()"
        >>> wrapped = wrap_code_body_as_function(code, "process", Input, "Output")
        >>> "async def process" in wrapped
        True
    """
    # Build parameter list from input schema
    params = []
    if hasattr(input_schema, "model_fields"):
        for field_name, field_info in input_schema.model_fields.items():
            if hasattr(field_info.annotation, "__name__"):
                field_type = field_info.annotation.__name__
            else:
                field_type = str(field_info.annotation)
            params.append(f"{field_name}: {field_type}")

    param_str = ", ".join(params) or "**kwargs"

    # Create function wrapper
    wrapped = f"async def {function_name}({param_str}) -> {output_schema_name}:\n"

    # Properly indent the code body
    dedented = textwrap.dedent(code)
    indented = textwrap.indent(dedented, "    ")
    wrapped += indented

    return wrapped


# ============================================================================
# Execution Result Handling
# ============================================================================


def detect_user_async_function(result_locals: dict[str, Any]) -> tuple[str, Any] | None:
    """
    Detect user-defined async functions in execution results.

    When code defines an async function and doesn't call it, we want to
    detect that and call it to get the result. This filters out wrapper
    functions and builtins.

    Args:
        result_locals: Dictionary of local variables from code execution

    Returns:
        Tuple of (function_name, function_object) or None if not found

    Example:
        >>> async def my_func(): return 42
        >>> result_locals = {'my_func': my_func, '__exec_wrapper': None, '_private': None}
        >>> name, func = detect_user_async_function(result_locals)
        >>> name
        'my_func'
    """
    user_funcs = [
        (name, func)
        for name, func in result_locals.items()
        if callable(func)
        and asyncio.iscoroutinefunction(func)
        and name not in ["__exec_wrapper"]
        and not name.startswith("_")
    ]

    return user_funcs[0] if user_funcs else None


def extract_execution_result(result_locals: dict[str, Any]) -> Any:
    """
    Extract final result from execution locals.

    Uses heuristics to find the result:
    1. Variable named 'output' (preferred)
    2. Variable named 'result'
    3. Last non-type value in locals
    4. None if nothing found

    Args:
        result_locals: Dictionary of local variables from code execution

    Returns:
        The extracted result value

    Example:
        >>> result_locals = {'output': 42, 'x': 1, 'y': 2}
        >>> extract_execution_result(result_locals)
        42

        >>> result_locals = {'result': 'hello', 'x': 1}
        >>> extract_execution_result(result_locals)
        'hello'
    """
    if "output" in result_locals:
        return result_locals["output"]
    elif "result" in result_locals:
        return result_locals["result"]
    elif result_locals:
        # Get last non-type value
        for value in reversed(list(result_locals.values())):
            if not isinstance(value, type):
                return value

    return None


def clean_execution_locals(
    result_locals: dict[str, Any], tools: list[Any] | None = None, agent_schemas: tuple[Any, Any] | None = None
) -> dict[str, Any]:
    """
    Clean execution locals by removing transient values.

    Removes:
    - Wrapper function (__exec_wrapper)
    - Tool objects and their schemas
    - Agent input/output schemas

    Args:
        result_locals: Execution locals dictionary
        tools: Optional list of Tool objects to remove
        agent_schemas: Optional tuple of (input_schema, output_schema)

    Returns:
        Cleaned dictionary with only user-defined variables

    Example:
        >>> result_locals = {
        ...     '__exec_wrapper': None,
        ...     'output': 42,
        ...     'tool_obj': object(),
        ...     'x': 1
        ... }
        >>> cleaned = clean_execution_locals(result_locals)
        >>> '__exec_wrapper' in cleaned
        False
        >>> 'output' in cleaned
        True
    """
    cleaned = result_locals.copy()

    # Remove wrapper function
    cleaned.pop("__exec_wrapper", None)

    # Remove tools and their schemas
    if tools:
        for tool in tools:
            cleaned.pop(tool.name, None)
            if hasattr(tool, "input_schema") and tool.input_schema:
                cleaned.pop(tool.input_schema.__name__, None)
            if hasattr(tool, "output_schema") and tool.output_schema:
                cleaned.pop(tool.output_schema.__name__, None)

    # Remove agent schemas
    if agent_schemas:
        input_schema, output_schema = agent_schemas
        if hasattr(input_schema, "__name__"):
            cleaned.pop(input_schema.__name__, None)
        if hasattr(output_schema, "__name__"):
            cleaned.pop(output_schema.__name__, None)

    return cleaned


# ============================================================================
# Executor State Management
# ============================================================================


def inject_schemas_to_executor_state(executor: Any, agent: Any) -> None:
    """
    Add input/output schemas to executor state for code execution.

    Args:
        executor: BaseExecutor instance
        agent: Agent with input_schema and output_schema

    Example:
        >>> # executor.state will now contain agent schemas
        >>> inject_schemas_to_executor_state(executor, agent)
    """
    if hasattr(agent.input_schema, "__name__"):
        executor.state[agent.input_schema.__name__] = agent.input_schema
    if hasattr(agent.output_schema, "__name__"):
        executor.state[agent.output_schema.__name__] = agent.output_schema


def clean_schemas_from_executor_state(executor: Any, agent: Any) -> None:
    """
    Remove input/output schemas from executor state after execution.

    Args:
        executor: BaseExecutor instance
        agent: Agent with input_schema and output_schema
    """
    if hasattr(agent.input_schema, "__name__"):
        executor.state.pop(agent.input_schema.__name__, None)
    if hasattr(agent.output_schema, "__name__"):
        executor.state.pop(agent.output_schema.__name__, None)


def populate_definitions_to_executor_state(executor: Any, definition_code: str) -> None:
    """
    Execute definition_code and extract symbols into executor.state.

    Definition code contains tool signatures, schemas, and imports that
    the generated code needs. We execute it in a sandbox and add the
    resulting symbols to executor.state.

    Args:
        executor: BaseExecutor instance
        definition_code: Code with definitions to extract

    Example:
        >>> definition_code = "def helper(): return 42\\nOutput = type('Output', (), {})"
        >>> populate_definitions_to_executor_state(executor, definition_code)
        >>> 'Output' in executor.state
        True
    """
    try:
        # Create temp environment with necessary symbols for definition code
        from typing import Any, Optional, Union

        from pydantic import BaseModel, Field

        temp_env: dict[str, Any] = {
            "BaseModel": BaseModel,
            "Field": Field,
            "Optional": Optional,
            "List": list,
            "Dict": dict,
            "Any": Any,
            "Union": Union,
        }

        exec(definition_code, temp_env, temp_env)
        # Filter out builtins and dunder names
        def_vars = {k: v for k, v in temp_env.items() if not k.startswith("__")}
        executor.state.update(def_vars)
    except Exception as e:
        logger.warning(f"Could not execute definitions to extract symbols: {e}")


# ============================================================================
# Code Fixing (formerly in codefix.py)
# ============================================================================


def fix_asyncio_run(code: str) -> str:
    """
    Replace asyncio.run() calls with await.

    LLMs sometimes generate asyncio.run(func()) which fails in async context.
    We need to replace it with: output = await func()

    Args:
        code: Python code string

    Returns:
        Fixed code with asyncio.run replaced
    """
    # Pattern: asyncio.run(anything())
    # Replace with: output = await anything()
    pattern = r"asyncio\.run\s*\(\s*([^)]+)\s*\(([^)]*)\)\s*\)"

    def replacement(match):
        func_name = match.group(1).strip()
        func_args = match.group(2).strip()
        if func_args:
            return f"output = await {func_name}({func_args})"
        else:
            return f"output = await {func_name}()"

    fixed = re.sub(pattern, replacement, code)

    if fixed != code:
        logger.info("Fixed asyncio.run() call in generated code")

    return fixed


def extract_function_name(code: str) -> str | None:
    """
    Extract the name of the first async function definition in code.

    Args:
        code: Python code string

    Returns:
        Function name or None if no async function found
    """
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, ast.AsyncFunctionDef):
                return node.name
        return None
    except SyntaxError:
        return None


def has_function_call(code: str, func_name: str) -> bool:
    """
    Check if code contains a call to the given function.

    Args:
        code: Python code string
        func_name: Function name to look for

    Returns:
        True if function is called in the code
    """
    # Look for func_name( or await func_name(
    pattern = rf"\b{re.escape(func_name)}\s*\("
    return bool(re.search(pattern, code))


def append_function_call(code: str, func_name: str, call_args: str = "**validated.model_dump()") -> str:
    """
    Append a function call to code if not already present.

    Args:
        code: Python code string
        func_name: Function name to call
        call_args: Arguments to pass to function

    Returns:
        Code with function call appended
    """
    if has_function_call(code, func_name):
        logger.debug(f"Function {func_name} already called in code")
        return code

    logger.info(f"Appending call to {func_name}()")
    return code + f"\n\noutput = await {func_name}({call_args})"


def fix_generated_code(
    code: str, is_aot: bool = False, function_name: str = "generated_func", agent: Any = None
) -> str:
    """
    Fix generated code for execution.

    For AOT (functions): If code is a function body (not a full function), wrap it in async def with proper signature.
    For JIT (code blocks): Fix asyncio.run() calls to use await instead.

    Args:
        code: Raw generated code
        is_aot: True if this is AOT mode (function expected), False for JIT (code block)
        function_name: Name for the generated function (only used in AOT mode)
        agent: Agent instance (used to extract function signature for AOT)

    Returns:
        Fixed code ready for execution
    """
    # First, fix asyncio.run() calls in both modes
    code = fix_asyncio_run(code)

    # For AOT mode: if code doesn't already define a function, wrap it
    if is_aot:
        # Check if code already has a function definition
        try:
            tree = ast.parse(code)
            has_func_def = any(isinstance(node, ast.AsyncFunctionDef) for node in tree.body)
        except SyntaxError:
            has_func_def = False

        # If no function definition, wrap the code as a function body with proper signature
        if not has_func_def:
            # Build function signature from agent's input schema
            param_str = "**kwargs"
            if agent and hasattr(agent, "input_schema") and hasattr(agent.input_schema, "model_fields"):
                params = []
                for field_name, field_info in agent.input_schema.model_fields.items():
                    params.append(field_name)
                if params:
                    param_str = ", ".join(params)

            # Get output type from agent
            output_type = "Any"
            if agent and hasattr(agent, "output_schema") and hasattr(agent.output_schema, "__name__"):
                output_type = agent.output_schema.__name__

            # Wrap code in async function with proper signature
            lines = code.split("\n")
            wrapped_lines = [f"async def {function_name}({param_str}) -> {output_type}:"]
            for line in lines:
                if line.strip():  # Only indent non-empty lines
                    wrapped_lines.append(f"    {line}")
                else:
                    wrapped_lines.append(line)
            code = "\n".join(wrapped_lines)

    return code


def normalize_generated_code(code: str) -> str:
    """
    Normalize LLM-generated code - handles multiple output formats.

    The LLM may return:
    1. Just the function body (ideal)
    2. Function body with extra indentation
    3. Complete function definition (need to extract body)
    4. Function with wrong name (extract body, ignore name)
    5. Code wrapped in markdown fences
    6. Code with comments

    This method:
    - Removes markdown code fences (```python, ```)
    - Extracts function body if full function definition is present
    - Normalizes indentation (removes common leading whitespace)
    - Removes comments (as they violate generation RULES)

    Args:
        code: Raw generated code from LLM

    Returns:
        Normalized function body ready for execution

    Example:
        >>> code = '''```python
        ... async def parser(problem: str) -> Output:
        ...     result = await llm(problem)
        ...     return result
        ... ```'''
        >>> print(normalize_generated_code(code))
        result = await llm(problem)
        return result
    """
    if not code:
        return code

    # Step 1: Remove markdown code fences (but don't strip yet - preserve line indentation)
    code = re.sub(r"^```python\s*\n", "", code)
    code = re.sub(r"^```py\s*\n", "", code)
    code = re.sub(r"^```\s*\n", "", code)
    code = re.sub(r"\n```\s*$", "", code)

    # Step 2: Check if this is a complete function definition
    # Try to parse it as Python to see if it's a function
    try:
        tree = ast.parse(code)
        # Check if the code is a single async function definition
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.AsyncFunctionDef):
            func_def = tree.body[0]

            # Extract just the function body
            # Get the source lines
            lines = code.splitlines()

            # Find where the function body starts (after the signature and docstring)
            body_start_line = None

            for i, node in enumerate(func_def.body):
                # Skip docstring if present
                if i == 0 and isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
                    continue
                # First non-docstring statement
                body_start_line = node.lineno - 1  # Convert to 0-indexed
                break

            if body_start_line is not None:
                # Extract lines from body_start_line onwards
                body_lines = lines[body_start_line:]
                code = "\n".join(body_lines)
    except SyntaxError:
        # Not a valid function definition, treat as body
        pass

    # Step 3: Normalize indentation (remove common leading whitespace)
    lines = code.splitlines()
    if not lines:
        return code

    # Find minimum indentation (ignoring empty lines)
    min_indent = float("inf")
    for line in lines:
        if line.strip():  # Skip empty lines
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)

    # Remove the common indentation
    if min_indent > 0 and min_indent != float("inf"):
        normalized_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                normalized_lines.append(line[min_indent:])
            else:  # Empty line
                normalized_lines.append("")
        code = "\n".join(normalized_lines)

    # Step 4: Remove comments (they violate RULES)
    lines = code.splitlines()
    code_lines = []
    for line in lines:
        # Remove inline comments but keep strings with # in them
        stripped = line.strip()
        if stripped.startswith("#"):
            # Skip full-line comments
            continue
        # Keep the line (inline comments in strings are preserved)
        code_lines.append(line)

    return "\n".join(code_lines).strip()


# ============================================================================
# Output Validation
# ============================================================================


def validate_code_output(raw_output: Any, output_schema: Any) -> Any:
    """
    Validate/convert execution output to match output_schema.

    Tries multiple strategies:
    1. If already correct schema instance, return as-is
    2. If Pydantic model (different class), convert via model_dump()
    3. If dict, instantiate schema directly
    4. If single-field schema, wrap raw value in that field
    5. Otherwise return raw output with warning

    Args:
        raw_output: Raw output from code execution
        output_schema: Pydantic BaseModel class

    Returns:
        Validated output instance or raw value if validation fails

    Example:
        >>> from pydantic import BaseModel
        >>> class Output(BaseModel):
        ...     message: str
        >>> result = validate_code_output("hello", Output)
        >>> result.message
        'hello'
    """
    # Check if already correct type
    if isinstance(raw_output, output_schema):
        return raw_output

    # Check if it's a Pydantic model but different class
    # (happens when code uses Output class from definitions vs agent's schema)
    if hasattr(raw_output, "model_dump") and hasattr(raw_output, "model_validate"):
        try:
            # Convert via model_dump to normalize it
            return output_schema(**raw_output.model_dump())
        except Exception:
            pass  # Fall through to other strategies

    try:
        if isinstance(raw_output, dict):
            return output_schema(**raw_output)

        fields = output_schema.model_fields
        if len(fields) == 1:
            field_name = list(fields.keys())[0]
            return output_schema(**{field_name: raw_output})

        raise ValueError(f"Cannot map output {raw_output} to schema with {len(fields)} fields")
    except Exception as e:
        logger.warning(f"Could not validate output against schema: {e}. Returning raw output.")
        return raw_output


__all__ = [
    # Analysis
    "extract_async_functions",
    "extract_non_stub_async_functions",
    "get_function_signature",
    "has_code_structure",
    # Wrapping
    "extract_future_imports",
    "wrap_code_in_async_function",
    "wrap_code_body_as_function",
    # Result handling
    "detect_user_async_function",
    "extract_execution_result",
    "clean_execution_locals",
    # State management
    "inject_schemas_to_executor_state",
    "clean_schemas_from_executor_state",
    "populate_definitions_to_executor_state",
    # Validation
    "validate_code_output",
    # Code fixing (formerly in codefix.py)
    "fix_asyncio_run",
    "extract_function_name",
    "has_function_call",
    "append_function_call",
    "fix_generated_code",
]
