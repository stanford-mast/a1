"""
Context management functions that work with the global runtime.

These functions are separated from runtime.py to avoid circular dependencies.
They import get_runtime inside the functions to break the import cycle.
"""

from .context import Context


def get_context(key: str = "main"):
    """
    Get or create a context by key.

    Args:
        key: Context key (default: "main")

    Returns:
        Context object
    """
    from .runtime import get_runtime

    runtime = get_runtime()
    if key not in runtime.CTX:
        # Create new context, linking to runtime for auto-save
        ctx = Context()
        # Link context to runtime for persistence
        if runtime.keep_updated and runtime.file_path:
            ctx._runtime_save = runtime._save
            ctx.keep_updated = True  # Enable auto-save for this context
        runtime.CTX[key] = ctx
        # Trigger initial save if runtime is persistent
        if runtime.keep_updated and runtime.file_path:
            runtime._save()
    return runtime.CTX[key]


def new_context(label: str = "intermediate", branch_from: Context | None = None):
    """
    Create a new context with auto-generated unique name and register it in Runtime.

    Context names follow pattern: {label}_{suffix} where suffix is a, b, c, ..., z, aa, ab, etc.

    Args:
        label: Label prefix for the context (e.g., "attempt", "intermediate", "main")
        branch_from: Optional source context to copy messages from

    Returns:
        Newly created Context object registered in Runtime.CTX

    Examples:
        >>> ctx1 = new_context("attempt")  # Creates "attempt_a"
        >>> ctx2 = new_context("attempt")  # Creates "attempt_b"
        >>> ctx3 = new_context("intermediate")  # Creates "intermediate_a"
    """
    from .runtime import get_runtime

    runtime = get_runtime()

    # Generate unique suffix (a, b, c, ..., z, aa, ab, ...)
    def gen_suffix(n):
        """Generate suffix: 0->a, 1->b, ..., 25->z, 26->aa, 27->ab, etc."""
        result = ""
        while True:
            result = chr(ord("a") + (n % 26)) + result
            n //= 26
            if n == 0:
                break
            n -= 1  # Adjust for aa coming after z
        return result

    # Find next available suffix for this label
    counter = 0
    while True:
        suffix = gen_suffix(counter)
        key = f"{label}_{suffix}"
        if key not in runtime.CTX:
            break
        counter += 1

    # Create new context
    ctx = Context()

    # Copy messages from source if provided
    if branch_from is not None:
        ctx.messages = branch_from.messages.copy()

    # Link context to runtime for persistence
    if runtime.keep_updated and runtime.file_path:
        ctx._runtime_save = runtime._save
        ctx.keep_updated = True

    runtime.CTX[key] = ctx

    # Trigger initial save if runtime is persistent
    if runtime.keep_updated and runtime.file_path:
        runtime._save()

    return ctx


__all__ = ["get_context", "new_context"]
