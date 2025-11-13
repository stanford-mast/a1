"""
Strategy classes for code generation, verification, and cost estimation.

This module provides backward compatibility by re-exporting classes from:
- codegen: Code generation strategies
- codecheck: Code verification strategies
- codecost: Cost estimation strategies
- cfg_builder: Control flow graph builder

For new code, prefer importing directly from the specific modules.
"""

# Re-export from individual modules for backward compatibility
from .cfg_builder import (
    BasicBlock,
    CFGBuilder,
)
from .codecheck import (
    BaseVerify,
    IsLoop,
    Verify,
    check_code_candidate,
    check_dangerous_ops,
    check_syntax,
)
from .extra_codecheck import (
    IsFunction,
    QualitativeCriteria,
)
from .codecost import (
    LOOP_MULTIPLIER,
    TOOL_LATENCIES,
    BaseCost,
    Cost,
    QuantitativeCriteria,
    compute_code_cost,
)
from .codegen import (
    EXAMPLE_CODE,
    EXAMPLE_FUNCTION,
    RULES,
    BaseGenerate,
    Generate,
)

__all__ = [
    # Generation
    "Generate",
    "BaseGenerate",
    "EXAMPLE_CODE",
    "EXAMPLE_FUNCTION",
    "RULES",
    # Verification
    "Verify",
    "BaseVerify",
    "QualitativeCriteria",
    "IsLoop",
    "IsFunction",
    "check_code_candidate",
    "check_syntax",
    "check_dangerous_ops",
    # Cost
    "Cost",
    "BaseCost",
    "QuantitativeCriteria",
    "compute_code_cost",
    "TOOL_LATENCIES",
    "LOOP_MULTIPLIER",
    # CFG
    "CFGBuilder",
    "BasicBlock",
]
