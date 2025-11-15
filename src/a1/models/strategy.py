"""Strategy models for retry and code generation configuration."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AttemptStrategy(BaseModel):
    """
    Base strategy for operations that may require multiple attempts.

    Attributes:
        max_iterations: Maximum retry attempts (default: 3)
    """

    max_iterations: int = Field(default=3, description="Maximum retry attempts per operation")


class ParallelStrategy(AttemptStrategy):
    """
    Strategy for parallel batch processing with rate limit handling.

    Used for operations that process large datasets in chunks with parallel execution.
    Implements adaptive concurrency control with exponential backoff on rate limit errors.

    Attributes:
        max_iterations: Maximum retry attempts per chunk (default: 3)
        chunk_size: Number of items per chunk (default: 2048)
        max_parallel_chunks: Maximum chunks to process concurrently (default: 16)
    """

    chunk_size: int = Field(default=2048, description="Number of items per chunk/batch")
    max_parallel_chunks: int = Field(default=16, description="Maximum chunks to process in parallel")


class RetryStrategy(AttemptStrategy):
    """
    Retry strategy for LLM operations.

    Controls retry behavior and parallel execution for operations that may need
    multiple attempts to succeed (e.g., LLM calls with structured output validation).

    Attributes:
        max_iterations: Maximum retry attempts per operation (default: 3)
        num_candidates: Number of parallel attempts to execute (default: 1)
    """

    num_candidates: int = Field(default=1, description="Number of parallel attempts to execute")


class Strategy(RetryStrategy):
    """
    Configuration strategy for code generation (aot/jit).

    Extends RetryStrategy with additional parameters for cost-based selection,
    early stopping, and customizable generation/verification/cost pipelines.

    Attributes:
        max_iterations: Maximum refinement iterations per candidate (default: 3)
        num_candidates: Number of candidates to generate in parallel (default: 3)
        min_candidates_for_comparison: Minimum valid candidates before early comparison (default: 1)
        accept_cost_threshold: If set, immediately accept candidate below this cost (default: None)
        compare_cost_threshold: If set, compare early when min_candidates below this cost (default: None)
        generate: Custom code generation strategy (default: None, uses runtime's)
        verify: Custom verification strategy or list of strategies (default: None, uses runtime's)
        cost: Custom cost estimation strategy (default: None, uses runtime's)
        compact: Custom code compaction strategy (default: None, uses runtime's)
    """

    num_candidates: int = Field(default=3, description="Number of parallel attempts to execute")  # Override default
    min_candidates_for_comparison: int = Field(default=1, description="Minimum candidates before early comparison")
    accept_cost_threshold: float | None = Field(default=None, description="Immediately accept if cost below threshold")
    compare_cost_threshold: float | None = Field(
        default=None, description="Compare early when min_candidates below threshold"
    )
    generate: Any | None = Field(default=None, description="Custom code generation strategy")
    verify: Any | None = Field(default=None, description="Custom verification strategy or list")
    cost: Any | None = Field(default=None, description="Custom cost estimation strategy")
    compact: Any | None = Field(default=None, description="Custom code compaction strategy")

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = ["AttemptStrategy", "ParallelStrategy", "RetryStrategy", "Strategy"]
