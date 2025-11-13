"""
Runtime for executing agents with code generation and execution.

Provides:
- Runtime: Configurable runtime with aot, jit, execute methods
- Global runtime management: get_runtime, set_runtime
- Context manager support
"""

import hashlib
import json
import logging
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from .context import BaseCompact, Context
from .executor import BaseExecutor
from .models import Agent, Message, Strategy, Tool
from .strategies import BaseCost, BaseGenerate, BaseVerify

if TYPE_CHECKING:
    from .executor import Executor
    from .strategies import Compact, Cost, Generate, Verify

logger = logging.getLogger(__name__)

# Global runtime context variable
_runtime_var: ContextVar[Optional["Runtime"]] = ContextVar("runtime", default=None)


class Runtime:
    """
    Configurable runtime for agent execution.

    The Runtime manages:
    - Code generation strategies
    - Code verification strategies
    - Cost estimation strategies
    - Context compaction strategies
    - Code execution
    - Conversation contexts (H)
    - Caching of compiled agents

    Args:
        generate: Code generation strategy
        verify: List of verification strategies
        cost: Cost estimation strategy
        compact: Context compaction strategy
        executor: Code executor
        cache_dir: Directory for caching compiled agents (default: .a1)
    """

    def __init__(
        self,
        generate: Optional["Generate"] = None,
        verify: list["Verify"] | None = None,
        cost: Optional["Cost"] = None,
        compact: Optional["Compact"] = None,
        executor: Optional["Executor"] = None,
        cache_dir: str = ".a1",
        strategy: Optional["Strategy"] = None,
        file_path: Path | None = None,
        keep_updated: bool = False,
    ):
        from typing import Any, Optional

        from pydantic import BaseModel, Field

        from .builtin_tools import LLM
        from .llm import no_context
        from .models import Strategy

        # Persistence settings
        self.file_path = Path(file_path) if file_path else None
        self.keep_updated = keep_updated

        # If strategy is provided, use its fields (they override individual params)
        if strategy is not None:
            generate = strategy.generate if strategy.generate is not None else generate
            verify = strategy.verify if strategy.verify is not None else verify
            cost = strategy.cost if strategy.cost is not None else cost
            compact = strategy.compact if strategy.compact is not None else compact
            # Store the strategy for later use
            self.strategy = strategy
        else:
            self.strategy = Strategy()  # Default strategy

        self.generate = generate or BaseGenerate()  # Defaults to gpt-4.1-mini
        self.verify = verify if verify is not None else [BaseVerify()]
        # Handle verify as single item or list
        if not isinstance(self.verify, list):
            self.verify = [self.verify]
        self.cost = cost or BaseCost()
        self.compact = compact or BaseCompact()

        # Create executor with necessary imports for generated code
        if executor is None:
            executor = BaseExecutor(
                additional_imports={
                    "get_context": get_context,
                    "no_context": no_context,
                    "BaseModel": BaseModel,
                    "Field": Field,
                    "Optional": Optional,
                    "List": list,
                    "Dict": dict,
                    "Any": Any,
                    "Union": Union,
                }
            )
        self.executor = executor

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Contexts: Dictionary of named message lists
        self.CTX: dict[str, Context] = {}

        # Current agent being executed (for tool calling)
        self.current_agent: Agent | None = None
        
        # Embedding cache: map from hash -> vector (or list of vectors)
        # Used by EM tools to avoid recomputing embeddings for repeated items.
        # Keys are hex hashes of the serialized item(s).
        self._embeddings_cache: dict[str, list[float]] = {}

    def _hash_string(self, s: str) -> str:
        """Compute a stable hash for a string to use as cache key."""
        return hashlib.sha256(s.encode('utf-8')).hexdigest()

    def get_or_compute_embedding(self, text: str, embed_fn) -> list[float]:
        """Return cached embedding for text or compute via embed_fn(text).

        embed_fn should be a callable that returns a list[float]. This centralizes
        caching so callers (EM tool) don't recompute embeddings repeatedly.
        """
        key = self._hash_string(text)
        if key in self._embeddings_cache:
            return self._embeddings_cache[key]
        vec = embed_fn(text)
        self._embeddings_cache[key] = vec
        return vec

    def get_embeddings_for_items(self, items: list[str], embed_fn) -> list[list[float]]:
        """Get embeddings for a list of stringified items, using cache.

        Returns list of vectors in the same order.
        """
        return [self.get_or_compute_embedding(it, embed_fn) for it in items]
        
        # Save initial state if persistence enabled
        if self.keep_updated and self.file_path:
            self._save()

    def _save(self):
        """Save runtime state to file if persistence is enabled."""
        if self.file_path:
            import json

            data = {
                "cache_dir": str(self.cache_dir),
                "contexts": {
                    # Use mode='json' to properly serialize datetime objects
                    name: [msg.model_dump(exclude_none=True, mode="json") for msg in ctx.messages]
                    for name, ctx in self.CTX.items()
                },
            }

            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(data, f, indent=2)

    @classmethod
    def from_file(cls, path: str, keep_updated: bool = False, **kwargs) -> "Runtime":
        """
        Load runtime state from a file, optionally enabling auto-save on changes.

        Args:
            path: Path to JSON file containing runtime state
            keep_updated: If True, auto-save runtime state on context changes
            **kwargs: Additional Runtime constructor arguments

        Returns:
            Runtime instance with loaded state

        Example:
            >>> runtime = Runtime.from_file("session.json", keep_updated=True)
            >>> ctx = get_context("main")  # Restored from file
            >>> ctx.user("Hello")  # Automatically saved
        """
        import json

        from .models import Message

        file_path = Path(path)

        if file_path.exists():
            with open(file_path) as f:
                data = json.load(f)

            # Extract cache_dir from saved state if not provided
            if "cache_dir" not in kwargs:
                kwargs["cache_dir"] = data.get("cache_dir", ".a1")
        else:
            data = {"contexts": {}}

        # Create runtime with persistence settings
        runtime = cls(file_path=file_path, keep_updated=keep_updated, **kwargs)

        # Restore contexts
        for name, messages_data in data.get("contexts", {}).items():
            messages = [Message(**msg) for msg in messages_data]
            ctx = Context(messages=messages)
            # Link context to runtime for auto-save
            if runtime.keep_updated:
                ctx._runtime_save = runtime._save
                ctx.keep_updated = True
            runtime.CTX[name] = ctx

        return runtime

    def __enter__(self):
        """Enter context manager - set as global runtime."""
        self._token = _runtime_var.set(self)
        return self

    def __exit__(self, *args):
        """Exit context manager - restore previous runtime."""
        _runtime_var.reset(self._token)

    async def aot(self, agent: Agent, cache: bool = True, strategy: Optional["Strategy"] = None) -> Tool:
        """
        Ahead-of-time compile an agent to a tool.

        Generates Python function code for the agent and caches it. Returns a Tool
        that executes the compiled code.

        If IsLoop verifier is present, uses a templated loop instead of LLM generation.

        Args:
            agent: Agent to compile
            cache: Whether to use cached compilation (default: True)
            strategy: Optional Strategy for generation config (overrides runtime strategy)

        Returns:
            Tool that executes the compiled agent
        """
        import asyncio

        from opentelemetry import trace

        from .models import Strategy
        from .strategies import IsFunction, IsLoop

        # Merge strategies: call strategy > runtime strategy > defaults
        if strategy is None:
            strategy = self.strategy
        else:
            # Merge with runtime strategy (call strategy takes precedence)
            merged = Strategy(
                max_iterations=strategy.max_iterations,
                num_candidates=strategy.num_candidates,
                min_candidates_for_comparison=strategy.min_candidates_for_comparison,
                accept_cost_threshold=strategy.accept_cost_threshold,
                compare_cost_threshold=strategy.compare_cost_threshold,
                generate=strategy.generate if strategy.generate is not None else self.strategy.generate,
                verify=strategy.verify if strategy.verify is not None else self.strategy.verify,
                cost=strategy.cost if strategy.cost is not None else self.strategy.cost,
                compact=strategy.compact if strategy.compact is not None else self.strategy.compact,
            )
            strategy = merged

        # Use strategy fields if provided, otherwise use runtime's
        generate = strategy.generate if strategy.generate is not None else self.generate
        verify = strategy.verify if strategy.verify is not None else self.verify
        if not isinstance(verify, list):
            verify = [verify]
        cost = strategy.cost if strategy.cost is not None else self.cost

        num_candidates = strategy.num_candidates
        max_retries = strategy.max_iterations

        tracer = trace.get_tracer(__name__)
        
        # Early verification: Check for large enums requiring EM tool BEFORE code generation
        for verifier in verify:
            if hasattr(verifier, '_check_large_enums'):
                has_error, error_msg = verifier._check_large_enums(agent)
                if has_error:
                    raise ValueError(f"Agent validation failed: {error_msg}")
        
        with tracer.start_as_current_span("aot") as span:
            span.set_attribute("agent.name", agent.name)
            span.set_attribute("cache.enabled", cache)
            span.set_attribute("num_candidates", num_candidates)

            # Check cache
            if cache:
                cache_key = self._get_cache_key(agent)
                cache_path = self.cache_dir / f"{cache_key}.py"

                if cache_path.exists():
                    logger.info(f"Loading cached compilation for {agent.name}")
                    code = cache_path.read_text()
                    span.set_attribute("cache.hit", True)
                    return self._code_to_tool(agent, code)

                span.set_attribute("cache.hit", False)

            # Check if we should use templated loop
            is_loop_verifiers = [v for v in verify if isinstance(v, IsLoop)]
            if is_loop_verifiers:
                # Use templated loop instead of generation
                logger.info(f"Using templated loop for {agent.name}")
                generated_code = self._generate_loop_template(agent)
                logger.info(f"Generated template code:\n{generated_code}")
                span.set_attribute("generation.type", "template")

                # For template verification, we just verify the generated code itself
                # (no separate definition_code needed for template)
                for verifier in verify:
                    is_valid, error = verifier.verify(generated_code, agent=agent)
                    if not is_valid:
                        raise RuntimeError(f"Template verification failed: {error}")
            else:
                # Generate code using LLM with parallel candidates
                logger.info(f"Generating {num_candidates} function candidates for {agent.name}")
                span.set_attribute("generation.type", "llm")

                # Generate candidates in parallel with retries
                async def generate_with_retries():
                    # Create a codegen context for this candidate
                    # This enables conversation continuity across retries
                    from .context import Context
                    codegen_context = Context()
                    
                    past_attempts = []
                    for attempt in range(max_retries):
                        result = await generate.generate(
                            agent=agent,
                            task=agent.description,
                            return_function=True,  # AOT generates functions
                            past_attempts=past_attempts if attempt > 0 else None,
                            context=codegen_context,  # Share context across retries
                        )

                        if not result:
                            continue

                        # Generate returns (definition_code, generated_code) tuple
                        definition_code, generated_code = result

                        # Fix the generated code (wrap if needed for AOT)
                        from .code_utils import fix_generated_code

                        fixed_code = fix_generated_code(
                            generated_code, is_aot=True, function_name=agent.name, agent=agent
                        )

                        # Concatenate definitions with fixed code for validation
                        full_code_for_verification = (
                            definition_code + "\n" + fixed_code if definition_code else fixed_code
                        )

                        # Validate with all verifiers (including IsFunction for AOT)
                        all_valid = True
                        validation_error = None

                        # First check IsFunction on JUST the fixed code (not concatenated)
                        is_function_verifier = IsFunction()
                        is_valid, error = is_function_verifier.verify(fixed_code, agent=agent)
                        if not is_valid:
                            all_valid = False
                            validation_error = f"IsFunction failed: {error}"
                        else:
                            # Then check other verifiers with tuple of (definition_code, fixed_code)
                            for verifier in verify:
                                is_valid, error = verifier.verify((definition_code, fixed_code), agent=agent)
                                if not is_valid:
                                    all_valid = False
                                    validation_error = error
                                    break

                        if all_valid:
                            # Compute cost using the strategy's cost estimator
                            code_cost = cost.compute_cost((definition_code, fixed_code), agent=agent)
                            return (fixed_code, code_cost, None)
                        else:
                            # Track attempt for retry
                            past_attempts.append((generated_code, validation_error))

                    # All retries failed
                    if past_attempts:
                        return (past_attempts[-1][0], float("inf"), past_attempts[-1][1])
                    return (None, float("inf"), "Failed to generate code")

                # Launch parallel generation tasks
                tasks = [generate_with_retries() for _ in range(num_candidates)]
                results = await asyncio.gather(*tasks)

                # Filter valid candidates and rank by cost
                valid_candidates = [(code, cost) for code, cost, error in results if error is None]

                if not valid_candidates:
                    # All candidates failed - report errors
                    errors = [error for _, _, error in results if error]
                    raise RuntimeError(f"All candidates failed validation: {errors}")

                # Select best candidate by cost
                generated_code, best_cost = min(valid_candidates, key=lambda x: x[1])
                logger.info(f"Selected best candidate with cost {best_cost} from {len(valid_candidates)} valid")
                span.set_attribute("generation.best_cost", best_cost)
                span.set_attribute("generation.num_valid", len(valid_candidates))

            # Cache ONLY the generated code (not definitions - they'll be reconstructed)
            # Tools will be provided by executor at runtime
            if cache:
                cache_path.write_text(generated_code)
                logger.info(f"Cached compilation for {agent.name}")

            return self._code_to_tool(agent, generated_code)

    async def jit(self, agent: Agent, strategy: Strategy | None = None, **kwargs) -> Any:
        """
        Just-in-time execute an agent.

        Generates and executes code on-the-fly without caching.
        Appends user/assistant messages to main context.

        Args:
            agent: Agent to execute
            strategy: Optional Strategy for generation config (default: Strategy())
            **kwargs: Input arguments matching agent's input_schema

        Returns:
            Output from the agent
        """
        import asyncio

        from opentelemetry import trace

        from .strategies import IsLoop

        # Handle auto-conversion of string input
        # If kwargs contains a single value and the agent's input schema
        # has exactly one field, try to auto-map intelligently
        if not kwargs:
            # No kwargs provided - this is okay, will fail validation if required
            validated_input = agent.input_schema()
        elif len(kwargs) == 1 and len(agent.input_schema.model_fields) == 1:
            # Single kwarg and single input field
            field_name = list(agent.input_schema.model_fields.keys())[0]
            kwarg_key = list(kwargs.keys())[0]
            kwarg_value = kwargs[kwarg_key]

            # Auto-convert by field name match or type compatibility
            if kwarg_key == field_name:
                # Exact field name match - use directly
                validated_input = agent.input_schema(**kwargs)
            else:
                # Different field name - try to auto-map, but validate type first
                try:
                    # Try mapping the value to the target field
                    # This will validate type compatibility via Pydantic
                    validated_input = agent.input_schema(**{field_name: kwarg_value})
                except Exception as e:
                    # If validation fails, raise with helpful error
                    raise ValueError(f"Cannot auto-map {kwarg_key}={repr(kwarg_value)} to field '{field_name}': {e}")
        else:
            # Multiple kwargs or multiple fields - use normal validation
            validated_input = agent.input_schema(**kwargs)
        input_dict = validated_input.model_dump()

        # Merge strategies: call strategy > runtime strategy > defaults
        from .models import Strategy

        if strategy is None:
            strategy = self.strategy
        else:
            # Merge with runtime strategy (call strategy takes precedence)
            merged = Strategy(
                max_iterations=strategy.max_iterations,
                num_candidates=strategy.num_candidates,
                min_candidates_for_comparison=strategy.min_candidates_for_comparison,
                accept_cost_threshold=strategy.accept_cost_threshold,
                compare_cost_threshold=strategy.compare_cost_threshold,
                generate=strategy.generate if strategy.generate is not None else self.strategy.generate,
                verify=strategy.verify if strategy.verify is not None else self.strategy.verify,
                cost=strategy.cost if strategy.cost is not None else self.strategy.cost,
                compact=strategy.compact if strategy.compact is not None else self.strategy.compact,
            )
            strategy = merged

        # Use strategy fields if provided, otherwise use runtime's
        generate = strategy.generate if strategy.generate is not None else self.generate
        verify = strategy.verify if strategy.verify is not None else self.verify
        if not isinstance(verify, list):
            verify = [verify]
        cost = strategy.cost if strategy.cost is not None else self.cost

        num_candidates = strategy.num_candidates
        max_retries = strategy.max_iterations

        tracer = trace.get_tracer(__name__)

        with tracer.start_as_current_span("jit") as span:
            span.set_attribute("agent.name", agent.name)
            span.set_attribute("num_candidates", num_candidates)

            # Get or create main context
            if "main" not in self.CTX:
                self.CTX["main"] = Context()

            context = self.CTX["main"]

            # Create attempt context by copying main context
            # This preserves conversation history while keeping attempts separate
            # Each code generation attempt gets its own context (attempt_a, attempt_b, etc.)
            attempt_context = new_context("attempt", branch_from=context)

            # Add user message to both main and attempt contexts
            # Main gets the clean successful history
            # Attempt tracks all the generation/execution attempts for this input
            user_message = json.dumps(input_dict)
            context.user(user_message)
            attempt_context.user(user_message)

            # Set current agent for tool execution
            self.current_agent = agent

            try:
                # Generate code using LLM with parallel candidates
                logger.info(f"Generating {num_candidates} code candidates for {agent.name}")

                # Generate candidates in parallel with retries
                async def generate_with_retries():
                    # Create a FRESH codegen context for this candidate
                    # Do NOT branch from attempt_context - that only has user task JSON
                    # Code generation needs its own clean context
                    from .context import Context
                    codegen_context = Context()
                    
                    past_attempts = []
                    for attempt in range(max_retries):
                        result = await generate.generate(
                            agent=agent,
                            task=json.dumps(input_dict),
                            return_function=False,  # JIT generates code blocks
                            past_attempts=past_attempts if attempt > 0 else None,
                            context=codegen_context,  # Share context across retries
                        )

                        if not result:
                            continue

                        # Generate returns (definition_code, generated_code) tuple
                        definition_code, generated_code = result

                        # Fix the generated code (just fix asyncio.run for JIT)
                        from .code_utils import fix_generated_code

                        fixed_code = fix_generated_code(generated_code, is_aot=False)

                        # For JIT, concatenate definitions with fixed code for validation
                        full_code = definition_code + "\n" + fixed_code if definition_code else fixed_code

                        # Validate with verifiers (skip IsLoop which is only for AOT)
                        all_valid = True
                        validation_error = None
                        for verifier in verify:
                            # Skip IsLoop verifier for JIT - it only applies to AOT
                            if isinstance(verifier, IsLoop):
                                continue
                            # Verify the full concatenated code
                            is_valid, error = verifier.verify(full_code, agent=agent)
                            if not is_valid:
                                all_valid = False
                                validation_error = error
                                break

                        if all_valid:
                            # Compute cost using the strategy's cost estimator
                            code_cost = cost.compute_cost((definition_code, fixed_code), agent=agent)
                            return (fixed_code, definition_code, code_cost, None)
                        else:
                            # Track attempt for retry
                            past_attempts.append((generated_code, validation_error))

                    # All retries failed
                    if past_attempts:
                        return (past_attempts[-1][0], "", float("inf"), past_attempts[-1][1])
                    return (None, "", float("inf"), "Failed to generate code")

                # Launch parallel generation tasks
                tasks = [generate_with_retries() for _ in range(num_candidates)]
                results = await asyncio.gather(*tasks)

                # Filter valid candidates and rank by cost
                valid_candidates = [(code, def_code, cost) for code, def_code, cost, error in results if error is None]

                if not valid_candidates:
                    # All candidates failed - report errors
                    errors = [error for _, _, _, error in results if error]
                    raise RuntimeError(f"All candidates failed validation: {errors}")

                # Try candidates in order of cost until one executes successfully
                # This provides execution-time retry in addition to validation-time retry
                valid_candidates.sort(key=lambda x: x[2])  # Sort by cost

                execution_errors = []
                for candidate_idx, (code, definition_code, candidate_cost) in enumerate(valid_candidates):
                    logger.info(
                        f"Trying candidate {candidate_idx + 1}/{len(valid_candidates)} with cost {candidate_cost}"
                    )

                    try:
                        # Use code_utils for schema injection
                        from .code_utils import (
                            clean_schemas_from_executor_state,
                            inject_schemas_to_executor_state,
                            populate_definitions_to_executor_state,
                            validate_code_output,
                        )

                        # Add agent's input/output schemas to executor state
                        inject_schemas_to_executor_state(self.executor, agent)

                        # Populate executor.state with names from definition_code (imports and schemas)
                        if definition_code:
                            populate_definitions_to_executor_state(self.executor, definition_code)

                        # Add input variables to executor state so generated code can access them
                        self.executor.state.update(input_dict)

                        # Execute just the generated code (not definition code - that's only for LLM)
                        exec_result = await self.executor.execute(code, tools=agent.get_all_tools())

                        # Clean up schemas from state
                        clean_schemas_from_executor_state(self.executor, agent)

                        if exec_result.error:
                            error_msg = f"Candidate {candidate_idx + 1}: {exec_result.error}"
                            execution_errors.append(error_msg)
                            logger.warning(f"Candidate {candidate_idx + 1} execution failed: {exec_result.error}")

                            # Track failure in attempt context
                            # Add assistant message with the failed code
                            attempt_context.assistant(f"```python\n{code}\n```")
                            # Add error as user message (as if user reported the error)
                            attempt_context.user(f"Execution error: {exec_result.error}")

                            continue  # Try next candidate

                        # Validate output against agent's output schema
                        raw_output = exec_result.output
                        validated_output = validate_code_output(raw_output, agent.output_schema)

                        # Success! Log and break
                        logger.info(f"Candidate {candidate_idx + 1} executed successfully")
                        span.set_attribute("generation.best_cost", candidate_cost)
                        span.set_attribute("generation.num_valid", len(valid_candidates))
                        span.set_attribute("generation.execution_attempts", candidate_idx + 1)

                        # Add assistant message
                        if hasattr(validated_output, "model_dump"):
                            output_str = str(validated_output.model_dump())
                        else:
                            output_str = str(validated_output)
                        context.assistant(output_str)

                        # Compact contexts if needed
                        self.CTX = self.compact.compact(self.CTX)

                        return validated_output

                    except Exception as e:
                        execution_errors.append(f"Candidate {candidate_idx + 1}: {str(e)}")
                        logger.warning(f"Candidate {candidate_idx + 1} failed with exception: {e}")
                        # Clean up on exception
                        try:
                            clean_schemas_from_executor_state(self.executor, agent)
                        except:
                            pass
                        continue  # Try next candidate

                # All candidates failed execution
                raise RuntimeError(f"All {len(valid_candidates)} candidates failed execution: {execution_errors}")

            finally:
                self.current_agent = None

    async def execute(self, tool: Tool, **kwargs) -> Any:
        """
        Execute a tool and track in context.

        If tool is an AOT-compiled agent, appends user/assistant messages.
        If tool is a non-LLM tool, appends function call/result messages.
        If tool is an LLM tool, it handles its own context.

        Args:
            tool: Tool to execute
            **kwargs: Input arguments to the tool

        Returns:
            Output from the tool
        """
        from opentelemetry import trace

        tracer = trace.get_tracer(__name__)

        # Temporarily set this runtime as the current one so tools can access it via get_runtime()
        old_runtime = _runtime_var.get()
        _runtime_var.set(self)

        try:
            with tracer.start_as_current_span("execute") as span:
                span.set_attribute("tool.name", tool.name)

            # Get or create main context
            if "main" not in self.CTX:
                self.CTX["main"] = Context()

            context = self.CTX["main"]

            # Check if this is an LLM tool (handles its own context)
            is_llm_tool = "llm" in tool.name.lower()

            # Filter kwargs for serialization - exclude Context and other non-JSON-serializable objects
            serializable_kwargs = {}
            for k, v in kwargs.items():
                if not isinstance(v, Context):
                    try:
                        json.dumps(v)
                        serializable_kwargs[k] = v
                    except (TypeError, ValueError):
                        # Skip non-serializable values
                        pass

            tool_call_id = f"call_{tool.name}_{hashlib.sha256(json.dumps(serializable_kwargs, sort_keys=True).encode()).hexdigest()[:8]}"

            if not is_llm_tool:
                # Add function call message (assistant calling the tool)
                context.assistant(
                    content="",
                    tool_calls=[
                        {
                            "id": tool_call_id,
                            "type": "function",
                            "function": {"name": tool.name, "arguments": json.dumps(kwargs)},
                        }
                    ],
                )

            # Execute tool - just call it with kwargs, let Tool.__call__ handle validation
            try:
                result = await tool(**kwargs)

                if not is_llm_tool:
                    # Add tool result message
                    result_str = json.dumps(result.model_dump()) if hasattr(result, "model_dump") else str(result)
                    context.tool(content=result_str, name=tool.name, tool_call_id=tool_call_id)

                # Compact contexts if needed
                self.CTX = self.compact.compact(self.CTX)

                return result

            except Exception as e:
                if not is_llm_tool:
                    # Add error message
                    context.tool(content=f"Error: {str(e)}", name=tool.name, tool_call_id=tool_call_id)
                raise
        finally:
            # Restore the previous runtime
            _runtime_var.set(old_runtime)

    def _get_cache_key(self, agent: Agent) -> str:
        """Generate cache key for agent."""
        # Hash agent definition
        agent_dict = {
            "name": agent.name,
            "description": agent.description,
            "tools": [t.name for t in agent.get_all_tools()],
        }
        agent_json = json.dumps(agent_dict, sort_keys=True)
        return hashlib.sha256(agent_json.encode()).hexdigest()[:16]

    def _code_to_tool(self, agent: Agent, generated_code: str) -> Tool:
        """
        Convert compiled code to a Tool.

        The generated_code is JUST the logic (function or executable statements).
        Tools will be provided by executor at runtime.
        """

        async def execute(**kwargs):
            # Extract context parameter if provided (not part of input schema)
            context_param = kwargs.pop("context", None)

            # Validate input
            validated = agent.input_schema(**kwargs)

            # Add validated input, schemas, and context to executor state
            self.executor.state["validated"] = validated
            if context_param is not None:
                self.executor.state["_context_param"] = context_param
            if hasattr(agent.output_schema, "__name__"):
                self.executor.state[agent.output_schema.__name__] = agent.output_schema
            if hasattr(agent.input_schema, "__name__"):
                self.executor.state[agent.input_schema.__name__] = agent.input_schema

            # Check if code has a function definition or is just executable code
            import ast

            exec_code = generated_code
            try:
                from .code_utils import (
                    extract_non_stub_async_functions,
                    has_code_structure,
                    wrap_code_body_as_function,
                )

                tree = ast.parse(generated_code)
                func_defs = extract_non_stub_async_functions(generated_code)

                if func_defs:
                    # Code has a main function definition - append a call to it
                    func_name = func_defs[0][0]
                    exec_code = generated_code + f"\n\noutput = await {func_name}(**validated.model_dump())"
                elif (
                    has_code_structure(generated_code, "while_loop")
                    or has_code_structure(generated_code, "for_loop")
                    or has_code_structure(generated_code, "if_statement")
                ):
                    # Code is executable statements (like IsLoop template) - use as-is
                    # It should set 'output' variable itself
                    exec_code = generated_code
                else:
                    # Code might be just a function body - wrap it
                    exec_code = wrap_code_body_as_function(
                        generated_code,
                        agent.name,
                        agent.input_schema,
                        output_schema_name=agent.output_schema.__name__
                        if hasattr(agent.output_schema, "__name__")
                        else "Output",
                    )
                    exec_code += f"\n\noutput = await {agent.name}(**validated.model_dump())"

            except SyntaxError:
                # If parsing fails, use code as-is
                pass

            # Execute with all tools available
            result = await self.executor.execute(exec_code, tools=agent.get_all_tools())

            # Clean up
            self.executor.state.pop("validated", None)
            self.executor.state.pop("_context_param", None)
            if hasattr(agent.output_schema, "__name__"):
                self.executor.state.pop(agent.output_schema.__name__, None)
            if hasattr(agent.input_schema, "__name__"):
                self.executor.state.pop(agent.input_schema.__name__, None)

            if result.error:
                raise RuntimeError(f"Execution error: {result.error}")

            return result.output

        return Tool(
            name=agent.name,
            description=agent.description,
            input_schema=agent.input_schema,
            output_schema=agent.output_schema,
            execute=execute,
            is_terminal=False,
        )

    def _generate_loop_template(self, agent: Agent) -> str:
        """
        Generate templated agentic loop code.

        The LLM tool now handles the agentic loop internally:
        - It loops until a terminal tool (Done) is called
        - When output_schema is provided, the terminal tool result is validated against it
        - Returns the terminal tool result (matching output_schema) or response_content (string)

        This template just provides context setup and calls the LLM once.
        The LLM handles all looping and tool calling internally.
        """
        # Find LLM tool
        llm_tool = None
        llm_tools = []
        for tool in agent.get_all_tools():
            if "llm" in tool.name.lower():
                llm_tools.append(tool)
                if llm_tool is None:
                    llm_tool = tool

        if not llm_tool:
            raise RuntimeError("No LLM tool found for loop template")

        # Generate short name for LLM tool using the same mapping as codegen
        from .codegen import generate_tool_names
        tool_name_map = generate_tool_names(llm_tools)
        llm_short_name = tool_name_map.get(llm_tool.name, llm_tool.name)

        # Get the output schema class name (e.g., "AgentOutput")
        output_schema_name = agent.output_schema.__name__ if hasattr(agent.output_schema, "__name__") else "Output"

        # Build list of non-LLM tool names for the generated code
        tool_names = ", ".join(t.name for t in agent.get_all_tools() if "llm" not in t.name.lower())
        available_tools_line = f"[{tool_names}]" if tool_names else "[]"

        # Template: call LLM with output_schema (LLM handles loop internally)
        code = f"""# Agentic loop for {agent.name}
# Get context - use provided context parameter or default to "main"
try:
    context = _context_param
except NameError:
    try:
        context = get_context("main")
    except:
        context = no_context()

# Build instruction
input_str = str(validated.model_dump() if hasattr(validated, 'model_dump') else validated)
instruction = f"Complete this task: {{input_str}}. When done, call the 'done' tool with the final result."

# Call LLM with output_schema
# The LLM tool loops internally until a terminal tool is called
available_tools = {available_tools_line}
output = await {llm_short_name}(
    content=instruction,
    tools=available_tools,
    context=context,
    output_schema={output_schema_name}
)

# output is now either:
# - An instance of {output_schema_name} (if terminal tool was called and result matched schema)
# - A string (response_content if no terminal tool or couldn't parse)
# Validate it's the right type
if not isinstance(output, {output_schema_name}):
    raise RuntimeError(f"LLM did not return {output_schema_name} instance, got {{type(output).__name__}}")
"""
        return code.strip()

    def get_full_context(self, labels: str | list[str] | None = None) -> list[Message]:
        """
        Get all messages from specified contexts, sorted by timestamp with deduplication.

        Args:
            labels: Context label(s) to include. Can be:
                    - None: Include all contexts
                    - str: Single label (e.g., "main", "attempt", "intermediate")
                    - List[str]: Multiple labels (e.g., ["main", "intermediate"])

        Returns:
            List of messages sorted by timestamp, deduplicated by message_id

        Examples:
            >>> runtime.get_full_context()  # All messages from all contexts
            >>> runtime.get_full_context("main")  # Only main context
            >>> runtime.get_full_context(["main", "intermediate"])  # Main + intermediate
        """

        # Normalize labels to a set
        if labels is None:
            # Include all contexts
            context_keys = list(self.CTX.keys())
        elif isinstance(labels, str):
            # Single label - find all contexts starting with this label
            context_keys = [k for k in self.CTX.keys() if k == labels or k.startswith(f"{labels}_")]
        else:
            # Multiple labels - find all contexts starting with any of these labels
            context_keys = []
            for label in labels:
                context_keys.extend([k for k in self.CTX.keys() if k == label or k.startswith(f"{label}_")])

        # Collect all messages
        all_messages = []
        for key in context_keys:
            if key in self.CTX:
                all_messages.extend(self.CTX[key].messages)

        # Deduplicate by message_id (keeps first occurrence)
        seen_ids = set()
        unique_messages = []
        for msg in all_messages:
            if msg.message_id not in seen_ids:
                seen_ids.add(msg.message_id)
                unique_messages.append(msg)

        # Sort by timestamp
        unique_messages.sort(key=lambda m: m.timestamp)

        return unique_messages


# Global runtime management


def get_runtime() -> Runtime:
    """Get the current global runtime."""
    runtime = _runtime_var.get()
    if runtime is None:
        # Create default runtime
        runtime = Runtime()
        _runtime_var.set(runtime)
    return runtime


def set_runtime(runtime: Runtime):
    """Set the global runtime."""
    _runtime_var.set(runtime)


def set_strategy(strategy: "Strategy"):
    """
    Set the strategy for the current global runtime.

    Args:
        strategy: Strategy to apply to the current runtime
    """
    runtime = get_runtime()

    # Update runtime's strategy
    runtime.strategy = strategy

    # Update runtime's generation/verification/cost if specified in strategy
    if strategy.generate is not None:
        runtime.generate = strategy.generate
    if strategy.verify is not None:
        runtime.verify = strategy.verify if isinstance(strategy.verify, list) else [strategy.verify]
    if strategy.cost is not None:
        runtime.cost = strategy.cost
    if strategy.compact is not None:
        runtime.compact = strategy.compact


def get_context(key: str = "main"):
    """
    Get or create a context by key.

    Args:
        key: Context key (default: "main")

    Returns:
        Context object
    """
    from .context import Context

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


def new_context(label: str = "intermediate", branch_from: Optional["Context"] = None):
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
    from .context import Context

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
    existing_keys = [k for k in runtime.CTX.keys() if k.startswith(f"{label}_")]
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


__all__ = [
    "Runtime",
    "get_runtime",
    "set_runtime",
    "set_strategy",
    "get_context",
    "new_context",
]
