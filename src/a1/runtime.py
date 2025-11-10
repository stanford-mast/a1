"""
Runtime for executing agents with code generation and execution.

Provides:
- Runtime: Configurable runtime with aot, jit, execute methods
- Global runtime management: get_runtime, set_runtime
- Context manager support
"""

import logging
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from contextvars import ContextVar

from .context import Context, BaseCompact
from .executor import BaseExecutor
from .strategies import BaseGenerate, BaseVerify, BaseCost
from .models import Agent, Tool, Strategy

if TYPE_CHECKING:
    from .strategies import Generate, Verify, Cost, Compact
    from .executor import Executor

logger = logging.getLogger(__name__)

# Global runtime context variable
_runtime_var: ContextVar[Optional['Runtime']] = ContextVar('runtime', default=None)


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
        generate: Optional['Generate'] = None,
        verify: Optional[List['Verify']] = None,
        cost: Optional['Cost'] = None,
        compact: Optional['Compact'] = None,
        executor: Optional['Executor'] = None,
        cache_dir: str = ".a1"
    ):
        from .builtin_tools import LLM
        from .llm import no_context
        from pydantic import BaseModel, Field
        from typing import Optional, List, Dict, Any, Union
        
        self.generate = generate or BaseGenerate(llm_tool=LLM("groq:openai/gpt-oss-20b"))
        self.verify = verify or [BaseVerify()]
        self.cost = cost or BaseCost()
        self.compact = compact or BaseCompact()
        
        # Create executor with necessary imports for generated code
        if executor is None:
            executor = BaseExecutor(
                additional_imports={
                    'get_context': get_context,
                    'no_context': no_context,
                    'BaseModel': BaseModel,
                    'Field': Field,
                    'Optional': Optional,
                    'List': List,
                    'Dict': Dict,
                    'Any': Any,
                    'Union': Union,
                }
            )
        self.executor = executor
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Contexts: Dictionary of named message lists
        self.CTX: Dict[str, Context] = {}
        
        # Current agent being executed (for tool calling)
        self.current_agent: Optional[Agent] = None
    
    def __enter__(self):
        """Enter context manager - set as global runtime."""
        self._token = _runtime_var.set(self)
        return self
    
    def __exit__(self, *args):
        """Exit context manager - restore previous runtime."""
        _runtime_var.reset(self._token)
    
    async def aot(
        self,
        agent: Agent,
        cache: bool = True,
        strategy: Optional['Strategy'] = None
    ) -> Tool:
        """
        Ahead-of-time compile an agent to a tool.
        
        Generates Python function code for the agent and caches it. Returns a Tool
        that executes the compiled code.
        
        If IsLoop verifier is present, uses a templated loop instead of LLM generation.
        
        Args:
            agent: Agent to compile
            cache: Whether to use cached compilation (default: True)
            strategy: Optional Strategy for generation config (default: Strategy())
        
        Returns:
            Tool that executes the compiled agent
        """
        from .models import Strategy
        from .strategies import IsLoop, IsFunction
        from opentelemetry import trace
        import asyncio
        
        # Use default strategy if not provided
        if strategy is None:
            strategy = Strategy()
        
        num_candidates = strategy.num_candidates
        max_retries = strategy.max_iterations
        
        tracer = trace.get_tracer(__name__)
        
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
            is_loop_verifiers = [v for v in self.verify if isinstance(v, IsLoop)]
            if is_loop_verifiers:
                # Use templated loop instead of generation
                logger.info(f"Using templated loop for {agent.name}")
                generated_code = self._generate_loop_template(agent)
                logger.info(f"Generated template code:\n{generated_code}")
                span.set_attribute("generation.type", "template")
                
                # For template verification, we just verify the generated code itself
                # (no separate definition_code needed for template)
                for verifier in self.verify:
                    is_valid, error = verifier.verify(generated_code, agent=agent)
                    if not is_valid:
                        raise RuntimeError(f"Template verification failed: {error}")
            else:
                # Generate code using LLM with parallel candidates
                logger.info(f"Generating {num_candidates} function candidates for {agent.name}")
                span.set_attribute("generation.type", "llm")
                
                # Generate candidates in parallel with retries
                async def generate_with_retries():
                    past_attempts = []
                    for attempt in range(max_retries):
                        result = await self.generate.generate(
                            agent=agent,
                            task=agent.description,
                            return_function=True,  # AOT generates functions
                            past_attempts=past_attempts if attempt > 0 else None
                        )
                        
                        if not result:
                            continue
                        
                        # Generate returns (definition_code, generated_code) tuple
                        definition_code, generated_code = result
                        
                        # Fix the generated code (wrap if needed for AOT)
                        from .code_utils import fix_generated_code
                        fixed_code = fix_generated_code(generated_code, is_aot=True, function_name=agent.name, agent=agent)
                        
                        # Concatenate definitions with fixed code for validation
                        full_code_for_verification = definition_code + "\n" + fixed_code if definition_code else fixed_code
                        
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
                            for verifier in self.verify:
                                is_valid, error = verifier.verify((definition_code, fixed_code), agent=agent)
                                if not is_valid:
                                    all_valid = False
                                    validation_error = error
                                    break
                        
                        if all_valid:
                            # Compute cost using the strategy's cost estimator
                            cost = self.cost.compute_cost((definition_code, fixed_code), agent=agent)
                            return (fixed_code, cost, None)
                        else:
                            # Track attempt for retry
                            past_attempts.append((generated_code, validation_error))
                    
                    # All retries failed
                    if past_attempts:
                        return (past_attempts[-1][0], float('inf'), past_attempts[-1][1])
                    return (None, float('inf'), "Failed to generate code")
                
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
    
    async def jit(
        self,
        agent: Agent,
        strategy: Optional[Strategy] = None,
        **kwargs
    ) -> Any:
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
        from .codecost import compute_code_cost
        from .strategies import IsLoop
        from opentelemetry import trace
        import asyncio
        
        # Validate input against agent's input schema
        validated_input = agent.input_schema(**kwargs)
        input_dict = validated_input.model_dump()
        
        # Use default strategy if not provided
        if strategy is None:
            strategy = Strategy()
        
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
            
            # Add user message with input
            context.user(json.dumps(input_dict))
            
            # Set current agent for tool execution
            self.current_agent = agent
            
            try:
                # Generate code using LLM with parallel candidates
                logger.info(f"Generating {num_candidates} code candidates for {agent.name}")
                
                # Generate candidates in parallel with retries
                async def generate_with_retries():
                    past_attempts = []
                    for attempt in range(max_retries):
                        result = await self.generate.generate(
                            agent=agent,
                            task=json.dumps(input_dict),
                            return_function=False,  # JIT generates code blocks
                            past_attempts=past_attempts if attempt > 0 else None
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
                        for verifier in self.verify:
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
                            cost = self.cost.compute_cost((definition_code, fixed_code), agent=agent)
                            return (fixed_code, definition_code, cost, None)
                        else:
                            # Track attempt for retry
                            past_attempts.append((generated_code, validation_error))
                    
                    # All retries failed
                    if past_attempts:
                        return (past_attempts[-1][0], "", float('inf'), past_attempts[-1][1])
                    return (None, "", float('inf'), "Failed to generate code")
                
                # Launch parallel generation tasks
                tasks = [generate_with_retries() for _ in range(num_candidates)]
                results = await asyncio.gather(*tasks)
                
                # Filter valid candidates and rank by cost
                valid_candidates = [(code, def_code, cost) for code, def_code, cost, error in results if error is None]
                
                if not valid_candidates:
                    # All candidates failed - report errors
                    errors = [error for _, _, _, error in results if error]
                    raise RuntimeError(f"All candidates failed validation: {errors}")
                
                # Select best candidate by cost
                code, definition_code, best_cost = min(valid_candidates, key=lambda x: x[2])
                logger.info(f"Selected best candidate with cost {best_cost} from {len(valid_candidates)} valid")
                span.set_attribute("generation.best_cost", best_cost)
                span.set_attribute("generation.num_valid", len(valid_candidates))
                
                # Use code_utils for schema injection
                from .code_utils import (
                    inject_schemas_to_executor_state,
                    clean_schemas_from_executor_state,
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
                    raise RuntimeError(f"Execution error: {exec_result.error}")
                
                # Validate output against agent's output schema
                raw_output = exec_result.output
                validated_output = validate_code_output(raw_output, agent.output_schema)
                
                # Add assistant message
                if hasattr(validated_output, 'model_dump'):
                    output_str = str(validated_output.model_dump())
                else:
                    output_str = str(validated_output)
                context.assistant(output_str)
                
                # Compact contexts if needed
                self.CTX = self.compact.compact(self.CTX)
                
                return validated_output
            
            finally:
                self.current_agent = None
    
    async def execute(
        self,
        tool: Tool,
        **kwargs
    ) -> Any:
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
                    tool_calls=[{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool.name,
                            "arguments": json.dumps(kwargs)
                        }
                    }]
                )
            
            # Execute tool - just call it with kwargs, let Tool.__call__ handle validation
            try:
                result = await tool(**kwargs)
                
                if not is_llm_tool:
                    # Add tool result message
                    result_str = json.dumps(result.model_dump()) if hasattr(result, 'model_dump') else str(result)
                    context.tool(
                        content=result_str,
                        name=tool.name,
                        tool_call_id=tool_call_id
                    )
                
                # Compact contexts if needed
                self.CTX = self.compact.compact(self.CTX)
                
                return result
            
            except Exception as e:
                if not is_llm_tool:
                    # Add error message
                    context.tool(
                        content=f"Error: {str(e)}",
                        name=tool.name,
                        tool_call_id=tool_call_id
                    )
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
            context_param = kwargs.pop('context', None)
            
            # Validate input
            validated = agent.input_schema(**kwargs)
            
            # Add validated input, schemas, and context to executor state
            self.executor.state['validated'] = validated
            if context_param is not None:
                self.executor.state['_context_param'] = context_param
            if hasattr(agent.output_schema, '__name__'):
                self.executor.state[agent.output_schema.__name__] = agent.output_schema
            if hasattr(agent.input_schema, '__name__'):
                self.executor.state[agent.input_schema.__name__] = agent.input_schema
            
            # Check if code has a function definition or is just executable code
            import ast
            exec_code = generated_code
            try:
                from .code_utils import (
                    extract_non_stub_async_functions,
                    has_code_structure,
                    wrap_code_body_as_function,
                    extract_execution_result,
                )
                
                tree = ast.parse(generated_code)
                func_defs = extract_non_stub_async_functions(generated_code)
                
                if func_defs:
                    # Code has a main function definition - append a call to it
                    func_name = func_defs[0][0]
                    exec_code = generated_code + f"\n\noutput = await {func_name}(**validated.model_dump())"
                elif has_code_structure(generated_code, 'while_loop') or has_code_structure(generated_code, 'for_loop') or has_code_structure(generated_code, 'if_statement'):
                    # Code is executable statements (like IsLoop template) - use as-is
                    # It should set 'output' variable itself
                    exec_code = generated_code
                else:
                    # Code might be just a function body - wrap it
                    exec_code = wrap_code_body_as_function(
                        generated_code,
                        agent.name,
                        agent.input_schema,
                        output_schema_name=agent.output_schema.__name__ if hasattr(agent.output_schema, '__name__') else "Output"
                    )
                    exec_code += f"\n\noutput = await {agent.name}(**validated.model_dump())"
                    
            except SyntaxError:
                # If parsing fails, use code as-is
                pass
            
            # Execute with all tools available
            result = await self.executor.execute(exec_code, tools=agent.get_all_tools())
            
            # Clean up
            self.executor.state.pop('validated', None)
            self.executor.state.pop('_context_param', None)
            if hasattr(agent.output_schema, '__name__'):
                self.executor.state.pop(agent.output_schema.__name__, None)
            if hasattr(agent.input_schema, '__name__'):
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
            is_terminal=False
        )
    
    def _generate_loop_template(self, agent: Agent) -> str:
        """
        Generate templated agentic loop code.
        
        The LLM tool handles function calling and auto-adds a Done terminal tool if needed.
        When output_schema is set, LLM returns a properly typed instance.
        The template just calls LLM with the agent's output schema as the output_schema.
        """
        # Find LLM tool
        llm_tool = None
        for tool in agent.get_all_tools():
            if "llm" in tool.name.lower():
                llm_tool = tool
                break
        
        if not llm_tool:
            raise RuntimeError("No LLM tool found for loop template")
        
        # Get the output schema class name (e.g., "AgentOutput")
        output_schema_name = agent.output_schema.__name__ if hasattr(agent.output_schema, '__name__') else 'Output'
        
        # Simple template: call LLM in a loop with agent's output schema as output_schema
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

# Call LLM with all non-LLM tools until we get an output
max_iterations = 20
available_tools = [{", ".join(f"{t.name}" for t in agent.get_all_tools() if "llm" not in t.name.lower())}]

iteration = 0
while iteration < max_iterations:
    output = await {llm_tool.name}(
        content=instruction if iteration == 0 else "Continue with the task.",
        tools=available_tools,
        context=context,
        output_schema={output_schema_name}
    )
    
    # If we got an output instance, we're done
    if isinstance(output, {output_schema_name}):
        break
    
    iteration += 1
else:
    # Max iterations reached - shouldn't happen with output_schema set
    raise RuntimeError("Failed to complete task in {{max_iterations}} iterations")
"""
        return code.strip()


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
        runtime.CTX[key] = Context()
    return runtime.CTX[key]


__all__ = [
    "Runtime",
    "get_runtime",
    "set_runtime",
    "get_context",
]
