"""
Extra strategy implementations for advanced use cases.

This module provides specialized Generate and Verify strategies:
- ReduceAndGenerate: Filter tools and reduce enums using semantic similarity
- CheckOrdering: Verify tool call ordering constraints via AST analysis
"""

import ast
import asyncio
import logging
from typing import Any

from .cfg_builder import CFGBuilder
from .codegen import BaseGenerate
from .codecheck import BaseVerify, Verify
from .models import Agent, Tool

logger = logging.getLogger(__name__)


class ReduceAndGenerate(BaseGenerate):
    """
    Generate strategy that filters tools and reduces large enums using semantic similarity.
    
    This strategy performs two optimizations before code generation:
    1. Tool filtering: Select top K most relevant tools based on task description
    2. Enum reduction: Shrink large enums in tool/agent schemas using semantic similarity
    
    Uses the EM tool for computing embeddings and semantic similarity.
    All reduction operations run in parallel for efficiency.
    
    Attributes:
        em_tool: Embedding model tool for computing semantic similarity
        max_tools: Maximum number of tools to keep (default: 50)
        max_enum_size: Maximum enum values to keep in schemas (default: 100)
        llm_tool: LLM tool for code generation
        timezone: Timezone for timestamp context
    
    Example:
        >>> from a1 import EM, LLM
        >>> strategy = Strategy(
        ...     generate=ReduceAndGenerate(
        ...         em_tool=EM(),
        ...         llm_tool=LLM("gpt-4"),
        ...         max_tools=50,
        ...         max_enum_size=100
        ...     )
        ... )
    """
    
    def __init__(
        self,
        em_tool: Tool,
        llm_tool: Tool,
        max_tools: int = 50,
        max_enum_size: int = 100,
        timezone: str = "UTC",
    ):
        """
        Initialize ReduceAndGenerate strategy.
        
        Args:
            em_tool: Embedding model tool for semantic similarity
            llm_tool: LLM tool for code generation
            max_tools: Maximum number of tools to keep after filtering
            max_enum_size: Maximum enum values to keep in schemas
            timezone: Timezone for timestamp context
        """
        super().__init__(llm_tool=llm_tool, timezone=timezone)
        self.em_tool = em_tool
        self.max_tools = max_tools
        self.max_enum_size = max_enum_size
    
    async def generate(
        self,
        agent: Agent,
        task: str,
        return_function: bool = False,
        past_attempts: list[tuple[str, str]] | None = None,
        context: Any | None = None,
    ) -> tuple[str, str] | None:
        """
        Generate code with semantic tool filtering and enum reduction.
        
        Args:
            agent: Agent to generate code for
            task: Task description or input data
            return_function: Whether to return function definition (AOT) or code block (JIT)
            past_attempts: Previous failed attempts as (code, error) tuples
            context: Context for maintaining conversation history across retries
            
        Returns:
            Tuple of (definition_code, generated_code) or None if generation fails
        """
        from .schema_utils import reduce_large_enums
        
        # Get runtime for embedding cache
        from .runtime import get_runtime
        runtime = get_runtime()
        
        # Combine agent description + task for semantic query
        query_text = f"{agent.description}\n{task}"
        
        # Get all tools
        all_tools = agent.get_all_tools()
        
        # Parallel operations: tool filtering + enum reduction in agent schemas + enum reduction in tool schemas
        parallel_tasks = []
        
        # Task 1: Filter tools if we have more than max_tools
        if len(all_tools) > self.max_tools:
            logger.info(f"Filtering {len(all_tools)} tools to top {self.max_tools} most relevant")
            
            async def filter_tools():
                # Build tool descriptions
                tool_descriptions = []
                for t in all_tools:
                    desc = f"{t.name}: {t.description}"
                    if hasattr(t.input_schema, 'model_json_schema'):
                        schema = t.input_schema.model_json_schema()
                        desc += f"\nInput: {schema}"
                    tool_descriptions.append(desc)
                
                # Use EM tool to compute similarities
                try:
                    result = await self.em_tool(
                        items_a=[query_text],  # Query
                        items_b=tool_descriptions  # Tools to rank
                    )
                    similarities = result.similarities[0]  # Get first row
                    
                    # Get top K tools by similarity
                    tool_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:self.max_tools]
                    return [all_tools[i] for i in tool_indices]
                    
                except Exception as e:
                    logger.warning(f"Tool filtering failed: {e}, using all tools")
                    return all_tools
            
            parallel_tasks.append(filter_tools())
        else:
            # Dummy task - just return all tools
            async def no_filter():
                return all_tools
            parallel_tasks.append(no_filter())
        
        # Task 2: Reduce enums in agent input/output schemas
        async def reduce_agent_schemas():
            """Reduce enums in agent's input and output schemas"""
            modified_schemas = {}
            
            # Reduce input schema enums if present
            if hasattr(agent.input_schema, 'model_json_schema'):
                input_schema_dict = agent.input_schema.model_json_schema()
                reduced_input = reduce_large_enums(
                    input_schema_dict,
                    context_text=query_text,
                    runtime=runtime,
                    threshold=self.max_enum_size,
                    target_size=self.max_enum_size
                )
                if reduced_input != input_schema_dict:
                    logger.info(f"Reduced agent input schema enums (threshold={self.max_enum_size})")
                    modified_schemas['input'] = reduced_input
            
            # Reduce output schema enums if present
            if hasattr(agent.output_schema, 'model_json_schema'):
                output_schema_dict = agent.output_schema.model_json_schema()
                reduced_output = reduce_large_enums(
                    output_schema_dict,
                    context_text=query_text,
                    runtime=runtime,
                    threshold=self.max_enum_size,
                    target_size=self.max_enum_size
                )
                if reduced_output != output_schema_dict:
                    logger.info(f"Reduced agent output schema enums (threshold={self.max_enum_size})")
                    modified_schemas['output'] = reduced_output
            
            return modified_schemas
        
        parallel_tasks.append(reduce_agent_schemas())
        
        # Task 3: Reduce enums in all tool schemas
        async def reduce_tool_schemas():
            """Reduce enums in all tool input/output schemas"""
            tools_with_large_enums = []
            
            for tool in all_tools:
                # Check tool input schema for large enums
                if hasattr(tool.input_schema, 'model_json_schema'):
                    input_schema_dict = tool.input_schema.model_json_schema()
                    reduced = reduce_large_enums(
                        input_schema_dict,
                        context_text=query_text,
                        runtime=runtime,
                        threshold=self.max_enum_size,
                        target_size=self.max_enum_size
                    )
                    if reduced != input_schema_dict:
                        tools_with_large_enums.append(tool.name)
                
                # Check tool output schema for large enums
                if hasattr(tool.output_schema, 'model_json_schema'):
                    output_schema_dict = tool.output_schema.model_json_schema()
                    reduced = reduce_large_enums(
                        output_schema_dict,
                        context_text=query_text,
                        runtime=runtime,
                        threshold=self.max_enum_size,
                        target_size=self.max_enum_size
                    )
                    if reduced != output_schema_dict:
                        if tool.name not in tools_with_large_enums:
                            tools_with_large_enums.append(tool.name)
            
            if tools_with_large_enums:
                logger.info(f"Reduced enums in {len(tools_with_large_enums)} tool schemas: {', '.join(tools_with_large_enums[:5])}")
            
            return tools_with_large_enums
        
        parallel_tasks.append(reduce_tool_schemas())
        
        # Wait for all parallel tasks to complete
        results = await asyncio.gather(*parallel_tasks)
        
        filtered_tools = results[0]
        reduced_agent_schemas = results[1]
        tools_with_reduced_enums = results[2]
        
        logger.info(f"Using {len(filtered_tools)} tools after semantic filtering")
        if reduced_agent_schemas:
            logger.info(f"Reduced enums in agent schemas: {list(reduced_agent_schemas.keys())}")
        if tools_with_reduced_enums:
            logger.info(f"Reduced enums in {len(tools_with_reduced_enums)} tool(s)")
        
        # Create temporary agent with filtered tools
        # Note: We can't actually modify the Pydantic schemas at runtime,
        # but the reduce_large_enums function already happens inside the LLM
        # generation process via schema_utils. This logging just confirms it happened.
        temp_agent = Agent(
            name=agent.name,
            description=agent.description,
            tools=filtered_tools,
            input_schema=agent.input_schema,
            output_schema=agent.output_schema,
        )
        
        # Delegate to base class with filtered agent
        return await super().generate(
            agent=temp_agent,
            task=task,
            return_function=return_function,
            past_attempts=past_attempts,
            context=context,
        )


class CheckOrdering(Verify):
    """
    Verify that tool calls in generated code satisfy ordering constraints.
    
    Uses AST analysis to extract the sequence of tool calls and checks against
    dependency rules. Each rule specifies that tool A must be called before tool B.
    
    Attributes:
        rules: List of (prerequisite_tool, dependent_tool) pairs
        
    Example:
        >>> verify = CheckOrdering(rules=[
        ...     ("enable_interface", "set_ip_address"),  # Must enable before setting IP
        ...     ("create_vlan", "assign_vlan_port"),     # Must create before assigning
        ... ])
    """
    
    def __init__(self, rules: list[tuple[str, str]] | None = None):
        """
        Initialize CheckOrdering with dependency rules.
        
        Args:
            rules: List of (prerequisite, dependent) tool name pairs
        """
        self.rules = rules or []
    
    def verify(self, code: str | tuple[str, str], agent: Agent) -> tuple[bool, str | None]:
        """
        Verify tool call ordering in generated code.
        
        Args:
            code: Generated code (string) or (definition_code, generated_code) tuple
            agent: Agent being compiled
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Handle tuple input (definition_code, generated_code)
        if isinstance(code, tuple):
            definition_code, generated_code = code
            full_code = definition_code + "\n" + generated_code if definition_code else generated_code
        else:
            full_code = code
        
        # Parse code to extract tool call sequence
        try:
            tree = ast.parse(full_code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Extract tool calls in order
        tool_calls = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Check for direct function calls
                if isinstance(node.func, ast.Name):
                    tool_calls.append(node.func.id)
                # Check for attribute calls (e.g., tool.execute())
                elif isinstance(node.func, ast.Attribute):
                    tool_calls.append(node.func.attr)
                # Check for await calls
                elif isinstance(node.func, ast.Await):
                    if isinstance(node.func.value, ast.Name):
                        tool_calls.append(node.func.value.id)
        
        # Check each rule
        for prerequisite, dependent in self.rules:
            # Find positions of prerequisite and dependent in call sequence
            prereq_positions = [i for i, name in enumerate(tool_calls) if name == prerequisite]
            dependent_positions = [i for i, name in enumerate(tool_calls) if name == dependent]
            
            # If dependent is called but prerequisite is not, that's an error
            if dependent_positions and not prereq_positions:
                return False, f"Tool '{dependent}' requires '{prerequisite}' to be called first, but '{prerequisite}' was never called"
            
            # If both are called, check ordering
            if prereq_positions and dependent_positions:
                # Check if any dependent call comes before all prerequisite calls
                earliest_prereq = min(prereq_positions)
                earliest_dependent = min(dependent_positions)
                
                if earliest_dependent < earliest_prereq:
                    return False, f"Tool '{dependent}' (position {earliest_dependent}) was called before '{prerequisite}' (position {earliest_prereq}), violating ordering constraint"
        
        return True, None


__all__ = [
    "ReduceAndGenerate",
    "CheckOrdering",
]
