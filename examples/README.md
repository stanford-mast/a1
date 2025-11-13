# Examples

## simple_agent.py

Basic agent example demonstrating core a1 concepts: defining tools, creating an agent with input/output schemas, and using both AOT (ahead-of-time compilation) and JIT (just-in-time execution) modes.

## router_config.py

Advanced example showing how to handle complex configuration generation with many tools, large enum parameters, and dependency constraints. Demonstrates `ReduceAndGenerate` strategy for semantic filtering of tools/enums and `CheckOrdering` verification for enforcing command dependencies. Useful pattern for any domain with 50+ tools or large parameter spaces with structural constraints.

## readme_example.py

Simple example used in the main README showing basic agent creation and execution patterns.
