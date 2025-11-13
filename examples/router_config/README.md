# Router Configuration Example

## Problem

Building AI agents that handle multiple commands or intents requires:
- Loading and validating complex configuration schemas
- Managing large enum types with many possible values
- Keeping prompts concise when dealing with dozens or hundreds of options
- Ensuring type safety across configurations

Traditional approaches either hardcode everything or use fragile dynamic loading with poor type safety.

## Why Agent Compilers?

Agent compilers solve configuration challenges through:
- **Compile-time optimization**: Schemas are analyzed before execution
- **Automatic enum reduction**: Large enums are intelligently filtered to relevant options
- **JSON schema integration**: External configs are validated and typed automatically
- **Strategy-based filtering**: Semantic search finds the most relevant subset of options

## Solution with A1

This example demonstrates a router agent that:

1. **Loads command definitions from JSON** - Router commands with descriptions, categories, and options are defined in `router_schema.json`
2. **Reduces complexity at compile-time** - A1's `ReduceAndGenerate` strategy uses semantic filtering to show only the 10 most relevant commands to the LLM
3. **Validates inputs automatically** - Pydantic Field constraints ensure router IDs match patterns and values are within bounds
4. **Handles dynamic schemas** - External JSON schemas are loaded and converted to Pydantic models

**Key features shown**:
- External JSON schema loading
- Compile-time enum reduction (50+ commands → 10 most relevant)
- Field validation with regex patterns (`ge`, `le`, `pattern`)
- Strategy composition (`ReduceAndGenerate` + `CheckOrdering`)

**Key takeaway**: A1 makes it trivial to build configuration-heavy agents that would otherwise require custom schema loaders, manual validation, and careful prompt engineering to avoid context overflow.
