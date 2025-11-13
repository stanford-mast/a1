# A1 Examples

This directory contains example agents demonstrating different capabilities of the A1 agent compiler. Each example is self-contained with its own README explaining the problem, why agent compilers help, and how A1 solves it.

## Quick Start

All examples can be run directly with:
```bash
# Install dependencies
uv sync --extra examples

# Run an example
python examples/<example_name>/<example_name>.py
```

## Examples

### [Simple Agent](./simple_agent/)
**The basics of A1** - A minimal conversational agent showing core concepts.

**What you'll learn:**
- How to define agents with input/output schemas
- Basic LLM tool usage
- Context management (automatic!)
- Running agents with `Runtime.jit()`

**Run it:**
```bash
python examples/simple_agent/simple_agent.py
```

---

### [Router Configuration](./router_config/)
**Advanced configuration management** - Loading complex schemas, handling large enums, and compile-time optimization.

**What you'll learn:**
- Loading JSON schemas as Pydantic models
- Compile-time enum reduction with `ReduceAndGenerate` strategy
- Field validation (regex patterns, bounds checking)
- Semantic filtering to reduce context size
- Strategy composition

**Run it:**
```bash
python examples/router_config/router_config.py
```

---

### [Z3 Logical Reasoning](./z3_reasoning/)
**Hybrid AI reasoning** - Combining LLMs with formal theorem provers for provably correct answers.

**What you'll learn:**
- Using Skills to teach domain knowledge (Z3 API)
- Code generation for formal verification  
- Integration with external libraries (z3-solver)
- StrategyQA-style reasoning problems
- Achieving mathematical certainty vs. statistical guessing

**Run it:**
```bash
python examples/z3_reasoning/z3_reasoning.py
```

**Inspired by:** [Proof of Thought paper](https://news.ycombinator.com/item?id=45475529) - Using LLMs to generate formal proofs verified by SMT solvers.

---

## Example Structure

Each example directory contains:
- **`README.md`** - Problem description, motivation, and A1 solution
- **`<name>.py`** - Runnable Python code demonstrating the concept
- **Additional files** - JSON schemas, configs, or test files as needed

## Dependencies

Core examples work with just:
```bash
pip install a1-compiler
```

Some examples require additional packages:
- **z3_reasoning**: `pip install z3-solver` (or `uv sync --extra examples`)

## Contributing

Have a cool example? We'd love to see it! Each example should:

1. **Solve a real problem** - Not just a toy demo
2. **Show A1's value** - Highlight what makes agent compilers useful
3. **Be self-contained** - Include all necessary files
4. **Include a README** - Following the Problem/Why/Solution format
5. **Be runnable** - Test it works before submitting!

See existing examples for the structure and style.
