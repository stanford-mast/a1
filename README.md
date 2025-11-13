<div align="center">
  <img src="docs/assets/blast_icon_only.sketchpad.png" width="200" height="200" alt="BLAST Logo">
</div>

<p align="center" style="font-size: 24px">The agent <i>compiler</i> framework</p>

<div align="center">

[![Documentation](https://img.shields.io/badge/Docs-FFE067)](https://docs.a1project.org)
[![Discord](https://img.shields.io/badge/Discord-FFE067)](https://discord.gg/NqrkJwYYh4)
[![Twitter Follow](https://img.shields.io/twitter/follow/realcalebwin?style=social)](https://x.com/realcalebwin)

</div>

A1 is a new kind of agent framework. It takes an `Agent` (a set of tools and a description) and compiles either AOT (ahead-of-time) into a `Tool` or JIT (just-in-time) for immediate execution optimized for each unique agent input.

```bash
uv pip install a1-compiler
# or
pip install a1-compiler
```

## 🏎️ Why use an agent compiler?

An agent compiler is a direct replacement for agent frameworks such as Langchain or aisdk, where you define an `Agent` and run. The diference is:

1. **Safety:** A1 generates code for each unique agent input, optimizing constantly to shrink the prompt injection attack surface. 
2. **Speed:** A1 makes codegen practical for tool-wielding agents with aggressive parallelism and static checking.
3. **Determinism:** A1 optimizes for determinism via an engineered cost function. For example, it may replace an LLM call with a fast RegEx but may revert on-the-fly if a tool's schema evolves.
4. **Flexibility** A tool in A1 can be instantly constructed from an OpenAPI document, an MCP server, a DB connection string, an fsspec path, a Python function, a Python package, or even just a documentation website URL.

Agent compilers emerged from frustration with the MCP protocol and SOTA agent frameworks where every agent runs a static while loop program. Slow, unsafe, and highly nondeterministic. 

An agent compiler can perform the same while loop (just set `Verify=IsLoop()`) but has the freedom to explore superoptimal execution plans, while subject to engineered constraints (e.g. type-safety).

Ultimately the goal is "determinism-maxing": specifying as much of your task as fully deterministic code (100% accuracy) and gradually reducing non-deterministic LLM calls to the bare minimum.

## 🚀 How to get started?

```python
from a1 import Agent, tool, LLM
from pydantic import BaseModel

# Define a simple tool
@tool(name="add", description="Add two numbers")
async def add(a: int, b: int) -> int:
    return a + b

# Define input/output schemas
class MathInput(BaseModel):
    problem: str

class MathOutput(BaseModel):
    answer: int

# Create an agent with tools and LLM
agent = Agent(
    name="math_agent",
    description="Solves simple math problems",
    input_schema=MathInput, # like DSPy modules, A1 agent behavior is specified via schemas. The difference is that in A1, an engineer may implement a Verify function to enforce agent-specific constraints such as order of tool calling.
    output_schema=MathOutput,
    tools=[add, LLM(model="gpt-4.1")],  # in A1, LLMs are tools!
)

async def main():
    # Compile ahead-of-time
    compiled = await agent.aot()
    result = await compiled.execute(problem="What is 2 + 2?")
    print(f"AOT result: {result}")

    # Or execute just-in-time
    result = await agent.jit(problem="What is 5 + 3?")
    print(f"JIT result: {result}")

import asyncio
asyncio.run(main())
```

See the `tests/` directory for extensive examples of everything A1 can do. Docs coming soon to [docs.a1project.org](https://docs.a1project.org)

## 📚 Examples

### Router Configuration from JSON

The `examples/router_config.py` demonstrates how to dynamically create tools from JSON schemas. This is useful when you have many commands (e.g., router CLI, database operations) defined in a configuration file:

```python
# Load command schemas from JSON
with open("router_schema.json") as f:
    commands = json.load(f)["commands"]

# Create Tool objects dynamically
tools = []
for cmd_name, cmd_config in commands.items():
    # Convert JSON schema to Pydantic model
    InputModel = json_schema_to_pydantic(cmd_name, cmd_config["schema"])
    
    # Create Tool with schema
    tool = Tool(
        name=cmd_name,
        description=cmd_config["description"],
        input_schema=InputModel,
        output_schema=CommandResult,
        execute=lambda **kwargs: {...}
    )
    tools.append(tool)

# Create agent with dynamic tools
agent = Agent(
    name="router_agent",
    description="Configure Cisco router",
    tools=tools + [LLM("gpt-4.1-mini"), Done()]
)
```

Run the full example:
```bash
export OPENAI_API_KEY=your_key_here
uv run python examples/router_config.py
```

This approach scales to thousands of commands while preserving Field validation (regex patterns, numeric bounds, etc.) from the JSON schema.

## ✨ Features

* **Import** any Langchain agent
* **Observability** via OpenTelemetry
* **Tools** instantiated from MCP or OpenAPI
* **RAG** instantiated given any SQL database or fsspec path (e.g. `s3://my-place/here`, `gs://...`, or local filesystem)
* **Skills** defined manually or by crawling online docs
* **Context engineering** via a simple API that lets compiled code manage multi-agent behavior
* **Zero lock-in** use any LLM, any secure code execution cloud
* Only gets better as researchers develop increasingly powerful methods to `Generate`, `Cost` estimate, and `Verify` agent code

## 🙋 FAQ

#### Should I use A1 or Langchain/aisdk/etc?
Prefer A1 if your task is latency-critical, works with untrusted data, or may need to run code.

#### Is A1 production-ready?
Yes in terms of API stability. The caveat is that A1 is new.

#### Can we get enterprise support?
Please don't hesitate to reach out (calebwin@stanford.edu)

## 🤝 Contributing

Awesome! See our [Contributing Guide](/CONTRIBUTING.md) for details.

## 📄 MIT License

As it should be!

## 📜 Citation

Paper coming soon! Reach out if you'd like to contribute.