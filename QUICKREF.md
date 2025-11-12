# a1 Quick Reference

## Installation

```bash
uv pip install a1
# or
pip install a1
```

## Basic Usage

### Define a Tool

```python
from a1 import Tool
from pydantic import BaseModel, Field

class CalculatorInput(BaseModel):
    a: float = Field(..., description="First number")
    b: float = Field(..., description="Second number")
    operation: str = Field(..., description="Operation: add, subtract, multiply, divide")

class CalculatorOutput(BaseModel):
    result: float = Field(..., description="Result")

async def calculator_execute(a: float, b: float, operation: str) -> dict:
    if operation == "add":
        return {"result": a + b}
    elif operation == "divide":
        return {"result": a / b}
    # ... etc

calculator = Tool(
    name="calculator",
    description="Perform basic arithmetic operations",
    input_schema=CalculatorInput,
    output_schema=CalculatorOutput,
    execute=calculator_execute,
    is_terminal=False
)
```

### Create an Agent

```python
from a1 import Agent, LLM
from pydantic import BaseModel, Field

class AgentInput(BaseModel):
    problem: str = Field(..., description="Problem to solve")

class AgentOutput(BaseModel):
    answer: str = Field(..., description="Solution")

agent = Agent(
    name="my_agent",
    description="Solves math problems",
    input_schema=AgentInput,
    output_schema=AgentOutput,
    tools=[calculator, LLM("groq:openai/gpt-oss-20b")]
)
```

### Execute Agent

```python
from a1 import Runtime

runtime = Runtime()

# AOT: Compile once, reuse
compiled = await runtime.aot(agent, cache=True)
result = await compiled(problem="What is 42 divided by 7?")

# JIT: Generate and execute on-the-fly
result = await runtime.jit(agent, problem="What is 10 times 5?")

# Direct tool execution
result = await runtime.execute(calculator, a=10, b=5, operation="add")
```

## Built-in Tools

### LLM

Supported providers:
- `"gpt-4.1"`, `"gpt-4.1-mini"` (OpenAI)
- `"claude-3-5-sonnet-20241022"` (Anthropic)
- `"groq:openai/gpt-oss-20b"` (Groq)
- `"mistral:mistral-small-latest"` (Mistral)

```python
from a1 import LLM

# Simple LLM tool
llm = LLM("gpt-4.1-mini")

# LLM with specific role
llm = LLM("claude-3-5-sonnet-20241022", role="expert")
```

### Done (Terminal Tool)

The LLM automatically adds a `Done` tool when compiling. You don't need to add it manually.

```python
# The Done tool is automatically available in generated code
# Just reference it as: Done(answer="your answer")
```

## Execution Modes

### AOT (Ahead-Of-Time) Compilation

- Generates code once, compiles it, caches it
- Fastest execution (code is pre-compiled)
- Best for: Production, repeated execution

```python
compiled = await runtime.aot(agent, cache=True)
result = await compiled(problem="What is 2 + 2?")
```

### JIT (Just-In-Time) Execution

- Generates code on-the-fly for each request
- More flexible for dynamic problems
- Best for: Development, variable inputs

```python
result = await runtime.jit(agent, problem="What is 2 + 2?")
```

### IsLoop Mode

Uses templated agentic loop instead of LLM generation for more predictable behavior.

```python
from a1 import Runtime, IsLoop

runtime = Runtime(verify=[IsLoop()])
compiled = await runtime.aot(agent)
```

## Code Verification

Automatic verification catches:
- Syntax errors
- Type mismatches (with ty)
- Dangerous operations
- Missing loop patterns (IsLoop)

```python
from a1 import Runtime, BaseVerify, IsLoop

runtime = Runtime(
    verify=[
        BaseVerify(),  # Safety checks
        IsLoop()       # Loop pattern verification
    ]
)
```

## Context Management

Contexts track conversation history and state across tool calls.

```python
from a1 import get_context, set_runtime, Runtime

runtime = Runtime()
set_runtime(runtime)

# Access or create context
ctx = get_context("main")
ctx.messages  # List of Message objects

# Create new context for different conversation
ctx_sidebar = get_context("sidebar")
```

## Messages

```python
from a1 import Message

# User message
msg = Message(role="user", content="What is 2 + 2?")

# Assistant response with tool calls
msg = Message(
    role="assistant",
    content="I'll calculate that",
    tool_calls=[...]
)

# Tool result
msg = Message(
    role="tool",
    content="The result is 4",
    name="calculator",
    tool_call_id="123"
)
```

## Custom Strategies

### Generate Strategy

Controls how agent code is generated.

```python
from a1 import Generate

class MyGenerate(Generate):
    async def generate(self, agent, task):
        # Your code generation logic
        return definition_code, generated_code
```

### Verify Strategy

Validates generated code before execution.

```python
from a1 import Verify

class MyVerify(Verify):
    def verify(self, code, agent):
        # Your validation logic
        return is_valid, error_message
```

### Cost Strategy

Ranks code candidates by estimated cost.

```python
from a1 import Cost

class MyCost(Cost):
    def compute_cost(self, code, agent):
        # Calculate cost (lower = better)
        return cost_value
```

## Environment Variables

```bash
# LLM API keys
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
export GROQ_API_KEY="..."
export MISTRAL_API_KEY="..."

# Optional: custom cache directory
export A1_CACHE_DIR=".a1"
```

## Runtime Configuration

```python
from a1 import Runtime, BaseGenerate, BaseVerify, BaseCost

runtime = Runtime(
    generate=BaseGenerate(
        llm_tool=LLM("gpt-4")
    ),
    verify=[BaseVerify()],
    cost=BaseCost(),
    compact=BaseCompact(window_size=20)
)
```

## Common Patterns

### Simple Tool Execution

```python
runtime = Runtime()
result = await runtime.execute(tool, **input_params)
```

### Multi-Tool Agent

```python
agent = Agent(
    name="worker",
    description="Uses multiple tools",
    input_schema=Input,
    output_schema=Output,
    tools=[calculator, searcher, llm]
)

compiled = await runtime.aot(agent)
```

### Agent with Context

```python
from a1 import get_context, set_runtime

runtime = Runtime()
set_runtime(runtime)

result = await runtime.jit(agent, problem="...")

# Access conversation history
ctx = get_context("main")
for msg in ctx.messages:
    print(f"{msg.role}: {msg.content}")
```

## Error Handling

```python
try:
    result = await runtime.jit(agent, input_data)
except RuntimeError as e:
    print(f"Execution failed: {e}")
except ValueError as e:
    print(f"Invalid input: {e}")
```

## Debugging

Enable logging to see execution details:

```python
import logging

logging.basicConfig(level=logging.INFO)

# Will show:
# - Code generation
# - LLM calls
# - Tool execution
# - Cache operations
# - Type checking results
```

## Best Practices

1. **Define clear schemas** - Use Pydantic models with descriptions
2. **Test tools independently** - Verify before adding to agents
3. **Use caching in production** - Set `cache=True` for `aot()`
4. **Check environment variables** - Ensure API keys are set
5. **Monitor with logging** - Track what's happening
6. **Handle errors gracefully** - Catch and log exceptions
7. **Use type hints** - Enables better schema generation

## Examples

### Math Agent

```python
from a1 import Agent, Tool, LLM, Runtime
from pydantic import BaseModel, Field

class MathInput(BaseModel):
    problem: str = Field(..., description="Math problem")

class MathOutput(BaseModel):
    answer: str = Field(..., description="Solution")

# Define calculator tool
calculator = Tool(
    name="calculator",
    description="Perform arithmetic",
    input_schema=CalculatorInput,
    output_schema=CalculatorOutput,
    execute=calculator_execute
)

# Create agent
agent = Agent(
    name="math_solver",
    description="Solves math problems",
    input_schema=MathInput,
    output_schema=MathOutput,
    tools=[calculator, LLM("gpt-4.1-mini")]
)

# Execute
runtime = Runtime()
result = await runtime.jit(agent, problem="What is 2 + 2?")
print(result.answer)  # "4"
```

## API Overview

### Core Classes

- `Agent` - Defines an agent with tools and schemas
- `Tool` - Wraps a callable with schema
- `Runtime` - Executes agents (AOT/JIT)
- `Context` - Tracks messages and history
- `Message` - Single message in context

### Key Functions

- `get_context(name)` - Get or create context
- `set_runtime(runtime)` - Set global runtime
- `get_runtime()` - Get global runtime

### Strategies

- `Generate` - How to generate code
- `Verify` - How to validate code
- `Cost` - How to rank code
- `Compact` - How to compress contexts

