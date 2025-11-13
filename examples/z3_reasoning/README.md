# Z3-theorem-proving agents with A1

## Problem

Existing agent frameworks run a static while loop where an LLM dispatches tools at each iteration. This is especially error-prone in domains that have hard logical constraints.

## Why Agent Compilers?

The **Proof of Thought** approach ([https://github.com/DebarghaG/proofofthought](https://github.com/DebarghaG/proofofthought)) previously demonstrated that a compound system comprised of an LLM and an SMT solver can work wonders.

However, such a system need not be specialized for just SMT solver problems. Agent compilers generalize this approach by compiling an agent to code either ahead-of-time (AOT) or just-in-time, optimizing generated code for each unique agent input.

```python
from a1 import Agent, Skill, LLM
from pydantic import BaseModel

class Answer(BaseModel):
    answer: bool

agent = Agent(
    name="z3_reasoner",
    input_schema={"question": str},
    output_schema=Answer,
    tools=[LLM("gpt-4.1-mini")],
    skills=[Skill(name="z3", content="...", modules=["z3"])]
)

result = await agent.jit(agent, question="Can humans breathe underwater?")
# Result: answer=False (proven via Z3 constraints)
```

An optimizing agent-to-code compiler like A1 offers several advantages:

1. **Safety** - Minimizes exposure of sensitive data to an LLM.
2. **Speed** - Up to 10x faster code generation.
3. **Determinism** - Code is optimized for minimal non-deterministic behavior (e.g. LLM calls replaced with code where possible).
4. **Flexibility** - Skills and Tools from any existing OpenAPI, MCP server, databases, fsspec paths, Python functions

## Solution with A1

This example demonstrates StrategyQA-style reasoning using Z3 via A1:

1. **Z3 Skill teaches formal reasoning** - In A1, an agent has tools and skills. A Z3 skill explains how to use the wonderful Z3 solver Python API.
2. **Agent generates verification code** - For each question, the LLM writes Z3 code that formally proves or disproves the statement.
3. **SMT solver provides certainty** - Z3 returns `sat` (satisfiable) or `unsat` (unsatisfiable), giving mathematically provable answers.

Example questions solved:
- "Could Javier Sotomayor jump over a giraffe?" → Encode heights as constraints → `unsat` (False)
- "Can a marathon runner complete a marathon in under 2 hours?" → Encode time constraints → `sat` (True)
- "Is a person on a pallet without a harness safe per OSHA?" → Encode safety rules → `unsat` (False)

To learn more about A1, take a look at our docs on [a1project.org](https://docs.a1project.org).
