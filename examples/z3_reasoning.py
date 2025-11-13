"""
Z3 Logical Reasoning Agent - StrategyQA-style Examples

Demonstrates using Z3 theorem prover for formal logical reasoning.
The agent generates Z3 code to solve logical problems, similar to 
the ProofOfThought approach discussed on HackerNews.

The key insight: LLMs can generate formal Z3 programs that provide
verifiable answers to reasoning questions. No special tool needed -
just teach the agent Z3py and let it write the verification code.

Based on:
- https://news.ycombinator.com/item?id=45475529
- https://ericpony.github.io/z3py-tutorial/guide-examples.htm

Run with: uv run python examples/z3_reasoning.py
"""

import asyncio
import logging

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from a1 import Agent, LLM, Runtime, Skill

# Load environment variables from .env file
load_dotenv()

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(name)s - %(message)s'
)


# ============================================================================
# Z3 Skill - Teaches the agent how to use Z3 theorem prover
# ============================================================================

Z3_SKILL = Skill(
    name="z3_reasoning",
    description="How to use Z3 theorem prover for logical reasoning and constraint solving",
    content="""
# Z3 Theorem Prover - Python API

Z3 is an SMT (Satisfiability Modulo Theories) solver that can verify logical statements
with mathematical certainty. Use it to prove whether statements are true or false.

IMPORTANT: Import specific functions from z3 module (e.g., `from z3 import Solver, Real, DeclareSort, Const, Function, RealSort, BoolSort, sat, unsat`).
DO NOT use `from z3 import *` as it won't work in this environment.

## Quick Examples

```python
from z3 import Solver, DeclareSort, Function, RealSort, Const, sat, unsat

# Example 1: Can Javier Sotomayor (2.45m jump) clear a giraffe (5.5m)?
Person = DeclareSort('Person')
Animal = DeclareSort('Animal')
jump_height = Function('jump_height', Person, RealSort())
height = Function('height', Animal, RealSort())

javier = Const('javier', Person)
giraffe = Const('giraffe', Animal)

s = Solver()
s.add(jump_height(javier) == 2.45)
s.add(height(giraffe) == 5.5)
s.add(jump_height(javier) >= height(giraffe))  # Can he clear it?

result = s.check()  # Returns: unsat (false - he cannot)

# Example 2: Basic constraint solving
from z3 import Int
x = Int('x')
y = Int('y')
s = Solver()
s.add(x > 2, y < 10, x + 2*y == 7)
if s.check() == sat:
    m = s.model()
    print(m[x], m[y])

# Example 3: Boolean logic with quantifiers
from z3 import BoolSort, ForAll, Implies, Not
Group = DeclareSort('Group')
oppose_allotment = Function('oppose_allotment', Group, BoolSort())
send_delegation = Function('send_delegation', Group, BoolSort())

cherokee = Const('cherokee', Group)
g = Const('g', Group)

s = Solver()
s.add(ForAll([g], Implies(send_delegation(g), oppose_allotment(g))))
s.add(send_delegation(cherokee))
s.add(Not(oppose_allotment(cherokee)))  # Did they oppose?

result = s.check()  # Returns: unsat (contradiction - they DID oppose)
```

## Core API

Import what you need from the `z3` module (NOT `z3-solver`):
- `from z3 import Solver, Int, Real, Bool, DeclareSort, Function, Const`
- `from z3 import sat, unsat, RealSort, IntSort, BoolSort`
- `from z3 import And, Or, Not, Implies, ForAll, Exists`

Key functions:
- `DeclareSort('Name')` - Create custom type
- `Function('name', Type1, Type2, ..., ReturnType)` - Declare function
- `Const('name', Sort)` - Create constant of a type
- `Int('x')`, `Real('y')`, `Bool('p')` - Create variables
- `s = Solver()` - Create solver
- `s.add(constraint)` - Add constraint
- `s.check()` - Returns `sat`, `unsat`, or `unknown`
- `s.model()` - Get solution if sat

## Logic Operators

- `And(a, b)`, `Or(a, b)`, `Not(a)`, `Implies(a, b)`
- `ForAll([var], expr)`, `Exists([var], expr)`
- `==`, `!=`, `<`, `>`, `<=`, `>=` for comparisons

## Strategy

1. Identify entities and their types (sorts)
2. Define properties as functions
3. Add known facts as constraints
4. Add the question as a constraint to test
5. Check result: `unsat` means the constraint is false, `sat` means it could be true
""",
    modules=["z3"],  # Python module name, not package name
)


# ============================================================================
# Main Example - Agent generates Z3 code to solve reasoning problems
# ============================================================================


async def main():
    """Demonstrate Z3-based reasoning on StrategyQA-style questions."""

    print("=" * 80)
    print("Z3 Logical Reasoning Agent - StrategyQA Examples")
    print("=" * 80)

    # Define input/output schemas
    class QuestionInput(BaseModel):
        """User's question requiring logical reasoning"""

        question: str = Field(description="A yes/no question requiring logical reasoning")

    class Answer(BaseModel):
        """Simple boolean answer"""

        answer: bool = Field(description="True or False")

    # Create agent with Z3 skill
    # The agent will generate Z3 code to formally verify the answer
    agent = Agent(
        name="z3_reasoner",
        description=(
            "Answer yes/no questions using Z3 theorem prover for formal verification. "
            "Generate Z3 Python code that encodes the problem as constraints, "
            "then run it to get a provably correct answer."
        ),
        input_schema=QuestionInput,
        output_schema=Answer,
        tools=[LLM("gpt-4.1-mini")],
        skills=[Z3_SKILL],
    )

    # Create runtime
    runtime = Runtime()

    # Test cases from StrategyQA-style reasoning
    test_cases = [
        {
            "question": "Could Javier Sotomayor jump over the head of the average giraffe?",
            "expected": False,
            "knowledge": "Javier Sotomayor's high jump record is 2.45m. Average giraffe height is 5.5m.",
        },
        {
            "question": "Can a marathon runner complete a marathon in under 2 hours?",
            "expected": True,
            "knowledge": "The world record for marathon is under 2 hours (Eliud Kipchoge, 1:59:40).",
        },
        {
            "question": "Could a person standing on a pallet without a harness be considered safe according to OSHA?",
            "expected": False,
            "knowledge": "OSHA requires fall protection when working at heights without proper equipment.",
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print("\n" + "=" * 80)
        print(f"Test Case {i}: {test_case['question']}")
        print("=" * 80)
        print(f"Knowledge: {test_case['knowledge']}")
        print(f"Expected Answer: {test_case['expected']}")

        try:
            # Run the agent - it will generate and execute Z3 code
            result = await runtime.jit(agent, question=test_case["question"])

            print(f"\n✅ Answer: {result.answer}")

            # Check if answer matches expected
            if result.answer == test_case["expected"]:
                print("✓ Correct!")
            else:
                print(f"✗ Wrong (expected {test_case['expected']})")

        except Exception as e:
            print(f"\n❌ Failed: {e}")

    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)
    print("\nKey Insight from ProofOfThought:")
    print("  LLMs can generate formal Z3 programs that provide verifiable answers.")
    print("  No special tools needed - just teach the agent Z3py via a Skill.")
    print("  The agent writes the verification code itself!")
    print("\nFeatures:")
    print("  ✓ Agent generates Z3 code to solve reasoning problems")
    print("  ✓ Formal verification with provable correctness")
    print("  ✓ StrategyQA-style real-world questions")
    print("  ✓ Simple output - just boolean answer")


if __name__ == "__main__":
    asyncio.run(main())
