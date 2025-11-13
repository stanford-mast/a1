# Simple Agent Example

## Problem

Building a conversational AI agent that can process user messages and maintain context across multiple interactions requires complex state management, prompt engineering, and careful orchestration of LLM calls. Traditional approaches lead to brittle code with hardcoded prompts and manual context handling.

## Why Agent Compilers?

Agent compilers like A1 let you:
- **Declare what you want** rather than how to implement it
- **Automatically optimize** code generation and execution
- **Eliminate boilerplate** for context management and tool calling
- **Get type safety** with Pydantic schemas for inputs and outputs

## Solution with A1

This example shows the simplest possible agent: one that takes a user message and responds. The agent is defined declaratively with just:

1. An input schema (user message)
2. An output schema (assistant response)  
3. Available tools (LLM)

A1 compiles this specification into optimized Python code that handles all the complexity:
- Context initialization and management
- LLM API calls with proper formatting
- Response parsing and validation
- Error handling

**Key takeaway**: With A1, a fully functional conversational agent requires just 20 lines of declarative code instead of hundreds of lines of imperative plumbing.
