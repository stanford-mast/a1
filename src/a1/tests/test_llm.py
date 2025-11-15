from a1 import LLM, Tool


def test_llm():
    llm = LLM("gpt-4.1-mini")

    assert llm.retry_strategy.max_iterations == 3
    assert llm.retry_strategy.num_candidates == 3
    assert isinstance(llm.tool, Tool)
    assert llm.tool.name == "llm_gpt_4_1_mini"
    assert llm.tool.description == "Call gpt-4.1-mini language model with function calling support"
    assert llm.tool.input_schema.model_json_schema() == {
        "$defs": {
            "Tool": {
                "description": "A tool is a callable function with schema validation.\n\nAttributes:\n    name: Unique identifier for the tool\n    description: Human-readable description\n    input_schema: Pydantic model for input validation\n    output_schema: Pydantic model for output validation\n    execute: Async function to execute\n    is_terminal: Whether this tool ends execution",
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "description": {"title": "Description", "type": "string"},
                    "input_schema": {"title": "Input Schema"},
                    "output_schema": {"title": "Output Schema"},
                    "is_terminal": {"default": False, "title": "Is Terminal", "type": "boolean"},
                },
                "required": ["name", "description", "input_schema", "output_schema"],
                "title": "Tool",
                "type": "object",
            }
        },
        "description": "Input for LLM tool - simplified for definition code.",
        "properties": {
            "content": {"description": "Input prompt or query", "title": "Content", "type": "string"},
            "tools": {
                "anyOf": [{"items": {"$ref": "#/$defs/Tool"}, "type": "array"}, {"type": "null"}],
                "default": None,
                "description": "Tools available for function calling",
                "title": "Tools",
            },
            "context": {
                "anyOf": [{}, {"type": "null"}],
                "default": None,
                "description": "Context object for message history tracking",
                "title": "Context",
            },
            "output_schema": {
                "anyOf": [{}, {"type": "null"}],
                "default": None,
                "description": "Optional schema to structure the output",
                "title": "Output Schema",
            },
        },
        "required": ["content"],
        "title": "LLMInput",
        "type": "object",
    }
    assert llm.tool.output_schema.model_json_schema() == {
        "description": "Output from LLM tool - simplified for definition code.",
        "properties": {"content": {"description": "Text response from LLM", "title": "Content", "type": "string"}},
        "required": ["content"],
        "title": "LLMOutput",
        "type": "object",
    }
    assert not llm.tool.is_terminal
