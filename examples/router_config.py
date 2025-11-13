"""
Router Configuration Agent - Simple JSON-based Example

Demonstrates loading command schemas from JSON and creating a1 tools dynamically.
This example shows the simplest approach: iterate through JSON, create Pydantic models
and tools, then let the agent figure out the rest.

Run with: uv run python examples/router_config.py
"""

import asyncio
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field, create_model

from a1 import Agent, Done, EM, LLM, Runtime, Tool, Strategy
from a1.strategies import BaseGenerate
from a1.extra_strategies import CheckOrdering, ReduceAndGenerate

# Load environment variables from .env file
load_dotenv()


# ============================================================================
# JSON Schema to Pydantic Conversion
# ============================================================================


def json_type_to_python(prop_schema: dict) -> type:
    """Convert JSON Schema type to Python type."""
    json_type = prop_schema.get("type", "string")
    
    if json_type == "string":
        return str
    elif json_type == "integer":
        return int
    elif json_type == "number":
        return float
    elif json_type == "boolean":
        return bool
    elif json_type == "array":
        # Simplified: assume list of strings
        return list[str]
    else:
        return str


def json_schema_to_pydantic(class_name: str, json_schema: dict) -> type[BaseModel]:
    """
    Create Pydantic model from JSON schema.
    
    Preserves Field validators like pattern (regex), minimum/maximum, etc.
    """
    properties = json_schema.get("properties", {})
    required = json_schema.get("required", [])
    
    if not properties:
        return create_model(class_name)
    
    fields = {}
    for prop_name, prop_schema in properties.items():
        prop_type = json_type_to_python(prop_schema)
        
        # Build Field kwargs with validators
        field_kwargs = {}
        
        if "description" in prop_schema:
            field_kwargs["description"] = prop_schema["description"]
        
        if "pattern" in prop_schema:
            field_kwargs["pattern"] = prop_schema["pattern"]
        
        if "minimum" in prop_schema:
            field_kwargs["ge"] = prop_schema["minimum"]
        
        if "maximum" in prop_schema:
            field_kwargs["le"] = prop_schema["maximum"]
        
        if "minLength" in prop_schema:
            field_kwargs["min_length"] = prop_schema["minLength"]
        
        if "maxLength" in prop_schema:
            field_kwargs["max_length"] = prop_schema["maxLength"]
        
        # Required vs optional
        is_required = prop_name in required
        default = ... if is_required else None
        
        # Create field
        if field_kwargs:
            fields[prop_name] = (prop_type, Field(default, **field_kwargs))
        else:
            fields[prop_name] = (prop_type, default)
    
    return create_model(class_name, **fields)


# ============================================================================
# Simple Tool Creation from JSON
# ============================================================================


class CommandResult(BaseModel):
    """Result from executing a router command"""
    command: str
    status: str
    parameters: dict[str, Any]


def create_tool_from_json(cmd_name: str, cmd_config: dict) -> Tool:
    """
    Create a Tool instance from JSON command config.
    
    Returns:
        Tool instance ready to use
    """
    description = cmd_config.get("description", f"Execute {cmd_name}")
    schema_dict = cmd_config.get("schema", {})
    
    # Create Pydantic model for the input schema
    schema_name = f"{cmd_name}_Input".replace("_", " ").title().replace(" ", "")
    InputModel = json_schema_to_pydantic(schema_name, schema_dict)
    
    # Create the execute function
    async def execute(**kwargs) -> CommandResult:
        """Dynamic tool function - validates and executes command"""
        return CommandResult(
            command=cmd_name,
            status="success",
            parameters=kwargs
        )
    
    # Create Tool instance
    return Tool(
        name=cmd_name,
        description=description,
        input_schema=InputModel,
        output_schema=CommandResult,
        execute=execute
    )


# ============================================================================
# Main Example
# ============================================================================


async def main():
    """Load router commands from JSON and create an agent."""
    
    print("=" * 80)
    print("Router Configuration Agent - Simple JSON Example")
    print("=" * 80)
    
    # Load command schemas from JSON
    json_file = Path(__file__).parent / "router_schema.json"
    
    if not json_file.exists():
        print(f"\n❌ Error: {json_file} not found")
        print("This example requires router_schema.json in the examples directory")
        return
    
    with open(json_file) as f:
        data = json.load(f)
    
    commands = data.get("commands", {})
    print(f"\n✓ Loaded {len(commands)} commands from JSON")
    
    # Create tools from JSON (simple iteration - no dependency graph)
    tools = []
    for cmd_name, cmd_config in commands.items():
        tool_func = create_tool_from_json(cmd_name, cmd_config)
        tools.append(tool_func)
        print(f"  • {cmd_name}: {cmd_config.get('description', 'No description')[:60]}...")
    
    # Add LLM and Done tools
    tools.append(LLM("gpt-4.1-mini"))
    tools.append(Done())
    
    # Define input schema for task description
    class TaskInput(BaseModel):
        """User's configuration task"""
        task: str = Field(description="Natural language description of what to configure")
    
    # Define output schema
    class RouterConfig(BaseModel):
        """Configuration result"""
        commands_executed: list[str] = Field(description="List of commands that were run")
        success: bool = Field(description="Whether configuration succeeded")
        summary: str = Field(description="Summary of what was configured")
    
    # Create agent
    agent = Agent(
        name="router_agent",
        description="Configure Cisco router using JSON-defined commands",
        input_schema=TaskInput,
        output_schema=RouterConfig,
        tools=tools,
    )
    
    # Create runtime with strategies:
    # - ReduceAndGenerate: Filter tools and reduce enums using embeddings, then generate code
    # - CheckOrdering: Verify logical ordering of tool calls
    strategy = Strategy(
        generate=ReduceAndGenerate(em_tool=EM(), llm_tool=LLM("gpt-4.1-mini")),
        verify=[CheckOrdering()],
        max_iterations=5
    )
    runtime = Runtime(strategy=strategy)
    
    # Test case 1: Simple VLAN configuration
    print("\n" + "=" * 80)
    print("Test Case 1: Create VLAN 100 named 'Engineering'")
    print("=" * 80)
    
    try:
        result = await runtime.jit(
            agent,
            task="Create VLAN 100 named 'Engineering'"
        )
        print("\n✅ Success!")
        print(f"Commands: {result.commands_executed}")
        print(f"Summary: {result.summary}")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
    
    # Test case 2: Interface configuration
    print("\n" + "=" * 80)
    print("Test Case 2: Configure interface Gi1/0/1 with IP 192.168.1.1")
    print("=" * 80)
    
    try:
        result = await runtime.jit(
            agent,
            task="Configure interface Gi1/0/1 with IP address 192.168.1.1 and subnet mask 255.255.255.0 in VLAN 100"
        )
        print("\n✅ Success!")
        print(f"Commands: {result.commands_executed}")
        print(f"Summary: {result.summary}")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
    
    # Test case 3: Multi-command task (VLAN + trunk configuration)
    print("\n" + "=" * 80)
    print("Test Case 3: Create VLAN 200 and configure Gi1/0/2 as trunk with VLANs 100,200")
    print("=" * 80)
    
    try:
        result = await runtime.jit(
            agent,
            task="Create VLAN 200 named 'Sales' and configure interface Gi1/0/2 as a trunk port allowing VLANs 100 and 200"
        )
        print("\n✅ Success!")
        print(f"Commands: {result.commands_executed}")
        print(f"Summary: {result.summary}")
    except Exception as e:
        print(f"\n❌ Failed: {e}")
    
    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)
    print("\nKey Features:")
    print("  ✓ Load command schemas from JSON file")
    print("  ✓ Dynamic Pydantic model creation")
    print("  ✓ Field validation (regex patterns, numeric bounds)")
    print("  ✓ Simple tool creation - just iterate through JSON")
    print("  ✓ CheckOrdering strategy - verifies logical tool call sequences")
    print("  ✓ ReduceAndGenerate strategy - simplifies generated code")
    print("  ✓ No complex dependency graph - let the agent figure it out")


if __name__ == "__main__":
    asyncio.run(main())
