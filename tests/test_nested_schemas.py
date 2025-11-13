"""
Tests for nested schema support in code generation and execution.

Tests that tools with arbitrarily nested input/output schemas
are properly handled in:
- Definition code generation (_build_definition_code)
- Generated code execution (BaseExecutor)
- Tool wrapper ergonomics (calling with **kwargs)
"""

import pytest
from pydantic import BaseModel, Field

from a1 import Agent, Runtime, Tool

# ============================================================================
# Nested Schema Definitions
# ============================================================================


class Address(BaseModel):
    """A nested address structure."""

    street: str = Field(..., description="Street address")
    city: str = Field(..., description="City name")
    zip_code: str = Field(..., description="Zip code")


class Contact(BaseModel):
    """A contact with nested address."""

    name: str = Field(..., description="Person's name")
    email: str = Field(..., description="Email address")
    address: Address = Field(..., description="Home address")
    phone: str | None = Field(None, description="Phone number")


class Restaurant(BaseModel):
    """A restaurant with nested menu items."""

    name: str = Field(..., description="Restaurant name")
    location: Address = Field(..., description="Location")

    class MenuItem(BaseModel):
        """A menu item."""

        name: str = Field(..., description="Item name")
        price: float = Field(..., description="Price")
        description: str = Field(..., description="Item description")

    menu_items: list[MenuItem] = Field(default_factory=list, description="Menu items")


class SearchRestaurantOutput(BaseModel):
    """Output from restaurant search - highly nested."""

    restaurants: list[Restaurant] = Field(..., description="Found restaurants")
    total_count: int = Field(..., description="Total count")


# ============================================================================
# Tool Definitions with Nested Schemas
# ============================================================================


class SearchRestaurantsInput(BaseModel):
    """Input for searching restaurants."""

    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, description="Maximum results")


async def search_restaurants(query: str, max_results: int = 10) -> SearchRestaurantOutput:
    """Search for restaurants."""
    return SearchRestaurantOutput(
        restaurants=[
            Restaurant(
                name="The Grill",
                location=Address(street="123 Main St", city="NYC", zip_code="10001"),
                menu_items=[
                    Restaurant.MenuItem(name="Steak", price=29.99, description="Prime steak"),
                    Restaurant.MenuItem(name="Pasta", price=15.99, description="Penne pasta"),
                ],
            )
        ],
        total_count=1,
    )


async def store_contact(name: str, email: str, address: Address, phone: str | None = None) -> dict:
    """Store a contact with nested address."""
    return {"status": "success", "contact_name": name, "city": address.city}


class StoreContactOutput(BaseModel):
    """Output from store_contact."""

    status: str = Field(..., description="Status")
    contact_name: str = Field(..., description="Contact name")
    city: str = Field(..., description="City")


# ============================================================================
# Test Cases
# ============================================================================


class TestNestedSchemaGeneration:
    """Test nested schema handling in definition code generation."""

    def test_simple_nested_input_schema(self):
        """Test definition code with nested input schema."""
        from a1 import LLM
        from a1.codegen import BaseGenerate

        # Create a tool with nested input
        tool = Tool(
            name="store_contact",
            description="Store a contact",
            input_schema=Contact,
            output_schema=StoreContactOutput,
            execute=store_contact,
            is_terminal=False,
        )

        # Create agent with this tool
        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        agent = Agent(
            name="test_agent", description="Test agent", input_schema=Input, output_schema=Output, tools=[tool]
        )

        # Generate definition code
        gen = BaseGenerate(LLM("gpt-4.1-mini"))
        def_code = gen._build_definition_code(agent, return_function=False)

        # Check that nested classes are generated
        assert "class Address(BaseModel):" in def_code
        assert "class StoreContactInput(BaseModel):" in def_code
        assert "street: str" in def_code
        assert "city: str" in def_code

    def test_deeply_nested_output_schema(self):
        """Test definition code with deeply nested output schema."""
        from a1 import LLM
        from a1.codegen import BaseGenerate

        tool = Tool(
            name="search_restaurants",
            description="Search restaurants",
            input_schema=SearchRestaurantsInput,
            output_schema=SearchRestaurantOutput,
            execute=search_restaurants,
            is_terminal=False,
        )

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        agent = Agent(
            name="test_agent", description="Test agent", input_schema=Input, output_schema=Output, tools=[tool]
        )

        gen = BaseGenerate(LLM("gpt-4.1-mini"))
        def_code = gen._build_definition_code(agent, return_function=False)

        # Check nested classes
        assert "class SearchRestaurantOutput(BaseModel):" in def_code
        assert "class Restaurant(BaseModel):" in def_code
        assert "class Address(BaseModel):" in def_code
        assert "menu_items:" in def_code and "List[" in def_code


class TestNestedSchemaExecution:
    """Test nested schema handling in code execution."""

    @pytest.mark.asyncio
    async def test_execute_with_nested_input(self):
        """Test executing code that uses nested input schemas."""
        from a1.executor import BaseExecutor

        # Create tool with nested schema
        tool = Tool(
            name="store_contact",
            description="Store a contact",
            input_schema=Contact,
            output_schema=StoreContactOutput,
            execute=store_contact,
            is_terminal=False,
        )

        executor = BaseExecutor()

        # Code that creates nested structure and calls tool
        code = """
address = Address(street="123 Main St", city="NYC", zip_code="10001")
contact = Contact(name="John", email="john@example.com", address=address, phone="555-1234")
result = await store_contact(contact)
output = result
"""

        # Execute with tool available
        output = await executor.execute(code, tools=[tool])

        assert output.error is None
        assert output.output is not None
        assert output.output.status == "success"
        assert output.output.contact_name == "John"
        assert output.output.city == "NYC"

    @pytest.mark.asyncio
    async def test_execute_nested_output_unpacking(self):
        """Test executing code that unpacks nested output."""
        from a1.executor import BaseExecutor

        tool = Tool(
            name="search_restaurants",
            description="Search restaurants",
            input_schema=SearchRestaurantsInput,
            output_schema=SearchRestaurantOutput,
            execute=search_restaurants,
            is_terminal=False,
        )

        executor = BaseExecutor()

        code = """
results = await search_restaurants(query="Thai", max_results=5)
first_restaurant = results.restaurants[0]
restaurant_name = first_restaurant.name
city = first_restaurant.location.city
output = {
    "restaurant": restaurant_name,
    "city": city,
    "menu_count": len(first_restaurant.menu_items)
}
"""

        output = await executor.execute(code, tools=[tool])

        assert output.error is None
        assert output.output["restaurant"] == "The Grill"
        assert output.output["city"] == "NYC"
        assert output.output["menu_count"] == 2

    @pytest.mark.asyncio
    async def test_ergonomic_kwargs_with_nested_schema(self):
        """Test ergonomic **kwargs calling with nested inputs."""
        from a1.executor import BaseExecutor

        tool = Tool(
            name="store_contact",
            description="Store a contact",
            input_schema=Contact,
            output_schema=StoreContactOutput,
            execute=store_contact,
            is_terminal=False,
        )

        executor = BaseExecutor()

        # Call tool with **kwargs that match nested schema fields
        code = """
result = await store_contact(
    name="Alice",
    email="alice@example.com",
    address=Address(street="456 Oak Ave", city="LA", zip_code="90001"),
    phone="555-5678"
)
output = result
"""

        output = await executor.execute(code, tools=[tool])

        assert output.error is None
        assert output.output.status == "success"
        assert output.output.contact_name == "Alice"
        assert output.output.city == "LA"


class TestNestedSchemaIntegration:
    """Integration tests with Runtime using nested schemas."""

    @pytest.mark.asyncio
    async def test_jit_with_nested_schemas(self):
        """Test JIT mode with tools that have nested schemas."""

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            restaurant_name: str
            city: str

        tool = Tool(
            name="search_restaurants",
            description="Search restaurants",
            input_schema=SearchRestaurantsInput,
            output_schema=SearchRestaurantOutput,
            execute=search_restaurants,
            is_terminal=False,
        )

        agent = Agent(
            name="restaurant_agent",
            description="Find restaurants and extract info",
            input_schema=Input,
            output_schema=Output,
            tools=[tool],
        )

        runtime = Runtime()

        # This should work without errors despite nested schemas
        # (may fail if LLM not available, but schema handling should be fine)
        try:
            result = await runtime.jit(agent, query="Good Italian food in NYC")
            assert result is not None
        except Exception as e:
            # If it fails due to missing LLM, that's OK - we're testing schema handling
            if "GROQ_API_KEY" not in str(e) and "api_key" not in str(e).lower():
                raise


class TestNestedSchemaEdgeCases:
    """Test edge cases with nested schemas."""

    def test_deeply_nested_objects(self):
        """Test support for deeply nested object structures."""
        from a1 import LLM
        from a1.codegen import BaseGenerate

        # Create a deeply nested schema (3+ levels)
        class Level3(BaseModel):
            value: str

        class Level2(BaseModel):
            nested: Level3

        class Level1(BaseModel):
            nested: Level2

        class DeeplyNested(BaseModel):
            level1: Level1

        class DeeplyNestedOutput(BaseModel):
            status: str

        tool = Tool(
            name="deep_tool",
            description="Tool with deeply nested schema",
            input_schema=DeeplyNested,
            output_schema=DeeplyNestedOutput,
            execute=lambda x: {"status": "ok"},
            is_terminal=False,
        )

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        agent = Agent(
            name="test_agent", description="Test agent", input_schema=Input, output_schema=Output, tools=[tool]
        )

        gen = BaseGenerate(LLM("gpt-4.1-mini"))
        def_code = gen._build_definition_code(agent, return_function=False)

        # Should generate the tool input schema and all nested levels
        assert "class DeepToolInput(BaseModel):" in def_code
        assert "class Level1(BaseModel):" in def_code
        assert "class Level2(BaseModel):" in def_code
        assert "class Level3(BaseModel):" in def_code

    def test_optional_nested_fields(self):
        """Test nested schemas with optional fields."""
        from a1 import LLM
        from a1.codegen import BaseGenerate

        class OptionalAddress(BaseModel):
            street: str | None = None
            city: str = Field(..., description="City")

        class PersonWithOptionalAddress(BaseModel):
            name: str
            address: OptionalAddress | None = None

        class PersonOutput(BaseModel):
            status: str

        tool = Tool(
            name="process_person",
            description="Process person",
            input_schema=PersonWithOptionalAddress,
            output_schema=PersonOutput,
            execute=lambda x: {"status": "ok"},
            is_terminal=False,
        )

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        agent = Agent(
            name="test_agent", description="Test agent", input_schema=Input, output_schema=Output, tools=[tool]
        )

        gen = BaseGenerate(LLM("gpt-4.1-mini"))
        def_code = gen._build_definition_code(agent, return_function=False)

        # Should handle Optional nested types
        assert "class ProcessPersonInput(BaseModel):" in def_code
        assert "class OptionalAddress(BaseModel):" in def_code
        # Should show address as Optional
        assert "Optional" in def_code or "= None" in def_code
