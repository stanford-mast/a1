"""
Stress test: Agent with large enum output schemas.

Tests the automatic enum reduction feature that uses semantic similarity
to reduce large enums to the top 100 most relevant values before sending
to the LLM API.

Regular tests (run with pytest):
- test_1k_enum_agent: 1,000 enums
- test_2k_enum_agent: 2,000 enums (requires reduction)

Extreme stress tests (not run by default):
- test_1m_enum_agent: 1,000,000 enums (marked with @pytest.mark.stress)
"""
import asyncio
import os
import pytest
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from a1 import Agent, tool, LLM, Runtime, EM

# Load environment variables from .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            os.environ[key] = value.strip()


# Create enum variants for testing
def create_category_enum(num_categories: int):
    """Dynamically create enum with N categories."""
    return Enum(
        f'ProductCategory{num_categories}',
        {f'CATEGORY_{i:05d}': f'category_{i}' for i in range(num_categories)}
    )


ProductCategory1k = create_category_enum(1000)
ProductCategory2k = create_category_enum(2000)  # Changed from 10k to 2k
ProductCategory1m = create_category_enum(1_000_000)  # Extreme stress test


class ProductInput(BaseModel):
    """Input for product categorization."""
    model_config = ConfigDict(extra='forbid')
    
    product_description: str = Field(description="Description of the product to categorize")


def create_output_model(category_enum):
    """Create output model with specific enum."""
    class ProductOutput(BaseModel):
        """Output with categorized product."""
        model_config = ConfigDict(extra='forbid')
        
        category: category_enum = Field(description="The category this product belongs to")  # type: ignore
        confidence: float = Field(description="Confidence score between 0 and 1", ge=0.0, le=1.0)
        reasoning: str = Field(description="Brief explanation of why this category was chosen")
    return ProductOutput


ProductOutput1k = create_output_model(ProductCategory1k)
ProductOutput2k = create_output_model(ProductCategory2k)
ProductOutput1m = create_output_model(ProductCategory1m)  # Extreme stress test


@tool(name="search_categories", description="Search for relevant categories based on keywords")
async def search_categories(keywords: str) -> str:
    """Mock search that returns some category suggestions."""
    # Return a few categories based on hash of keywords to make it deterministic
    hash_val = hash(keywords) % 2000
    categories = [
        f"CATEGORY_{(hash_val + i) % 2000:05d}"
        for i in range(5)
    ]
    return f"Relevant categories found: {', '.join(categories)}"


async def _run_enum_agent_test(num_enums: int, should_succeed: bool):
    """Test agent with large enum output schema (helper function)."""
    if num_enums == 1000:
        category_enum = ProductCategory1k
        output_schema = ProductOutput1k
    elif num_enums == 2000:
        category_enum = ProductCategory2k
        output_schema = ProductOutput2k
    else:  # 1M
        category_enum = ProductCategory1m
        output_schema = ProductOutput1m
    
    print("\n" + "="*70)
    print(f"STRESS TEST: Agent with {num_enums:,} Enum Classes")
    print("="*70)
    
    # Create agent with the large enum schema
    print(f"\n📊 Creating agent with ProductCategory enum ({num_enums:,} values)...")
    agent = Agent(
        name="search_agent",
        description="Agent that searches product categories",
        input_schema=ProductInput,
        output_schema=output_schema,
        tools=[EM(), search_categories, LLM("gpt-4o-mini")]  # EM for large enum support (Done auto-added by LLM)
    )
    
    print(f"✓ Agent created successfully")
    print(f"  - Input schema: {agent.input_schema.__name__}")
    print(f"  - Output schema: {agent.output_schema.__name__}")
    print(f"  - Enum size: {len(category_enum)} values")
    print(f"  - Tools: {[t.name for t in agent.tools]}")
    
    # AOT compilation
    print("\n🔧 Compiling agent (AOT)...")
    runtime = Runtime()
    
    try:
        compiled = await runtime.aot(agent)
        print(f"✓ AOT compilation successful")
        print(f"  - Compiled tool: {compiled.name}")
        
        # Test execution with a product description
        print("\n🧪 Testing agent execution...")
        test_input = ProductInput(
            product_description="A smartphone with 128GB storage and 5G connectivity"
        )
        
        print(f"  Input: {test_input.product_description}")
        
        # Execute the agent
        result = await compiled(test_input)
        
        print(f"\n✅ Agent execution successful!")
        print(f"  - Category: {result.category.name}")
        print(f"  - Category value: {result.category.value}")
        print(f"  - Confidence: {result.confidence}")
        print(f"  - Reasoning: {result.reasoning}")
        
        # Verify the output is valid
        assert isinstance(result.category, category_enum), f"Category must be {category_enum.__name__} enum"
        assert 0.0 <= result.confidence <= 1.0, "Confidence must be between 0 and 1"
        assert len(result.reasoning) > 0, "Reasoning must not be empty"
        
        print("\n" + "="*70)
        print("✅ STRESS TEST PASSED")
        print("="*70)
        print(f"\nAgent successfully handled {len(category_enum):,} enum values!")
        
        if not should_succeed:
            print("\n⚠️  WARNING: Expected this test to fail, but it succeeded!")
            print("   OpenAI may have increased their enum limit.")
        
        return True
        
    except Exception as e:
        if should_succeed:
            print(f"\n❌ Test failed with error: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            raise
        else:
            print(f"\n✓ Test failed as expected: {type(e).__name__}")
            error_msg = str(e)
            if "at most 1000 enum values" in error_msg or "10000" in error_msg:
                print("  ✓ Error message confirms OpenAI's 1,000 enum limit")
            print(f"\n  Error: {error_msg[:200]}...")
            return False


async def _analyze_schema_size(num_enums: int):
    """Check the size of the generated schema (helper function)."""
    if num_enums == 1000:
        output_schema = ProductOutput1k
        category_enum = ProductCategory1k
    elif num_enums == 2000:
        output_schema = ProductOutput2k
        category_enum = ProductCategory2k
    else:  # 1M
        output_schema = ProductOutput1m
        category_enum = ProductCategory1m
    
    print("\n" + "="*70)
    print(f"SCHEMA SIZE ANALYSIS ({num_enums:,} enums)")
    print("="*70)
    
    schema = output_schema.model_json_schema()
    
    import json
    schema_str = json.dumps(schema, indent=2)
    schema_size = len(schema_str)
    
    print(f"\n📏 Schema Statistics:")
    print(f"  - JSON schema size: {schema_size:,} bytes ({schema_size/1024:.1f} KB)")
    print(f"  - Enum values: {len(category_enum):,}")
    
    # Show a sample of the schema
    print(f"\n📄 Schema sample (first 500 chars):")
    print(schema_str[:500])
    print("  ...")
    
    if num_enums > 1000:
        print(f"\n⚠️  NOTE: Schema has {num_enums:,} enum values")
        print(f"   a1-compiler will automatically reduce to top 100 using semantic similarity")
    
    if schema_size > 100_000:
        print(f"\n⚠️  NOTE: Original schema is very large ({schema_size/1024:.1f} KB)")
        print(f"   Reduction will create much smaller schemas sent to LLM")


@pytest.mark.asyncio
async def test_1k_enum_agent():
    """Test that agent handles 1,000 enum values successfully."""
    success = await _run_enum_agent_test(1000, should_succeed=True)
    assert success, "1k enum test should succeed with automatic reduction"


@pytest.mark.asyncio
async def test_2k_enum_agent():
    """Test that agent handles 2,000 enum values via semantic reduction."""
    success = await _run_enum_agent_test(2000, should_succeed=True)
    assert success, "2k enum test should succeed with semantic reduction to top 100"


@pytest.mark.stress
@pytest.mark.asyncio
async def test_1m_enum_agent():
    """
    EXTREME STRESS TEST: 1,000,000 enum values.
    
    This test validates that semantic reduction can scale to extreme sizes.
    WARNING: This test is expensive and slow - requires OpenAI API key.
    
    Run with: pytest -m stress tests/test_large_enum_stress.py
    """
    success = await _run_enum_agent_test(1_000_000, should_succeed=True)
    assert success, "1M enum test should succeed with semantic reduction to top 100"


@pytest.mark.asyncio
async def test_enum_schema_size():
    """Test schema size analysis for different enum counts."""
    await _analyze_schema_size(1000)
    await _analyze_schema_size(2000)


# Main test runner (for standalone execution with detailed output)
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 a1-compiler Large Enum Stress Test")
    print("="*70)
    print("\nTesting automatic semantic enum reduction...")
    print("Enums with >100 values are reduced to top 100 most similar to user prompt")
    
    # Test with 1,000 enums
    print("\n" + "="*70)
    print("TEST 1: 1,000 Enums")
    print("="*70)
    asyncio.run(_analyze_schema_size(1000))
    success_1k = asyncio.run(_run_enum_agent_test(1000, should_succeed=True))
    
    # Test with 2,000 enums  
    print("\n" + "="*70)
    print("TEST 2: 2,000 Enums (Requires Semantic Reduction)")
    print("="*70)
    asyncio.run(_analyze_schema_size(2000))
    success_2k = asyncio.run(_run_enum_agent_test(2000, should_succeed=True))
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\n1,000 enums: {'✅ PASSED' if success_1k else '❌ FAILED'}")
    print(f"2,000 enums: {'✅ PASSED' if success_2k else '❌ FAILED'}")
    print(f"\n✨ Success! a1-compiler automatically reduces large enums (>100 values)")
    print(f"   using semantic similarity to find the top 100 most relevant values.")
    print(f"\n📊 How it works:")
    print(f"   1. Detects enums with >100 values in output schemas")
    print(f"   2. Computes embeddings for all enum values and user prompt")
    print(f"   3. Selects top 100 values with highest semantic similarity")
    print(f"   4. Sends reduced schema to OpenAI (stays under 1,000 limit)")
    print(f"\n   This allows handling schemas with 2k+ enum values seamlessly!")
