"""
Tests for Skills and SkillSets functionality.

Tests that skills and skillsets are properly integrated with agents,
included in generated definition code, and available during execution.
"""

import pytest
from pydantic import BaseModel, Field

from a1 import LLM, Agent, Skill, SkillSet, Tool
from a1.codegen import BaseGenerate


class TestSkillIntegration:
    """Test basic skill integration with agents."""

    def test_agent_with_single_skill(self):
        """Test agent with a single skill."""
        # Create a simple pandas skill
        pandas_skill = Skill(
            name="pandas_basics",
            description="Pandas library basics for data manipulation",
            content="""
# Pandas Basics

## Loading CSV Files
import pandas as pd
df = pd.read_csv('file.csv')

## Basic Operations
- df.head() - view first rows
- df.describe() - get statistics
- df.groupby('column').sum() - group and aggregate
- df['column'].mean() - calculate mean
- df[df['column'] > value] - filter rows

## Writing Results
df.to_csv('output.csv', index=False)
""",
            modules=["pandas"],
        )

        class Input(BaseModel):
            task: str = Field(..., description="Data task to perform")

        class Output(BaseModel):
            result: str = Field(..., description="Result of data operation")

        agent = Agent(
            name="data_agent",
            description="Agent that performs data operations",
            input_schema=Input,
            output_schema=Output,
            skills=[pandas_skill],
        )

        assert agent.name == "data_agent"
        assert len(agent.skills) == 1
        assert agent.skills[0].name == "pandas_basics"

    def test_skill_content_in_definition_code(self):
        """Test that skill content appears in generated definition code."""
        pandas_skill = Skill(
            name="pandas_csv",
            description="How to load and process CSV files with pandas",
            content="import pandas as pd\ndf = pd.read_csv('data.csv')\ndf.head()",
            modules=["pandas"],
        )

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            result: str

        agent = Agent(
            name="csv_agent",
            description="Process CSV files",
            input_schema=Input,
            output_schema=Output,
            skills=[pandas_skill],
        )

        gen = BaseGenerate(LLM("gpt-4.1-mini"))
        def_code = gen._build_definition_code(agent, return_function=False)

        # Check that skill content is in definition code
        assert "AVAILABLE SKILLS" in def_code
        assert "pandas_csv" in def_code
        assert "How to load and process CSV files" in def_code
        assert "import pandas as pd" in def_code

    def test_skill_modules_imported_in_definition_code(self):
        """Test that skill modules are imported in definition code."""
        pandas_skill = Skill(
            name="pandas_data",
            description="Pandas data operations",
            content="Use pandas for data manipulation",
            modules=["pandas", "numpy"],
        )

        class Input(BaseModel):
            data: str

        class Output(BaseModel):
            result: str

        agent = Agent(
            name="data_agent",
            description="Data processing",
            input_schema=Input,
            output_schema=Output,
            skills=[pandas_skill],
        )

        gen = BaseGenerate(LLM("gpt-4.1-mini"))
        def_code = gen._build_definition_code(agent, return_function=False)

        # Check that modules are imported
        assert "import pandas" in def_code
        assert "import numpy" in def_code

    def test_skillset_with_multiple_skills(self):
        """Test SkillSet containing multiple skills."""
        data_skill = Skill(
            name="data_loading",
            description="Loading data from various sources",
            content="Use pandas.read_csv() for CSV files",
            modules=["pandas"],
        )

        analysis_skill = Skill(
            name="data_analysis",
            description="Analyzing data with statistics",
            content="Use df.describe() for summary statistics",
            modules=["pandas", "scipy"],
        )

        data_skillset = SkillSet(
            name="data_science", description="Data science and analysis skills", skills=[data_skill, analysis_skill]
        )

        class Input(BaseModel):
            task: str

        class Output(BaseModel):
            result: str

        agent = Agent(
            name="ds_agent",
            description="Data science agent",
            input_schema=Input,
            output_schema=Output,
            skills=[data_skillset],
        )

        assert len(agent.skills) == 1
        assert agent.skills[0].name == "data_science"
        assert len(agent.skills[0].skills) == 2

    def test_nested_skillset_in_definition_code(self):
        """Test that nested skillset content appears in definition code."""
        skill1 = Skill(
            name="skill_one", description="First skill", content="Content of skill one", modules=["module_a"]
        )

        skill2 = Skill(
            name="skill_two", description="Second skill", content="Content of skill two", modules=["module_b"]
        )

        skillset = SkillSet(name="combined_skills", description="Collection of skills", skills=[skill1, skill2])

        class Input(BaseModel):
            input: str

        class Output(BaseModel):
            output: str

        agent = Agent(
            name="agent", description="Test agent", input_schema=Input, output_schema=Output, skills=[skillset]
        )

        gen = BaseGenerate(LLM("gpt-4.1-mini"))
        def_code = gen._build_definition_code(agent, return_function=False)

        # Check both skills appear in output
        assert "skill_one" in def_code
        assert "skill_two" in def_code
        assert "Content of skill one" in def_code
        assert "Content of skill two" in def_code
        # Check skillset appears
        assert "combined_skills" in def_code

    def test_mixed_tools_and_skills(self):
        """Test agent with both tools and skills."""

        # Create a simple tool
        class CalcInput(BaseModel):
            a: int
            b: int

        class CalcOutput(BaseModel):
            result: int

        async def calculator(a: int, b: int) -> dict:
            return {"result": a + b}

        tool = Tool(
            name="calculator",
            description="Add two numbers",
            input_schema=CalcInput,
            output_schema=CalcOutput,
            execute=calculator,
            is_terminal=False,
        )

        # Create a skill
        math_skill = Skill(
            name="math_concepts",
            description="Mathematical concepts and techniques",
            content="Arithmetic operations: addition, subtraction, multiplication, division",
            modules=["math"],
        )

        class Input(BaseModel):
            query: str

        class Output(BaseModel):
            answer: str

        agent = Agent(
            name="math_agent",
            description="Math solving agent",
            input_schema=Input,
            output_schema=Output,
            tools=[tool],
            skills=[math_skill],
        )

        assert len(agent.tools) == 1
        assert len(agent.skills) == 1

        gen = BaseGenerate(LLM("gpt-4.1-mini"))
        def_code = gen._build_definition_code(agent, return_function=False)

        # Check both tool and skill appear
        assert "calculator" in def_code
        assert "math_concepts" in def_code
        assert "import math" in def_code

    def test_pandas_csv_skill_example(self):
        """Test with a realistic pandas CSV skill."""
        pandas_csv_skill = Skill(
            name="csv_processing",
            description="Loading and basic processing of CSV files with Pandas",
            content="""
# CSV File Processing with Pandas

## Load a CSV File
```python
import pandas as pd
df = pd.read_csv('data.csv')
```

## Explore the Data
```python
df.head()  # View first 5 rows
df.shape  # Get dimensions
df.info()  # Get column info
df.describe()  # Statistical summary
```

## Basic Filtering
```python
filtered_df = df[df['column_name'] > value]
```

## Aggregations
```python
df.groupby('category').sum()
df['price'].mean()
df['quantity'].max()
```

## Save Results
```python
df.to_csv('output.csv', index=False)
```
""",
            modules=["pandas"],
        )

        class Input(BaseModel):
            filename: str = Field(..., description="CSV file to process")
            operation: str = Field(..., description="Operation: load, filter, aggregate, save")

        class Output(BaseModel):
            result: str = Field(..., description="Result of operation")

        agent = Agent(
            name="csv_processor",
            description="Process CSV files",
            input_schema=Input,
            output_schema=Output,
            skills=[pandas_csv_skill],
        )

        gen = BaseGenerate(LLM("gpt-4.1-mini"))
        def_code = gen._build_definition_code(agent, return_function=False)

        # Verify skill content is accessible in definition code
        assert "CSV File Processing" in def_code
        assert "Load a CSV File" in def_code
        assert "df.head()" in def_code
        assert "import pandas" in def_code


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
