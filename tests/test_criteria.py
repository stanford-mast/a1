"""
Tests for QualitativeCriteria and QuantitativeCriteria.

Tests LLM-based verification and cost estimation with sampling and aggregation.
"""

import os
from unittest.mock import MagicMock

import pytest

from a1 import LLM, Agent, QualitativeCriteria, QuantitativeCriteria, Strategy


# Mock LLM for testing without real API calls
@pytest.fixture
def mock_llm():
    """Mock LLM tool that returns simple responses."""
    llm = MagicMock()
    llm.name = "mock_llm"

    async def mock_call(content):
        # Return plain string (no tools, so LLM returns string)
        if "respond with only 'true' or 'false'" in content.lower():
            return "true"
        # Return numeric score for quantitative criteria tests
        elif "provide a score" in content.lower():
            return "5.0"
        return "true"

    llm.side_effect = mock_call
    return llm


class TestQualitativeCriteria:
    """Test QualitativeCriteria for LLM-based verification."""

    def test_single_sample_pass(self, mock_llm):
        """Test qualitative criteria with single sample passing."""
        criteria = QualitativeCriteria(expression="Code is readable and well-structured", llm=mock_llm, num_samples=1)

        code = "result = 42"
        agent = Agent(tools=[])

        is_valid, error = criteria.verify(code, agent)

        assert is_valid is True
        assert error is None

    def test_single_sample_fail(self, mock_llm):
        """Test qualitative criteria with single sample returning false."""

        # Mock LLM to return "false"
        async def return_false(content):
            return "false"

        mock_llm.side_effect = return_false

        criteria = QualitativeCriteria(llm=mock_llm, expression="This requirement should fail")
        agent = Agent(name="test", llm=mock_llm, tools=[], instructions="test")
        code = "def bad_func(): pass"

        is_valid, error = criteria.verify(code, agent)

        assert not is_valid
        assert error is not None

    def test_invalid_llm_response(self, mock_llm):
        """Test handling of invalid LLM response."""

        # Mock LLM to return invalid response
        async def return_invalid(content):
            return "maybe"

        invalid_llm = MagicMock()
        invalid_llm.side_effect = return_invalid

        criteria = QualitativeCriteria(expression="Code is good", llm=invalid_llm, num_samples=1)

        code = "result = 42"
        agent = Agent(tools=[])

        is_valid, error = criteria.verify(code, agent)

        assert is_valid is False
        assert "invalid llm response" in error.lower()


class TestQuantitativeCriteria:
    """Test QuantitativeCriteria for LLM-based cost estimation."""

    def test_single_sample(self, mock_llm):
        """Test quantitative criteria with single sample."""
        criteria = QuantitativeCriteria(
            expression="How complex is this code? (0=simple, 10=very complex)",
            llm=mock_llm,
            min=0,
            max=10,
            num_samples=1,
        )

        code = "result = 42"
        agent = Agent(tools=[])

        cost = criteria.compute(code, agent)

        assert isinstance(cost, float)
        assert 0 <= cost <= 10

    def test_out_of_range_clamping(self, mock_llm):
        """Test that out-of-range scores are clamped."""

        # Mock LLM to return out-of-range value
        async def return_high(content):
            return "15.0"  # Above max

        oor_llm = MagicMock()
        oor_llm.side_effect = return_high

        criteria = QuantitativeCriteria(expression="Score this code", llm=oor_llm, min=0, max=10, num_samples=1)

        code = "result = 42"
        agent = Agent(tools=[])

        cost = criteria.compute(code, agent)

        assert cost == 10.0  # Clamped to max

    def test_invalid_response_uses_midpoint(self, mock_llm):
        """Test that invalid responses default to midpoint."""

        # Mock LLM to return invalid response
        async def return_invalid(content):
            return "not a number"

        invalid_llm = MagicMock()
        invalid_llm.side_effect = return_invalid

        criteria = QuantitativeCriteria(expression="Score this code", llm=invalid_llm, min=0, max=10, num_samples=1)

        code = "result = 42"
        agent = Agent(tools=[])

        cost = criteria.compute(code, agent)

        assert cost == 5.0  # Midpoint of 0-10


class TestStrategy:
    """Test Strategy configuration."""

    def test_default_strategy(self):
        """Test default strategy values."""
        strategy = Strategy()

        assert strategy.max_iterations == 3
        assert strategy.num_candidates == 3  # Changed from 1 to 3
        assert strategy.min_candidates_for_comparison == 1
        assert strategy.accept_cost_threshold is None
        assert strategy.compare_cost_threshold is None

    def test_custom_strategy(self):
        """Test custom strategy values."""
        strategy = Strategy(
            max_iterations=5,
            num_candidates=3,
            min_candidates_for_comparison=2,
            accept_cost_threshold=10.0,
            compare_cost_threshold=20.0,
        )

        assert strategy.max_iterations == 5
        assert strategy.num_candidates == 3
        assert strategy.min_candidates_for_comparison == 2
        assert strategy.accept_cost_threshold == 10.0
        assert strategy.compare_cost_threshold == 20.0


# Real LLM integration tests (skip if no API key)
@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_qualitative_criteria_real_llm():
    """Test qualitative criteria with real LLM (Groq for speed)."""
    llm = LLM("gpt-4.1-mini")  # This model supports json_schema

    criteria = QualitativeCriteria(expression="The code is syntactically valid Python", llm=llm, num_samples=1)

    valid_code = "result = 42"
    agent = Agent(tools=[])

    is_valid, error = criteria.verify(valid_code, agent)

    # Should pass - code is syntactically valid
    assert is_valid is True


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="GROQ_API_KEY not set")
@pytest.mark.asyncio
async def test_quantitative_criteria_real_llm():
    """Test quantitative criteria with real LLM (Groq for speed)."""
    llm = LLM("gpt-4.1-mini")  # This model supports json_schema

    criteria = QuantitativeCriteria(
        expression="Rate the code complexity from 0 (very simple) to 10 (very complex)",
        llm=llm,
        min=0,
        max=10,
        num_samples=1,
    )

    simple_code = "result = 42"
    agent = Agent(tools=[])

    cost = criteria.compute(simple_code, agent)

    # Should be a valid score in range
    assert isinstance(cost, float)
    assert 0 <= cost <= 10
    # Simple assignment should have low cost
    assert cost < 7.0


__all__ = [
    "TestQualitativeCriteria",
    "TestQuantitativeCriteria",
    "TestStrategy",
]
