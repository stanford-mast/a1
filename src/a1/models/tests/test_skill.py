import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from a1 import Runtime, Skill, set_runtime


class TestSkillFromUrlCache:
    """Test caching functionality for Skill.from_url."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for skill caching."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.mark.asyncio
    async def test_from_url_cache_true(self, temp_cache_dir):
        """Test that cache=True creates and uses cache file."""
        # Create runtime with temp cache dir
        runtime = Runtime(cache_dir=temp_cache_dir)
        set_runtime(runtime)

        # Mock crawl4ai and LLM
        mock_crawler_result = MagicMock()
        mock_crawler_result.success = True
        mock_crawler_result.markdown = "# Test Content\nThis is test content from URL."

        mock_llm_response = MagicMock()
        mock_llm_response.content = "# Test Skill\nThis is a test skill summary."

        with patch("crawl4ai.AsyncWebCrawler") as mock_crawler_class, patch("a1.models.skill.LLM") as mock_llm_class:
            # Setup mocks
            mock_crawler = AsyncMock()
            mock_crawler.arun = AsyncMock(return_value=mock_crawler_result)
            mock_crawler_class.return_value = mock_crawler

            mock_llm_instance = MagicMock()
            mock_llm_tool = AsyncMock(return_value=mock_llm_response)
            mock_llm_instance.tool = mock_llm_tool
            mock_llm_class.return_value = mock_llm_instance

            # First call - should create cache
            skill1 = await Skill.from_url(
                urls="https://example.com/test",
                name="test_skill",
                description="Test skill description",
                cache=True,
            )

            # Verify cache file was created
            cache_dir = Path(temp_cache_dir)
            cache_files = list(cache_dir.glob("skill_*.json"))
            assert len(cache_files) == 1, "Cache file should be created"

            # Verify cache file contains correct data
            cache_file = cache_files[0]
            cached_data = json.loads(cache_file.read_text())
            assert cached_data["name"] == "test_skill"
            assert cached_data["description"] == "Test skill description"
            assert cached_data["content"] == "# Test Skill\nThis is a test skill summary."
            assert cached_data["modules"] == []

            # Verify LLM was called (not cached)
            assert mock_llm_tool.call_count == 1, "LLM should be called on first request"

            # Reset mock call count
            mock_llm_tool.reset_mock()

            # Second call with same parameters - should use cache
            skill2 = await Skill.from_url(
                urls="https://example.com/test",
                name="test_skill",
                description="Test skill description",
                cache=True,
            )

            # Verify LLM was NOT called (used cache)
            assert mock_llm_tool.call_count == 0, "LLM should not be called when using cache"

            # Verify skills are the same
            assert skill1.name == skill2.name
            assert skill1.description == skill2.description
            assert skill1.content == skill2.content
            assert skill1.modules == skill2.modules

    @pytest.mark.asyncio
    async def test_from_url_cache_false(self, temp_cache_dir):
        """Test that cache=False does not use cache even if it exists."""
        # Create runtime with temp cache dir
        runtime = Runtime(cache_dir=temp_cache_dir)
        set_runtime(runtime)

        # Mock crawl4ai and LLM
        mock_crawler_result = MagicMock()
        mock_crawler_result.success = True
        mock_crawler_result.markdown = "# Test Content\nThis is test content from URL."

        mock_llm_response = MagicMock()
        mock_llm_response.content = "# Test Skill\nThis is a test skill summary."

        with patch("crawl4ai.AsyncWebCrawler") as mock_crawler_class, patch("a1.models.skill.LLM") as mock_llm_class:
            # Setup mocks
            mock_crawler = AsyncMock()
            mock_crawler.arun = AsyncMock(return_value=mock_crawler_result)
            mock_crawler_class.return_value = mock_crawler

            mock_llm_instance = MagicMock()
            mock_llm_tool = AsyncMock(return_value=mock_llm_response)
            mock_llm_instance.tool = mock_llm_tool
            mock_llm_class.return_value = mock_llm_instance

            # First call with cache=True - creates cache
            await Skill.from_url(
                urls="https://example.com/test2",
                name="test_skill2",
                description="Test skill description 2",
                cache=True,
            )

            # Verify cache file exists
            cache_dir = Path(temp_cache_dir)
            cache_files = list(cache_dir.glob("skill_*.json"))
            assert len(cache_files) == 1, "Cache file should exist"

            # Reset mock call count
            mock_llm_tool.reset_mock()

            # Second call with cache=False - should NOT use cache
            await Skill.from_url(
                urls="https://example.com/test2",
                name="test_skill2",
                description="Test skill description 2",
                cache=False,
            )

            # Verify LLM was called (cache was ignored)
            assert mock_llm_tool.call_count == 1, "LLM should be called when cache=False"

    @pytest.mark.asyncio
    async def test_from_url_cache_key_consistency(self, temp_cache_dir):
        """Test that cache key is consistent for same parameters."""
        runtime = Runtime(cache_dir=temp_cache_dir)
        set_runtime(runtime)

        # Generate cache keys with same parameters
        key1 = Skill._get_cache_key(
            urls=["https://example.com"],
            name="test",
            description="test desc",
            chunk_size=2000,
            instructions=None,
            modules=None,
        )

        key2 = Skill._get_cache_key(
            urls=["https://example.com"],
            name="test",
            description="test desc",
            chunk_size=2000,
            instructions=None,
            modules=None,
        )

        # Same parameters should produce same key
        assert key1 == key2, "Cache keys should be identical for same parameters"

        # Different parameters should produce different key
        key3 = Skill._get_cache_key(
            urls=["https://example.com"],
            name="test",
            description="different desc",  # Different description
            chunk_size=2000,
            instructions=None,
            modules=None,
        )

        assert key1 != key3, "Cache keys should differ for different parameters"

    @pytest.mark.asyncio
    async def test_from_url_cache_different_parameters(self, temp_cache_dir):
        """Test that different parameters create different cache entries."""
        runtime = Runtime(cache_dir=temp_cache_dir)
        set_runtime(runtime)

        mock_crawler_result = MagicMock()
        mock_crawler_result.success = True
        mock_crawler_result.markdown = "# Test Content"

        mock_llm_response = MagicMock()
        mock_llm_response.content = "# Test Skill"

        with patch("crawl4ai.AsyncWebCrawler") as mock_crawler_class, patch("a1.models.skill.LLM") as mock_llm_class:
            mock_crawler = AsyncMock()
            mock_crawler.arun = AsyncMock(return_value=mock_crawler_result)
            mock_crawler_class.return_value = mock_crawler

            mock_llm_instance = MagicMock()
            mock_llm_tool = AsyncMock(return_value=mock_llm_response)
            mock_llm_instance.tool = mock_llm_tool
            mock_llm_class.return_value = mock_llm_instance

            # Create skill with first set of parameters
            await Skill.from_url(
                urls="https://example.com/test1",
                name="skill1",
                description="Description 1",
                cache=True,
            )

            # Create skill with different parameters
            await Skill.from_url(
                urls="https://example.com/test2",
                name="skill2",
                description="Description 2",
                cache=True,
            )

            # Should have 2 cache files
            cache_dir = Path(temp_cache_dir)
            cache_files = list(cache_dir.glob("skill_*.json"))
            assert len(cache_files) == 2, "Should have 2 separate cache files for different parameters"
