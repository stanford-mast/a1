"""Skill and SkillSet models for reusable knowledge units."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Skill(BaseModel):
    """
    A reusable skill/knowledge unit with code and module dependencies.

    Skills encapsulate domain-specific knowledge, patterns, and best practices
    that can be selectively loaded into agents. Each skill includes content
    (code snippets, examples, instructions) and specifies required Python modules.

    Attributes:
        name: Unique identifier for the skill
        description: Human-readable description of what this skill provides
        content: The actual skill content (code, examples, documentation, instructions)
        modules: List of Python module names that this skill depends on
    """

    name: str
    description: str
    content: str
    modules: list[str] = Field(default_factory=list, description="Python modules required for this skill")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    async def from_url(
        cls,
        urls: str | list[str],
        name: str,
        description: str,
        chunk_size: int = 2000,
        llm: Any | None = None,
        instructions: str | None = None,
        modules: list[str] | None = None,
    ) -> "Skill":
        """
        Load skill content from one or more URLs using crawl4ai and LLM summarization.

        Uses crawl4ai to fetch and parse content from URLs, chunks the content,
        and uses an LLM to generate a concise skill summary and extract relevant
        Python modules. This enables creating skills from web documentation,
        articles, tutorials, and other online resources.

        Args:
            urls: Single URL or list of URLs to load content from
            name: Name for the generated skill
            description: Description of what the skill provides
            chunk_size: Size of content chunks for LLM processing (default: 2000 chars)
            llm: Optional LLM tool to use for summarization (uses default if not provided)
            instructions: Optional specific instructions for skill generation
            modules: Optional list of Python modules (auto-detected if not provided)

        Returns:
            Skill with content from the URLs

        Raises:
            ImportError: If crawl4ai is not installed
            ValueError: If URL loading fails

        Note:
            Requires: pip install crawl4ai
            The LLM used for summarization should be fast/cheap (e.g., gpt-3.5-turbo)
        """
        try:
            from crawl4ai import AsyncWebCrawler
        except ImportError:
            raise ImportError("crawl4ai is required for Skill.from_url. Install it with: pip install crawl4ai")

        # Normalize urls to list
        if isinstance(urls, str):
            urls = [urls]

        # Use default LLM if not provided
        if llm is None:
            from ..llm import LLM

            llm = LLM("gpt-4.1-mini")

        # Fetch content from URLs using crawl4ai
        crawler = AsyncWebCrawler()
        all_content = []

        for url in urls:
            try:
                result = await crawler.arun(url)
                if result.success:
                    all_content.append(f"# From: {url}\n{result.markdown}")
                else:
                    raise ValueError(f"Failed to crawl {url}: {result.error}")
            except Exception as e:
                raise ValueError(f"Error crawling {url}: {str(e)}")

        full_content = "\n\n".join(all_content)

        # Chunk content for LLM processing
        # chunks = [full_content[i : i + chunk_size] for i in range(0, len(full_content), chunk_size)]

        # Use LLM to generate skill content
        summarization_prompt = f"""
Given the following content from URL(s), create a concise skill summary that:
1. Captures the key concepts and best practices
2. Provides practical examples or code snippets
3. Is organized and easy to reference
4. Lists any Python modules that would be needed (if applicable)

Content to summarize:
{full_content[:5000]}  # Use first 5000 chars to save tokens

{f"Additional instructions: {instructions}" if instructions else ""}

Format the response as a markdown skill guide.
"""

        # Get LLM response (assuming llm is a Tool)
        summary_response = await llm(content=summarization_prompt)

        # Extract content from response
        if hasattr(summary_response, "content"):
            skill_content = summary_response.content
        else:
            skill_content = str(summary_response)

        # Auto-detect modules if not provided
        detected_modules = modules or []
        if not modules:
            # Simple heuristic: look for common module names in content
            common_modules = [
                "pandas",
                "numpy",
                "requests",
                "beautifulsoup4",
                "sqlalchemy",
                "flask",
                "django",
                "sklearn",
                "pytorch",
                "tensorflow",
                "matplotlib",
                "seaborn",
                "plotly",
                "asyncio",
                "aiohttp",
            ]
            for module in common_modules:
                if module.lower() in full_content.lower():
                    detected_modules.append(module)

        return cls(name=name, description=description, content=skill_content, modules=detected_modules)


class SkillSet(BaseModel):
    """
    A collection of related skills.

    SkillSets group multiple skills together for organizational purposes,
    allowing agents to have access to collections of domain-specific knowledge.

    Attributes:
        name: Unique identifier for the skillset
        description: Human-readable description of the skillset
        skills: List of skills in this collection
    """

    name: str
    description: str
    skills: list[Skill]

    model_config = ConfigDict(arbitrary_types_allowed=True)


__all__ = ["Skill", "SkillSet"]
