"""
Embedding (EM) tool for semantic similarity computation.

Provides:
- EM(): Factory function to create embedding tool  
- Real OpenAI embeddings API with batching, caching, and retry logic
- Pseudo-embeddings fallback (deterministic hashing) when no API key
- Helper functions for text stringification and cosine similarity
"""
from typing import Any, List, Tuple, Dict, Union
import math
import json
import hashlib
import logging
import numpy as np

from pydantic import BaseModel, Field, create_model
from .models import Tool
from .runtime import get_runtime

logger = logging.getLogger(__name__)


def _stringify_item(x: Any) -> str:
    """Convert any item to a JSON string for embedding."""
    try:
        return json.dumps(x, default=lambda o: getattr(o, '__dict__', str(o)), sort_keys=True)
    except Exception:
        return str(x)


def _pseudo_embed(text: str, dim: int = 512) -> np.ndarray:
    """
    Deterministic pseudo-embedding based on SHA256 digest.
    
    Returns normalized vector in R^dim. Not a real semantic embedding,
    but deterministic and fast for testing/fallback when no API key.
    
    Args:
        text: Text to embed
        dim: Embedding dimension (default: 512)
        
    Returns:
        Normalized numpy array of shape (dim,)
    """
    # Use multiple hashes with different salts to get enough entropy
    vals = []
    for offset in range((dim + 31) // 32):  # 32 floats per hash
        salt = f"{offset}:{text}"
        h = hashlib.sha256(salt.encode('utf-8')).digest()
        for i in range(min(32, dim - len(vals))):
            byte = h[i % len(h)]
            vals.append((byte / 255.0) * 2.0 - 1.0)  # Map to [-1, 1]
    
    # Convert to numpy and normalize
    vec = np.array(vals[:dim], dtype=np.float32)
    vec = vec / np.linalg.norm(vec)
    return vec


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Assumes vectors are already normalized (for efficiency).
    If not normalized, result is still valid cosine similarity.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity in [-1, 1]
    """
    return float(np.dot(a, b))


def EM(model: str = "text-embedding-3-small") -> Tool:
    """
    Create an EM (Embedding) tool for computing semantic similarity.
    
    Computes cosine similarity between items using embeddings.
    Supports batching, caching, parallelization, and retry logic.
    
    Args:
        model: OpenAI embedding model name (default: text-embedding-3-small)
               Supported: text-embedding-3-small (1536-dim), text-embedding-3-large (3072-dim)
    
    Returns:
        Tool that computes pairwise similarities between two lists
    
    Example:
        similarities = await EM_tool(
            items_a=["apple", "banana"],
            items_b=["fruit", "vegetable", "car"]
        )
        # Returns: [[0.85, 0.45, 0.1], [0.9, 0.4, 0.05]]
    """
    import asyncio
    from openai import AsyncOpenAI
    from typing import List, Any
    import os
    
    # Input schema: two lists of items to compare
    InputSchema = create_model(
        "EMInput",
        items_a=(List[Any], Field(description="First list of items to embed")),
        items_b=(List[Any], Field(description="Second list of items to compare against"))
    )
    
    # Output schema: 2D array of similarity scores
    OutputSchema = create_model(
        "EMOutput",
        similarities=(List[List[float]], Field(description="2D array: similarities[i][j] = similarity between items_a[i] and items_b[j]"))
    )
    
    async def execute(items_a: List[Any], items_b: List[Any]) -> Dict[str, Any]:
        """
        Compute pairwise cosine similarities between two lists of items.
        
        Uses OpenAI embeddings API with:
        - Batch processing (up to 2048 items per request)
        - Caching in runtime.enum_cache
        - Parallel requests for large batches
        - Retry logic with exponential backoff
        """
        from .runtime import get_runtime
        import numpy as np
        
        runtime = get_runtime()
        
        # Convert all items to strings
        str_items_a = [_stringify_item(item) for item in items_a]
        str_items_b = [_stringify_item(item) for item in items_b]
        
        # Use pseudo-embeddings if no API key (fallback)
        api_key = os.environ.get("OPENAI_API_KEY")
        use_real_embeddings = api_key is not None and api_key.strip() != ""
        
        if not use_real_embeddings:
            logger.warning(f"No OPENAI_API_KEY found, using pseudo-embeddings (deterministic hashing)")
            # Fallback to pseudo-embeddings
            vecs_a = [_pseudo_embed(s) for s in str_items_a]
            vecs_b = [_pseudo_embed(s) for s in str_items_b]
        else:
            # Get embeddings with caching and batching
            vecs_a = await _get_embeddings_batch(str_items_a, model, runtime)
            vecs_b = await _get_embeddings_batch(str_items_b, model, runtime)
        
        # Compute cosine similarities (already normalized, so just dot product)
        similarities = []
        for vec_a in vecs_a:
            row = []
            for vec_b in vecs_b:
                sim = _cosine_similarity(vec_a, vec_b)
                row.append(float(sim))
            similarities.append(row)
        
        return {"similarities": similarities}
    
    return Tool(
        name="EM",
        description=f"Compute semantic similarity between items using {model} embeddings",
        input_schema=InputSchema,
        output_schema=OutputSchema,
        execute=execute
    )


async def _get_embeddings_batch(
    texts: List[str],
    model: str,
    runtime: Any,
    parallel_strategy: Any = None
) -> List[np.ndarray]:
    """
    Get embeddings for a list of texts with caching, parallel batching, and adaptive rate limit handling.
    
    Uses ParallelStrategy to control:
    - Chunk size (default: 2048 items per API request)
    - Max parallel chunks (default: 16 concurrent requests)
    - Adaptive concurrency on rate limit errors (halves concurrency, exponential backoff)
    
    Args:
        texts: List of text strings to embed
        model: OpenAI embedding model name
        runtime: Runtime instance for caching
        parallel_strategy: ParallelStrategy for batch control (default: chunk_size=2048, max_parallel=16)
        
    Returns:
        List of embedding vectors (numpy arrays)
    """
    import asyncio
    from openai import AsyncOpenAI
    import numpy as np
    from .models import ParallelStrategy
    
    # Use provided strategy or create default
    if parallel_strategy is None:
        parallel_strategy = ParallelStrategy(
            chunk_size=2048,
            max_parallel_chunks=16,
            max_iterations=3
        )
    
    # Check cache first
    cache = runtime.enum_cache if hasattr(runtime, 'enum_cache') else {}
    results = [None] * len(texts)
    uncached_indices = []
    uncached_texts = []
    
    for i, text in enumerate(texts):
        cache_key = f"emb:{model}:{hash(text)}"
        if cache_key in cache:
            results[i] = cache[cache_key]
        else:
            uncached_indices.append(i)
            uncached_texts.append(text)
    
    # If everything cached, return immediately
    if not uncached_texts:
        return results
    
    logger.info(f"Fetching {len(uncached_texts)} embeddings from {model} (cached: {len(texts) - len(uncached_texts)})")
    
    # Split into chunks based on strategy
    client = AsyncOpenAI()
    chunk_size = parallel_strategy.chunk_size
    batches = []
    for i in range(0, len(uncached_texts), chunk_size):
        batch = uncached_texts[i:i + chunk_size]
        batches.append(batch)
    
    logger.info(f"Processing {len(batches)} batches of up to {chunk_size} items each with adaptive parallelism...")
    
    # Adaptive rate limit handling with exponential backoff
    current_parallelism = parallel_strategy.max_parallel_chunks
    all_embeddings = []
    batch_idx = 0
    
    while batch_idx < len(batches):
        # Process next chunk of batches in parallel
        end_idx = min(batch_idx + current_parallelism, len(batches))
        current_batches = batches[batch_idx:end_idx]
        
        logger.info(f"  Processing batches {batch_idx + 1}-{end_idx} with parallelism={current_parallelism}")
        
        async def _fetch_batch_with_retry(local_idx: int, batch_texts: List[str]) -> List[np.ndarray]:
            """Fetch embeddings for a batch with exponential backoff retry."""
            max_retries = parallel_strategy.max_iterations
            
            for attempt in range(max_retries):
                try:
                    response = await client.embeddings.create(
                        model=model,
                        input=batch_texts,
                        encoding_format="float"
                    )
                    # Extract embeddings and convert to numpy
                    embeddings = []
                    for item in response.data:
                        emb = np.array(item.embedding, dtype=np.float32)
                        # Normalize to unit length (cosine similarity = dot product)
                        emb = emb / np.linalg.norm(emb)
                        embeddings.append(emb)
                    
                    return embeddings
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = 'rate' in error_str or 'limit' in error_str or '429' in error_str
                    
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        logger.warning(f"Embedding API error batch {batch_idx + local_idx + 1} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"Embedding API failed after {max_retries} attempts: {e}")
                        # Re-raise to trigger adaptive backoff
                        raise
        
        try:
            # Fetch current batch group in parallel
            batch_results = await asyncio.gather(*[
                _fetch_batch_with_retry(i, batch) 
                for i, batch in enumerate(current_batches)
            ])
            
            # Success - add to results and move forward
            for batch_embs in batch_results:
                all_embeddings.extend(batch_embs)
            
            batch_idx = end_idx
            
            # Log progress
            if batch_idx % 10 == 0 or batch_idx == len(batches):
                logger.info(f"  Completed {batch_idx}/{len(batches)} batches")
            
        except Exception as e:
            # Rate limit or other error - reduce parallelism and retry
            error_str = str(e).lower()
            is_rate_limit = 'rate' in error_str or 'limit' in error_str or '429' in error_str
            
            if is_rate_limit and current_parallelism > 1:
                # Halve parallelism for rate limits
                current_parallelism = max(1, current_parallelism // 2)
                logger.warning(f"Rate limit detected, reducing parallelism to {current_parallelism}")
                await asyncio.sleep(2)  # Brief pause before retry
                # Don't advance batch_idx - retry same batches with lower parallelism
            else:
                # Non-rate-limit error or already at minimum parallelism - fail
                logger.error(f"Failed to process batches {batch_idx + 1}-{end_idx}: {e}")
                raise
    
    # Store in cache and results array
    for i, emb in zip(uncached_indices, all_embeddings):
        text = texts[i]
        cache_key = f"emb:{model}:{hash(text)}"
        cache[cache_key] = emb
        results[i] = emb
    
    logger.info(f"Successfully fetched and cached {len(all_embeddings)} new embeddings")
    
    return results
