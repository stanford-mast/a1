"""
Schema utility functions for handling large enums and schema transformations.

Provides:
- Enum detection and reduction using semantic similarity
- Schema cleaning for OpenAI strict mode
- Embedding caching for performance
"""

import asyncio
import copy
import logging
from typing import Any, Dict, List, Tuple
import numpy as np

from .em import _stringify_item, _pseudo_embed

logger = logging.getLogger(__name__)


def de_enum_large_enums(schema_dict: Dict[str, Any], threshold: int = 100) -> Dict[str, Any]:
    """
    Remove large enum constraints from a schema, replacing with unconstrained strings.
    
    This is useful for definition code generation where we don't want to inline
    huge enum lists in the prompt. The schema is modified in-place and also returned.
    
    Args:
        schema_dict: JSON schema dictionary (will be modified in-place)
        threshold: Enum size threshold for removal (default: 100)
        
    Returns:
        The modified schema_dict (same object, modified in-place)
    """
    def _sanitize(sch: dict):
        """Recursively de-enum large enums in schema."""
        if not isinstance(sch, dict):
            return
        
        if 'enum' in sch and isinstance(sch['enum'], list) and len(sch['enum']) > threshold:
            enum_count = len(sch['enum'])
            # Replace large enum with unconstrained string for definition code
            sch.pop('enum', None)
            sch['type'] = 'string'
            sch['description'] = (sch.get('description', '') + f' (originally enum of {enum_count} values - de-enumed for prompt)')
        
        # Recurse into properties and items
        for k in ('properties', 'items', 'additionalProperties'):
            if k in sch:
                if isinstance(sch[k], dict):
                    if k == 'properties':
                        for p in sch[k].values():
                            _sanitize(p)
                    else:
                        _sanitize(sch[k])
    
    _sanitize(schema_dict)
    return schema_dict


def detect_large_enums(schema_dict: Dict[str, Any], threshold: int = 100) -> List[Tuple[str, int]]:

    """
    Detect large enums in a JSON schema.
    
    Args:
        schema_dict: JSON schema dictionary
        threshold: Enum size threshold (default: 100)
        
    Returns:
        List of (path, size) tuples for each large enum found
    """
    large_enums = []
    
    def _check_node(node: Any, path: str = ""):
        if isinstance(node, dict):
            # Check if this is an enum definition
            if 'enum' in node and isinstance(node['enum'], list):
                enum_size = len(node['enum'])
                if enum_size > threshold:
                    large_enums.append((path, enum_size))
            
            # Recursively check nested structures
            for key, value in node.items():
                new_path = f"{path}.{key}" if path else key
                _check_node(value, new_path)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                _check_node(item, f"{path}[{i}]")
    
    _check_node(schema_dict)
    return large_enums


async def reduce_large_enums_in_tool_schemas(
    api_tools: List[Dict[str, Any]],
    context_text: str,
    runtime: Any = None,
    threshold: int = 100,
    target_size: int = 100,
    model: str = "text-embedding-3-small"
) -> None:
    """
    Reduce large enums in API tool schemas using async OpenAI embeddings (modifies in-place).
    
    Uses batched parallel embedding computation for performance:
    - Batches of 2048 items per OpenAI API request
    - Multiple batches processed in parallel with asyncio.gather
    - Caching to avoid recomputing embeddings
    
    Args:
        api_tools: List of tool schema dicts (OpenAI format)
        context_text: Text to compute similarity against
        runtime: Runtime instance for caching
        threshold: Enum size threshold for reduction (default: 100)
        target_size: Target size after reduction (default: 100)
        model: OpenAI embedding model (default: text-embedding-3-small)
    """
    from .runtime import get_runtime
    from .em import _get_embeddings_batch
    import os
    
    if runtime is None:
        runtime = get_runtime()
    
    # Check if we should use real embeddings
    api_key = os.environ.get("OPENAI_API_KEY")
    use_real_embeddings = api_key is not None and api_key.strip() != ""
    
    async def _recurse(node: dict):
        """Recursively find and reduce enums."""
        if not isinstance(node, dict):
            return
        
        if 'enum' in node and isinstance(node['enum'], list) and len(node['enum']) > threshold:
            orig_count = len(node['enum'])
            logger.info(f"Found large enum with {orig_count} values, reducing to top {target_size}...")
            
            # Prepare candidate strings
            candidates = [_stringify_item(x) for x in node['enum']]
            
            if use_real_embeddings:
                # Use async batch embeddings with OpenAI API (fast, parallel)
                logger.info(f"Computing embeddings for {len(candidates)} enum values using {model}...")
                
                # Compute embeddings for enum values and context in parallel
                enum_vecs, context_vecs = await asyncio.gather(
                    _get_embeddings_batch(candidates, model, runtime),
                    _get_embeddings_batch([context_text], model, runtime)
                )
                qvec = context_vecs[0]
                
                # Compute cosine similarities (vectors already normalized)
                sims = [(i, float(np.dot(qvec, vec))) for i, vec in enumerate(enum_vecs)]
            else:
                # Fallback to pseudo-embeddings (synchronous, deterministic)
                logger.warning("Using pseudo-embeddings (no OpenAI API key). This is slower for large enums.")
                vecs = [_pseudo_embed(c) for c in candidates]
                qvec = _pseudo_embed(context_text)
                
                # Compute similarities
                def _cos(a, b):
                    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                sims = [(i, _cos(qvec, v)) for i, v in enumerate(vecs)]
            
            # Sort by similarity and take top candidates
            sims.sort(key=lambda x: x[1], reverse=True)
            top_indices = [i for i, _ in sims[:target_size]]
            
            # Replace enum with top candidates (original values)
            node['enum'] = [node['enum'][i] for i in top_indices]
            logger.info(f"Reduced enum from {orig_count} to {len(node['enum'])} values (scores: {sims[0][1]:.3f} to {sims[target_size-1][1]:.3f})")
            node['description'] = (node.get('description', '') + f" (reduced to top {len(node['enum'])} via semantic similarity)")
        
        # Recurse into properties and items
        for k in ('properties', 'items', 'additionalProperties'):
            if k in node:
                if isinstance(node[k], dict):
                    if k == 'properties':
                        for p in node[k].values():
                            await _recurse(p)
                    else:
                        await _recurse(node[k])
    
    # Process each tool's parameters
    for tool_schema in api_tools:
        func = tool_schema.get('function') or {}
        params = func.get('parameters')
        if params:
            await _recurse(params)


def reduce_large_enums(
    schema_dict: Dict[str, Any],
    context_text: str,
    runtime: Any = None,
    threshold: int = 100,
    target_size: int = 100
) -> Dict[str, Any]:
    """
    Reduce large enums in schema using semantic similarity to context.
    
    Args:
        schema_dict: JSON schema dictionary (will be modified in-place)
        context_text: Context text to compute similarity against
        runtime: Runtime instance for caching embeddings (optional)
        threshold: Enum size threshold for reduction (default: 100)
        target_size: Target size after reduction (default: 100)
        
    Returns:
        Modified schema_dict with reduced enums
    """
    # Get or compute context embedding
    cache_key = f"context:{hash(context_text)}"
    if runtime and hasattr(runtime, 'enum_cache') and cache_key in runtime.enum_cache:
        context_emb = runtime.enum_cache[cache_key]
    else:
        context_emb = _pseudo_embed(context_text)
        if runtime and hasattr(runtime, 'enum_cache'):
            runtime.enum_cache[cache_key] = context_emb
    
    def _reduce_node(node: Any) -> Any:
        """Recursively reduce large enums in schema."""
        if isinstance(node, dict):
            # Check if this is an enum definition
            if 'enum' in node and isinstance(node['enum'], list):
                enum_values = node['enum']
                
                if len(enum_values) > threshold:
                    logger.info(f"Found large enum with {len(enum_values)} values, reducing to top {target_size}...")
                    
                    # Get or compute embeddings for enum values
                    enum_embs = []
                    for val in enum_values:
                        val_str = _stringify_item(val)
                        cache_key = f"enum:{hash(val_str)}"
                        
                        if runtime and hasattr(runtime, 'enum_cache') and cache_key in runtime.enum_cache:
                            enum_emb = runtime.enum_cache[cache_key]
                        else:
                            enum_emb = _pseudo_embed(val_str)
                            if runtime and hasattr(runtime, 'enum_cache'):
                                runtime.enum_cache[cache_key] = enum_emb
                        
                        enum_embs.append(enum_emb)
                    
                    # Compute cosine similarities
                    similarities = []
                    for val, enum_emb in zip(enum_values, enum_embs):
                        # Cosine similarity = dot product of normalized vectors
                        sim = np.dot(context_emb, enum_emb) / (
                            np.linalg.norm(context_emb) * np.linalg.norm(enum_emb)
                        )
                        similarities.append((val, sim))
                    
                    # Sort by similarity (descending) and take top N
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    reduced_values = [val for val, _ in similarities[:target_size]]
                    
                    logger.info(
                        f"Reduced enum from {len(enum_values)} to {len(reduced_values)} values "
                        f"(similarity range: {similarities[0][1]:.3f} to {similarities[target_size-1][1]:.3f})"
                    )
                    
                    # Update the enum in-place
                    node['enum'] = reduced_values
            
            # Recursively process nested objects
            return {k: _reduce_node(v) for k, v in node.items()}
        
        elif isinstance(node, list):
            return [_reduce_node(item) for item in node]
        
        return node
    
    return _reduce_node(schema_dict)


def clean_schema_for_openai(schema_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean schema for OpenAI strict mode requirements.
    
    OpenAI strict mode requires:
    - $ref objects to only have $ref key (no additional properties like 'description')
    - additionalProperties set to false at all levels for object types
    
    Args:
        schema_dict: JSON schema dictionary
        
    Returns:
        Cleaned schema dictionary
    """
    def _clean_node(node: Any) -> Any:
        if isinstance(node, dict):
            # If has $ref, keep ONLY $ref (remove description, title, etc.)
            if '$ref' in node:
                return {'$ref': node['$ref']}
            
            # Recursively clean nested objects first
            cleaned = {k: _clean_node(v) for k, v in node.items()}
            
            # Add additionalProperties: false to all object types
            # This is required by OpenAI's structured output API
            if cleaned.get('type') == 'object' and 'additionalProperties' not in cleaned:
                cleaned['additionalProperties'] = False
            
            return cleaned
        
        elif isinstance(node, list):
            return [_clean_node(item) for item in node]
        
        return node
    
    return _clean_node(schema_dict)


def prepare_response_format(
    schema_dict: Dict[str, Any],
    name: str,
    strict: bool = True
) -> Dict[str, Any]:
    """
    Prepare response_format for OpenAI structured outputs.
    
    Args:
        schema_dict: Cleaned JSON schema dictionary
        name: Name for the schema
        strict: Whether to use strict mode (default: True)
        
    Returns:
        Response format dict ready for OpenAI API
    """
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema_dict,
            "strict": strict
        }
    }
