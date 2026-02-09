import numpy as np
import pytest

from src.query_cache import QueryEmbeddingCache
from src.search_engine import JobSearchEngine


def test_single_cache_instance_injected():
    jobs = [
        {"title": "Data Scientist", "company": "Acme", "_meta": {"workplace_type": "Remote", "is_remote": True}},
        {"title": "ML Engineer", "company": "Beta", "_meta": {"workplace_type": "Hybrid", "is_remote": False}},
    ]
    embeddings = {"explicit": np.random.rand(2, 4).astype(np.float32)}
    cache = QueryEmbeddingCache(max_size=5)

    engine = JobSearchEngine(
        jobs,
        embeddings,
        debug_cache=False,
        emoji_ok=False,
        explain=False,
        query_cache=cache,
    )

    assert engine._query_cache is cache


def test_query_cache_required():
    jobs = [{"title": "Data Scientist", "company": "Acme", "_meta": {}}]
    embeddings = {"explicit": np.random.rand(1, 4).astype(np.float32)}

    with pytest.raises(ValueError):
        JobSearchEngine(jobs, embeddings, query_cache=None)
