import numpy as np
import pickle

from src.query_cache import QueryEmbeddingCache


def test_query_cache_api_evictions_and_requests(tmp_path):
    cache_path = tmp_path / "cache.pkl"
    cache = QueryEmbeddingCache(max_size=3, path=str(cache_path), enable_disk=False)
    stats = cache.get_stats()
    assert stats["current_size"] == 0
    assert stats["total_requests"] == 0

    cache.put("data science", np.random.rand(1536))
    cache.put("python developer", np.random.rand(1536))
    cache.put("ML engineer", np.random.rand(1536))
    stats = cache.get_stats()
    assert stats["current_size"] == 3
    assert stats["evictions"] == 0
    assert stats["total_requests"] == 0

    cache.put("data science", np.random.rand(1536))
    stats = cache.get_stats()
    assert stats["evictions"] == 0
    assert stats["total_requests"] == 0

    cache.put("remote job", np.random.rand(1536))
    stats = cache.get_stats()
    assert stats["current_size"] == 3
    assert stats["evictions"] == 1
    assert stats["total_requests"] == 0

    vec, source = cache.get("data science")
    assert vec is not None
    stats = cache.get_stats()
    assert stats["hits"] == 1
    assert stats["total_requests"] == 1

    vec, source = cache.get("missing")
    assert vec is None
    stats = cache.get_stats()
    assert stats["misses"] == 1
    assert stats["total_requests"] == 2


def test_query_cache_trim_on_load_not_eviction(tmp_path):
    path = tmp_path / "cache.pkl"
    items = [(f"k{i}", np.random.rand(1536).astype(np.float32)) for i in range(5)]
    with open(path, "wb") as f:
        pickle.dump(items, f)

    cache = QueryEmbeddingCache(max_size=3, path=str(path), enable_disk=True)
    cache.load()
    stats = cache.get_stats()
    assert stats["current_size"] == 3
    assert stats["evictions"] == 0
