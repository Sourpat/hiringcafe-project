import os
import pickle
import numpy as np

from src.search_engine import JobSearchEngine
from src.query_cache import (
    QueryEmbeddingCache,
    normalize_query_key,
    query_cache_prewarm_write,
    query_cache_write,
)


def _make_engine(tmp_path, dim=3):
    embeddings = {
        "explicit": np.zeros((1, dim), dtype=np.float32),
        "inferred": np.zeros((1, dim), dtype=np.float32),
        "company": np.zeros((1, dim), dtype=np.float32),
        "_normalized": True,
    }
    engine = JobSearchEngine(jobs=[{"_meta": {}}], embeddings=embeddings)
    engine._embedding_dim = dim
    engine._query_cache = QueryEmbeddingCache(
        max_size=engine._query_cache.max_size,
        path=str(tmp_path / "query_vec_cache.pkl"),
        enable_disk=query_cache_write(),
        embedding_dim=dim,
        debug=False,
    )
    engine._prewarm_write = query_cache_prewarm_write()
    engine._prewarm_keys = set()
    engine._prewarm_cache = {}
    return engine


def test_cache_key_normalization(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("QUERY_CACHE_PREWARM_WRITE", "1")
    engine = _make_engine(tmp_path, dim=3)

    engine.embed_query("Senior ML Engineer")
    engine.embed_query("  senior   ml engineer ")

    assert engine.get_query_cache_size() == 1
    assert normalize_query_key("Senior ML Engineer") in engine.get_query_cache_keys_mru()


def test_prewarm_no_disk_write(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("QUERY_CACHE_PREWARM_WRITE", "0")
    monkeypatch.setenv("QUERY_CACHE_WRITE", "1")
    engine = _make_engine(tmp_path, dim=3)

    engine.mark_prewarm_keys(["data science jobs"])
    engine.embed_query("data science jobs")
    engine._query_cache.save()

    with open(engine._query_cache.path, "rb") as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        entries = len(data)
    else:
        entries = len(list(data))

    assert entries == 0


def test_lru_trim_and_order(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("QUERY_CACHE_PREWARM_WRITE", "1")
    monkeypatch.setenv("QUERY_CACHE_WRITE", "0")
    engine = _make_engine(tmp_path, dim=3)
    engine._query_cache.max_size = 5

    for i in range(12):
        engine.embed_query(f"query {i}")

    keys = list(reversed(engine.get_query_cache_keys_mru()))
    assert len(keys) == 5
    assert keys == ["query 7", "query 8", "query 9", "query 10", "query 11"]


def test_persistence_disk_hit(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("QUERY_CACHE_PREWARM_WRITE", "1")
    monkeypatch.setenv("QUERY_CACHE_WRITE", "1")

    engine = _make_engine(tmp_path, dim=3)
    engine.embed_query("Senior ML Engineer")
    engine._query_cache.save()

    engine2 = _make_engine(tmp_path, dim=3)
    vec = engine2.embed_query(" senior   ml engineer ")
    assert isinstance(vec, np.ndarray)
    assert engine2._last_embed_source in ("disk", "mem")
