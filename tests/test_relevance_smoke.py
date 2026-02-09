import os
import numpy as np

from src.data_loader import JobDataLoader
from src.search_engine import JobSearchEngine


def _tokenize(text):
    return set("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _count_hits(titles, expected_tokens, expected_phrases=None):
    expected_phrases = expected_phrases or []
    hits = 0
    for title in titles:
        tl = title.lower()
        tokens = _tokenize(tl)
        if any(tok in tokens for tok in expected_tokens):
            hits += 1
            continue
        if any(phrase in tl for phrase in expected_phrases):
            hits += 1
    return hits


def _load_engine():
    os.environ["OFFLINE_MODE"] = "1"
    os.environ["QUERY_CACHE_WRITE"] = "0"
    os.environ["QUERY_CACHE_PREWARM"] = "0"
    loader = JobDataLoader("data/jobs.jsonl")
    loader.load(fast_demo=True, fast_n=int(os.getenv("FAST_DEMO_N", "20000")))
    jobs = loader.get_jobs()
    embeddings = loader.get_embeddings()
    return JobSearchEngine(jobs, embeddings, debug_cache=False, emoji_ok=False)


def _top_titles(engine, query, top_k=10):
    results = engine.search(query, top_k=top_k)
    titles = []
    for job in results:
        meta = job.get("_meta", {}) or {}
        titles.append(meta.get("title") or job.get("title") or "")
    return titles


def test_relevance_smoke_data_science():
    engine = _load_engine()
    titles = _top_titles(engine, "data science jobs", top_k=10)
    expected_tokens = {"data", "science", "scientist", "ml", "analytics", "engineer"}
    expected_phrases = {"machine learning"}
    hits = _count_hits(titles, expected_tokens, expected_phrases)
    assert hits >= 6


def test_relevance_smoke_ml_engineer():
    engine = _load_engine()
    titles = _top_titles(engine, "senior ML engineer", top_k=10)
    expected_tokens = {"ml", "engineer", "platform", "mlops", "llmops", "sre", "software"}
    expected_phrases = {"machine learning"}
    hits = _count_hits(titles, expected_tokens, expected_phrases)
    assert hits >= 6


def test_relevance_smoke_product_manager():
    engine = _load_engine()
    titles = _top_titles(engine, "product manager", top_k=10)
    expected_tokens = {"product"}
    hits = _count_hits(titles, expected_tokens)
    assert hits >= 6
