from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import sys
import time
import numpy as np
from contextlib import contextmanager

if os.getenv("PROBE_SCHEMA") == "1":
    from src.schema_probe import main as schema_probe_main
    schema_probe_main()
    raise SystemExit(0)
from src.data_loader import JobDataLoader
from src.search_engine import JobSearchEngine
from src.context import SearchContext
from src.token_tracker import tracker

EMOJI_OK = True
DEFAULT_PREWARM_QUERIES = [
    "data science jobs",
    "senior ML engineer",
    "startup founder roles",
    "product manager",
]

def env_bool(name, default=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip().lower()
    if value == "":
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default

def env_int(name, default):
    raw = os.getenv(name)
    if raw is None:
        return default
    value = str(raw).strip()
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default

@contextmanager
def time_block(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        prefix = "⏱️  " if EMOJI_OK else "TIME "
        print(f"{prefix}{label}: {elapsed:.2f}s")

def main():
    global EMOJI_OK
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        EMOJI_OK = True
    except Exception:
        EMOJI_OK = False

    raw_prewarm = os.getenv("QUERY_CACHE_PREWARM")
    raw_write = os.getenv("QUERY_CACHE_WRITE")
    raw_prewarm_write = os.getenv("QUERY_CACHE_PREWARM_WRITE")
    raw_prewarm_n = os.getenv("QUERY_CACHE_PREWARM_N")

    parser = argparse.ArgumentParser(description="HiringCafe Job Search Demo")
    parser.add_argument("--offline", action="store_true", help="Run without OpenAI calls")
    parser.add_argument("--reset-cache", action="store_true", help="Delete query cache file before running")
    parser.add_argument("--stress-cache", action="store_true", help="Stress test query cache eviction")
    parser.add_argument("--stress-n", type=int, default=12, help="Number of stress queries")
    parser.add_argument("--print-cache-keys", action="store_true", help="Print query cache keys (MRU->LRU)")
    parser.add_argument("--debug-cache", action="store_true", help="Print query cache debug per query")
    args = parser.parse_args()

    if args.offline:
        os.environ["OFFLINE_MODE"] = "1"
        if os.getenv("QUERY_CACHE_PREWARM") is None:
            os.environ["QUERY_CACHE_PREWARM"] = "1"
        if os.getenv("QUERY_CACHE_PREWARM_WRITE") is None:
            os.environ["QUERY_CACHE_PREWARM_WRITE"] = "1"
        if os.getenv("QUERY_CACHE_WRITE") is None:
            os.environ["QUERY_CACHE_WRITE"] = "0"
        print("OFFLINE MODE ENABLED")
    else:
        print("ONLINE MODE ENABLED")
        api_key = (
            os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or ""
        ).strip().strip('"').strip("'")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... or OPENAI_KEY=... and rerun."
            )

    fast_demo = os.getenv("FAST_DEMO") == "1"
    fast_n = int(os.getenv("FAST_DEMO_N", "20000"))

    prewarm_raw = os.getenv("QUERY_CACHE_PREWARM_N")
    if prewarm_raw is None or str(prewarm_raw).strip() == "":
        prewarm_n_debug = len(DEFAULT_PREWARM_QUERIES)
    else:
        try:
            prewarm_n_debug = int(str(prewarm_raw).strip())
        except ValueError:
            prewarm_n_debug = len(DEFAULT_PREWARM_QUERIES)
    prewarm_n_debug = max(0, min(prewarm_n_debug, len(DEFAULT_PREWARM_QUERIES)))
    prewarm_eff = int(env_bool("QUERY_CACHE_PREWARM", default=True))
    write_eff = int(env_bool("QUERY_CACHE_WRITE", default=False))
    prewarm_write_eff = int(env_bool("QUERY_CACHE_PREWARM_WRITE", default=False))
    os.environ["QUERY_CACHE_PREWARM"] = str(prewarm_eff)
    os.environ["QUERY_CACHE_WRITE"] = str(write_eff)
    os.environ["QUERY_CACHE_PREWARM_WRITE"] = str(prewarm_write_eff)
    cache_abs_path = os.path.abspath(os.getenv("QUERY_CACHE_PATH", os.path.join("data", "query_vec_cache.pkl")))
    print(
        "Config debug (raw): "
        f"PREWARM={raw_prewarm}, "
        f"WRITE={raw_write}, "
        f"PREWARM_WRITE={raw_prewarm_write}, "
        f"PREWARM_N={raw_prewarm_n}"
    )
    print(
        "Config debug (effective): "
        f"PREWARM={prewarm_eff}, "
        f"WRITE={write_eff}, "
        f"PREWARM_WRITE={prewarm_write_eff}, "
        f"PREWARM_N={prewarm_n_debug}, "
        f"cache_path={cache_abs_path}"
    )

    if args.reset_cache:
        try:
            os.remove(os.path.join("data", "query_vec_cache.pkl"))
        except FileNotFoundError:
            pass

    print("="*70)
    print("HIRINGCAFE JOB SEARCH ENGINE - DEMO")
    print("="*70)
    
    # Load data
    print("\n📦 Loading job data..." if EMOJI_OK else "\nLoading job data...")
    loader = JobDataLoader('data/jobs.jsonl')
    with time_block("load jobs/cache"):
        loader.load(fast_demo=fast_demo, fast_n=fast_n)
    if fast_demo:
        print(f"FAST_DEMO loader path: {loader.load_source} (meta+npy)")
    
    jobs = loader.get_jobs()
    embeddings = loader.get_embeddings()

    if fast_demo:
        if EMOJI_OK:
            print(f"⚡ FAST_DEMO enabled: using {len(jobs)} jobs")
        else:
            print(f"FAST_DEMO enabled: using {len(jobs)} jobs")
    
    if EMOJI_OK:
        print(f"\n✅ Loaded {len(jobs)} jobs with embeddings\n")
    else:
        print(f"\nLoaded {len(jobs)} jobs with embeddings\n")
    
    # Initialize search engine
    is_mmap = isinstance(embeddings.get("explicit"), np.memmap) if isinstance(embeddings, dict) else False
    emb_explicit = embeddings.get("explicit") if isinstance(embeddings, dict) else None
    if emb_explicit is not None:
        print(f"embeddings dtype: {emb_explicit.dtype}, shape: {getattr(emb_explicit, 'shape', None)}, memmap: {is_mmap}")
    if is_mmap or (isinstance(embeddings, dict) and embeddings.get("_normalized") is True):
        print("embeddings matrix: reused (mmap)")
        with time_block("build embeddings matrix"):
            search_engine = JobSearchEngine(jobs, embeddings, debug_cache=args.debug_cache, emoji_ok=EMOJI_OK)
    else:
        with time_block("build embeddings matrix"):
            search_engine = JobSearchEngine(jobs, embeddings, debug_cache=args.debug_cache, emoji_ok=EMOJI_OK)
    cache_exists = os.path.exists(cache_abs_path)
    cache_size = os.path.getsize(cache_abs_path) if cache_exists else None
    print(f"Cache file exists={cache_exists}, size={cache_size}")
    
    # Test queries
    print("\n" + "="*70)
    print("SINGLE TURN SEARCHES")
    print("="*70)
    
    single_queries = [
        "data science jobs",
        "senior ML engineer",
        "product manager",
        "startup founder roles",
        "non-profit climate impact jobs",
        "remote python developer"
    ]
    if fast_demo:
        single_queries = single_queries[:3]
    results_per_query = int(os.getenv("RESULTS_PER_QUERY", "3" if fast_demo else "5"))

    prewarm_enabled = env_bool("QUERY_CACHE_PREWARM", default=True)
    if not prewarm_enabled:
        print("Prewarm: disabled (QUERY_CACHE_PREWARM=0)")
    else:
        seeds = DEFAULT_PREWARM_QUERIES
        prewarm_raw = os.getenv("QUERY_CACHE_PREWARM_N")
        if prewarm_raw is None or str(prewarm_raw).strip() == "":
            prewarm_n = len(seeds)
        else:
            try:
                prewarm_n = int(str(prewarm_raw).strip())
            except ValueError:
                prewarm_n = len(seeds)
        prewarm_n = max(0, min(prewarm_n, len(seeds)))
        selected = seeds[:prewarm_n] if prewarm_n > 0 else []
        print(
            f"Prewarm debug: candidates={len(seeds)} prewarm_n={prewarm_n} selected={len(selected)}"
        )
        search_engine.mark_prewarm_keys(selected)
        for q in selected:
            search_engine.embed_query(q)
        print(f"Prewarmed {len(selected)} query embeddings")
        prewarm_write_env = os.getenv("QUERY_CACHE_PREWARM_WRITE", "0")
        prewarm_write = env_bool("QUERY_CACHE_PREWARM_WRITE", default=False)
        saved = False
        if prewarm_write:
            saved = search_engine._query_cache.save(force=True)
            prewarm_note = f"persisted={str(saved)}"
        else:
            prewarm_note = "in-memory only; not persisted"
        print(
            f"Prewarm: {len(selected)} embeddings ({prewarm_note}; QUERY_CACHE_PREWARM_WRITE={prewarm_write_env})"
        )
    
    for query in single_queries:
        with time_block(f"search: {query}"):
            results = search_engine.search(query, top_k=results_per_query)
            search_engine.print_results(results)
            print()
        stats = getattr(search_engine, "last_search_stats", {}) or {}
        if stats:
            print(f"candidates_k used: {stats.get('candidates_k')}")
            print(f"vector search time: {stats.get('vector_time_s'):.3f}s")
            print(f"rerank time: {stats.get('rerank_time_s'):.3f}s")
            print(f"t_embed: {stats.get('t_embed_s'):.3f}s ({stats.get('embed_source')}, hit={stats.get('embed_cache_hit')})")
            print(f"t_vector: {stats.get('t_vector_s'):.3f}s")
            print(f"t_lexical: {stats.get('t_lexical_s'):.3f}s")
            print(f"t_filters: {stats.get('t_filters_s'):.3f}s")
            print(f"t_format: {stats.get('t_format_s'):.3f}s")

    if args.stress_cache:
        stress_queries = [
            "data science jobs",
            "senior ml engineer",
            "product manager ai",
            "business analyst healthcare",
            "remote product owner",
            "fintech product manager",
            "compliance analyst",
            "clinical data scientist",
            "mlops engineer",
            "technical program manager",
            "qa automation engineer",
            "devops engineer",
            "security engineer",
            "data engineer",
            "solutions architect",
        ]
        stress_n = max(1, min(args.stress_n, len(stress_queries)))
        for q in stress_queries[:stress_n]:
            with time_block(f"stress search: {q}"):
                results = search_engine.search(q, top_k=results_per_query)
                search_engine.print_results(results)
                print()
            cache_entries = search_engine.get_query_cache_size()
            cache_max = search_engine.get_query_cache_max()
            print(f"cache_entries={cache_entries}  cache_max={cache_max}")
        cache_entries = len(getattr(search_engine, "_query_vec_cache", {}))
        cache_entries = search_engine.get_query_cache_size()
        cache_max = search_engine.get_query_cache_max()
        print(f"Stress test complete: cache_entries={cache_entries}, max={cache_max}")
        if args.print_cache_keys:
            print("Query cache keys (MRU -> LRU):")
            for k in search_engine.get_query_cache_keys_mru():
                print(f"  {k}")

    if args.print_cache_keys and not args.stress_cache:
        print("Query cache keys (MRU -> LRU):")
        for k in search_engine.get_query_cache_keys_mru():
            print(f"  {k}")
    
    # Test multi-turn refinement
    print("\n" + "="*70)
    print("MULTI-TURN REFINEMENT EXAMPLE")
    print("="*70)
    
    context = SearchContext(
        search_engine,
        initial_top_k=50 if fast_demo else 100,
        mission_top_k=50 if fast_demo else 200,
        emoji_ok=EMOJI_OK,
    )
    
    print("\nTurn 1: User asks 'data science jobs'")
    with time_block("refine: turn 1"):
        results = context.refine("data science jobs")
        search_engine.print_results(results)
    stats = getattr(search_engine, "last_search_stats", {}) or {}
    if stats:
        print(f"candidates_k used: {stats.get('candidates_k')}")
        print(f"vector search time: {stats.get('vector_time_s'):.3f}s")
        print(f"rerank time: {stats.get('rerank_time_s'):.3f}s")
        print(f"t_embed: {stats.get('t_embed_s'):.3f}s ({stats.get('embed_source')}, hit={stats.get('embed_cache_hit')})")
        print(f"t_vector: {stats.get('t_vector_s'):.3f}s")
        print(f"t_lexical: {stats.get('t_lexical_s'):.3f}s")
        print(f"t_filters: {stats.get('t_filters_s'):.3f}s")
        print(f"t_format: {stats.get('t_format_s'):.3f}s")
    
    print("\nTurn 2: User refines 'at companies that care about social good'")
    with time_block("refine: turn 2"):
        results = context.refine("at non-profits or companies focused on social impact")
        search_engine.print_results(results)
    stats = getattr(search_engine, "last_search_stats", {}) or {}
    if stats:
        print(f"candidates_k used: {stats.get('candidates_k')}")
        print(f"vector search time: {stats.get('vector_time_s'):.3f}s")
        print(f"rerank time: {stats.get('rerank_time_s'):.3f}s")
        print(f"t_embed: {stats.get('t_embed_s'):.3f}s ({stats.get('embed_source')}, hit={stats.get('embed_cache_hit')})")
        print(f"t_vector: {stats.get('t_vector_s'):.3f}s")
        print(f"t_lexical: {stats.get('t_lexical_s'):.3f}s")
        print(f"t_filters: {stats.get('t_filters_s'):.3f}s")
        print(f"t_format: {stats.get('t_format_s'):.3f}s")
    
    if not fast_demo:
        print("\nTurn 3: User refines 'make it remote'")
        with time_block("refine: turn 3"):
            results = context.refine("remote only")
            search_engine.print_results(results)
        stats = getattr(search_engine, "last_search_stats", {}) or {}
        if stats:
            print(f"candidates_k used: {stats.get('candidates_k')}")
            print(f"vector search time: {stats.get('vector_time_s'):.3f}s")
            print(f"rerank time: {stats.get('rerank_time_s'):.3f}s")
            print(f"t_embed: {stats.get('t_embed_s'):.3f}s ({stats.get('embed_source')}, hit={stats.get('embed_cache_hit')})")
            print(f"t_vector: {stats.get('t_vector_s'):.3f}s")
            print(f"t_lexical: {stats.get('t_lexical_s'):.3f}s")
            print(f"t_filters: {stats.get('t_filters_s'):.3f}s")
            print(f"t_format: {stats.get('t_format_s'):.3f}s")

    print("\n" + "="*70)
    print("CACHE SUMMARY")
    print("="*70)
    cache_stats = search_engine.get_query_cache_stats()
    hits = cache_stats.get("hits", 0)
    misses = cache_stats.get("misses", 0)
    total = hits + misses
    pct = (hits / total * 100) if total else 0
    print(f"Saved embedding calls via cache: {hits} / {total} ({pct:.0f}%)")
    
    # Save token report
    print("\n" + "="*70)
    print("TOKEN USAGE")
    print("="*70)
    tracker.get_summary()
    with time_block("write token report"):
        tracker.save_report('tokens_report.txt')
    if EMOJI_OK:
        print("✅ Report saved to tokens_report.txt")
    else:
        print("Report saved to tokens_report.txt")

if __name__ == "__main__":
    main()
