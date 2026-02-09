import argparse
import json
import os
import sys
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from demo import build_query_cache
from src.data_loader import JobDataLoader
from src.search_engine import JobSearchEngine, get_job_id


GOLDEN_PATH = os.path.join("data", "golden_queries.json")
SNAPSHOT_PATH = os.path.join("data", "golden_snapshot.md")


def _load_golden(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_golden(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _build_engine():
    os.environ["OFFLINE_MODE"] = "1"
    if "CANDIDATE_K" not in os.environ:
        os.environ["CANDIDATE_K"] = "600"
    fast_demo = os.getenv("FAST_DEMO") == "1"
    fast_n = int(os.getenv("FAST_DEMO_N", "20000"))

    loader = JobDataLoader("data/jobs.jsonl")
    loader.load(fast_demo=fast_demo, fast_n=fast_n)
    jobs = loader.get_jobs()
    embeddings = loader.get_embeddings()

    cache = build_query_cache(debug=False)
    engine = JobSearchEngine(
        jobs,
        embeddings,
        debug_cache=False,
        emoji_ok=False,
        explain=False,
        query_cache=cache,
    )
    return engine


def main():
    parser = argparse.ArgumentParser(description="Generate golden expectations")
    parser.add_argument("--force", action="store_true", help="Regenerate all expected_top_ids")
    args = parser.parse_args()

    engine = _build_engine()
    golden = _load_golden(GOLDEN_PATH)

    snapshot_lines = [
        "# Golden Snapshot",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    updated = False
    for entry in golden:
        query = entry.get("query")
        if not query:
            continue
        expected = entry.get("expected_top_ids") or []
        results = engine.search(query, top_k=10)
        top5 = results[:5]
        top_ids = [get_job_id(job) for job in top5]
        if args.force or not expected:
            entry["expected_top_ids"] = top_ids
            updated = True

        snapshot_lines.append(f"## {query}")
        snapshot_lines.append("")
        for idx, job in enumerate(top5, start=1):
            snapshot_lines.append(engine.format_result(job, rank=idx))
        snapshot_lines.append("")

    if updated:
        _save_golden(GOLDEN_PATH, golden)

    with open(SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(snapshot_lines))


if __name__ == "__main__":
    main()
