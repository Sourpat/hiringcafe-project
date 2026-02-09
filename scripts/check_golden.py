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
DIFF_PATH = os.path.join("data", "golden_diff.md")


def _load_golden(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    return engine, jobs


def _job_index(jobs):
    index = {}
    for job in jobs:
        job_id = get_job_id(job)
        if job_id not in index:
            index[job_id] = job
    return index


def _format_job(engine, job, rank):
    if job is None:
        return f"{rank}. <missing job>"
    return engine.format_result(job, rank=rank)


def main():
    parser = argparse.ArgumentParser(description="Check golden ranking regression")
    parser.add_argument("--threshold", type=int, default=2, help="Max mismatches allowed in top5")
    parser.add_argument("--single", type=str, default=None, help="Run a single query")
    args = parser.parse_args()

    engine, jobs = _build_engine()
    golden = _load_golden(GOLDEN_PATH)
    job_lookup = None

    diff_lines = [
        "# Golden Diff",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
    ]

    drift = False
    for entry in golden:
        query = entry.get("query")
        if not query:
            continue
        if args.single and query != args.single:
            continue

        expected = entry.get("expected_top_ids") or []
        results = engine.search(query, top_k=10)
        top5 = results[:5]
        actual = [get_job_id(job) for job in top5]

        mismatches = 0
        if len(expected) < 5:
            mismatches = 5
        else:
            mismatches = sum(1 for i in range(5) if expected[i] != actual[i])

        if mismatches > args.threshold:
            drift = True
            if job_lookup is None:
                job_lookup = _job_index(jobs)
            diff_lines.append(f"## {query}")
            diff_lines.append("")
            diff_lines.append(f"mismatches: {mismatches} (threshold={args.threshold})")
            diff_lines.append("")
            diff_lines.append("### Expected IDs")
            diff_lines.append("")
            for idx, job_id in enumerate(expected[:5], start=1):
                diff_lines.append(f"{idx}. {job_id}")
            diff_lines.append("")
            diff_lines.append("### Actual IDs")
            diff_lines.append("")
            for idx, job_id in enumerate(actual[:5], start=1):
                diff_lines.append(f"{idx}. {job_id}")
            diff_lines.append("")
            diff_lines.append("### Expected Results")
            diff_lines.append("")
            for idx, job_id in enumerate(expected[:5], start=1):
                diff_lines.append(_format_job(engine, job_lookup.get(job_id), idx))
            diff_lines.append("")
            diff_lines.append("### Actual Results")
            diff_lines.append("")
            for idx, job in enumerate(top5, start=1):
                diff_lines.append(_format_job(engine, job, idx))
            diff_lines.append("")

    if drift:
        with open(DIFF_PATH, "w", encoding="utf-8") as f:
            f.write("\n".join(diff_lines))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
