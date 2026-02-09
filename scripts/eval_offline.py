import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader import JobDataLoader
from src.search_engine import JobSearchEngine, ROLE_KEYWORDS, ROLE_PHRASES
from demo import build_query_cache


QUERY_SUITE = [
    "data science jobs",
    "senior ML engineer",
    "mlops engineer",
    "product manager",
    "business analyst",
    "product owner remote",
    "remote python developer",
    "nonprofit climate impact jobs",
    "mission-driven data scientist",
    "electrician apprentice",
    "registered nurse",
    "warehouse associate",
]

MISSION_HINTS = {"nonprofit", "mission", "climate", "health"}


def _tokenize(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _title_has_role(title):
    if not title:
        return False
    lower = title.lower()
    if any(phrase in lower for phrase in ROLE_PHRASES):
        return True
    tokens = _tokenize(lower)
    return bool(tokens & ROLE_KEYWORDS)


def _workplace_is_remote(value):
    if not value:
        return False
    lower = str(value).lower()
    return "remote" in lower or "hybrid" in lower


def _query_has_remote(query):
    return "remote" in query.lower()


def _query_has_mission(query):
    tokens = _tokenize(query)
    return bool(tokens & MISSION_HINTS)


def run_eval(output_dir="data", single_query=None, debug=False):
    os.environ["OFFLINE_MODE"] = "1"
    if "CANDIDATE_K" not in os.environ:
        os.environ["CANDIDATE_K"] = "600"
    os.makedirs(output_dir, exist_ok=True)

    fast_demo = os.getenv("FAST_DEMO") == "1"
    fast_n = int(os.getenv("FAST_DEMO_N", "20000"))

    loader = JobDataLoader("data/jobs.jsonl")
    loader.load(fast_demo=fast_demo, fast_n=fast_n)
    jobs = loader.get_jobs()
    embeddings = loader.get_embeddings()

    query_cache = build_query_cache(debug=False)
    engine = JobSearchEngine(
        jobs,
        embeddings,
        debug_cache=False,
        emoji_ok=False,
        explain=True,
        query_cache=query_cache,
    )

    queries = [single_query] if single_query else QUERY_SUITE

    def _avg(values):
        values = [v for v in values if v is not None]
        return sum(values) / len(values) if values else None

    def run_pass(label):
        print(f"{label} starting...")
        start_stats = engine.get_query_cache_stats()
        results = []
        for query in queries:
            top = engine.search(query, top_k=10)
            top10 = top[:10]
            titles = []
            companies = []
            remote_hits = 0
            mission_hits = 0
            role_hits = 0
            for job in top10:
                meta = job.get("_meta", {}) or {}
                title = meta.get("title") or job.get("title") or ""
                company = meta.get("company") or job.get("company") or ""
                workplace = meta.get("workplace_type") or job.get("workplace_type") or ""
                titles.append(title)
                companies.append(company)
                if _title_has_role(title):
                    role_hits += 1
                if _workplace_is_remote(workplace):
                    remote_hits += 1
                if meta.get("mission_match") is True:
                    mission_hits += 1

            denom = max(1, len(top10))
            role_intent_hit_rate = role_hits / denom
            diversity = len({c for c in companies if c}) / denom

            remote_precision = None
            if _query_has_remote(query):
                remote_precision = remote_hits / denom

            mission_precision = None
            if _query_has_mission(query):
                mission_precision = mission_hits / denom

            stats = getattr(engine, "last_search_stats", {}) or {}
            timings = {
                "embed": stats.get("t_embed_s"),
                "vector": stats.get("t_vector_s"),
                "rerank": stats.get("t_lexical_s"),
            }

            notes = []
            if timings["rerank"] is not None and timings["rerank"] > 0.2:
                notes.append("rerank_slow")

            if debug:
                candidate_k = stats.get("candidates_k")
                candidate_count = stats.get("candidate_count")
                t_mission_s = stats.get("t_mission_s")
                t_lexical_s = stats.get("t_lexical_inner_s")
                t_role_gate_s = stats.get("t_role_gate_s")
                mission_matches = sum(1 for job in top10 if (job.get("_meta", {}) or {}).get("mission_match") is True)
                print("DEBUG eval query:", query)
                print(f"  candidate_k={candidate_k}")
                print(f"  candidate_count={candidate_count}")
                print(f"  t_mission_s={t_mission_s:.6f}" if t_mission_s is not None else "  t_mission_s=None")
                print(f"  t_lexical_s={t_lexical_s:.6f}" if t_lexical_s is not None else "  t_lexical_s=None")
                print(f"  t_role_gate_s={t_role_gate_s:.6f}" if t_role_gate_s is not None else "  t_role_gate_s=None")
                print(f"  mission_matches_top10={mission_matches}")

            results.append(
                {
                    "query": query,
                    "metrics": {
                        "role_intent_hit_rate": role_intent_hit_rate,
                        "remote_precision": remote_precision,
                        "mission_precision": mission_precision,
                        "diversity": diversity,
                    },
                    "timings": timings,
                    "notes": notes,
                    "top_titles": titles,
                    "top_companies": companies,
                }
            )

        end_stats = engine.get_query_cache_stats()
        hits = end_stats.get("hits", 0) - start_stats.get("hits", 0)
        misses = end_stats.get("misses", 0) - start_stats.get("misses", 0)
        total = hits + misses
        cache_hit_rate = hits / total if total else 0.0

        aggregate = {
            "role_intent_hit_rate": _avg([r["metrics"]["role_intent_hit_rate"] for r in results]),
            "remote_precision": _avg([r["metrics"]["remote_precision"] for r in results]),
            "mission_precision": _avg([r["metrics"]["mission_precision"] for r in results]),
            "diversity": _avg([r["metrics"]["diversity"] for r in results]),
            "t_embed_s": _avg([r["timings"]["embed"] for r in results]),
            "t_vector_s": _avg([r["timings"]["vector"] for r in results]),
            "t_rerank_s": _avg([r["timings"]["rerank"] for r in results]),
        }

        print(
            f"{label} summary: cache_hit_rate={cache_hit_rate:.2f} "
            f"avg_embed={aggregate['t_embed_s']:.3f}s "
            f"avg_vector={aggregate['t_vector_s']:.3f}s "
            f"avg_rerank={aggregate['t_rerank_s']:.3f}s"
        )

        return results, aggregate, cache_hit_rate

    pass_results = []
    results, aggregate, cache_hit_rate = run_pass("Pass 1 (cold)")
    pass_results.append({
        "label": "Pass 1 (cold)",
        "results": results,
        "aggregate": aggregate,
        "cache_hit_rate": cache_hit_rate,
    })
    results, aggregate, cache_hit_rate = run_pass("Pass 2 (warm)")
    pass_results.append({
        "label": "Pass 2 (warm)",
        "results": results,
        "aggregate": aggregate,
        "cache_hit_rate": cache_hit_rate,
    })

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "offline",
        "fast_demo": fast_demo,
        "candidate_k": int(os.getenv("CANDIDATE_K", "600")),
        "passes": pass_results,
    }

    json_path = os.path.join(output_dir, "eval_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_path = os.path.join(output_dir, "eval_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Offline Evaluation Report\n\n")
        f.write(f"Generated: {report['generated_at']}\n\n")
        for pass_entry in pass_results:
            label = pass_entry["label"]
            results = pass_entry["results"]
            aggregate = pass_entry["aggregate"]
            cache_hit_rate = pass_entry["cache_hit_rate"]

            f.write(f"\n## {label}\n\n")
            f.write(f"Cache hit rate: {cache_hit_rate:.2f}\n\n")
            f.write("| Query | Role hit rate | Remote precision | Mission precision | Diversity | t_embed (s) | t_vector (s) | t_rerank (s) |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for row in results:
                metrics = row["metrics"]
                timings = row["timings"]
                f.write(
                    "| {query} | {role:.2f} | {remote} | {mission} | {diversity:.2f} | {embed:.3f} | {vector:.3f} | {rerank:.3f} |\n".format(
                        query=row["query"],
                        role=metrics["role_intent_hit_rate"],
                        remote="-" if metrics["remote_precision"] is None else f"{metrics['remote_precision']:.2f}",
                        mission="-" if metrics["mission_precision"] is None else f"{metrics['mission_precision']:.2f}",
                        diversity=metrics["diversity"],
                        embed=timings["embed"] or 0.0,
                        vector=timings["vector"] or 0.0,
                        rerank=timings["rerank"] or 0.0,
                    )
                )

            f.write("\n### Aggregate Averages\n\n")
            f.write(
                "- role_intent_hit_rate: {value}\n".format(
                    value="-" if aggregate["role_intent_hit_rate"] is None else f"{aggregate['role_intent_hit_rate']:.2f}"
                )
            )
            f.write(
                "- remote_precision: {value}\n".format(
                    value="-" if aggregate["remote_precision"] is None else f"{aggregate['remote_precision']:.2f}"
                )
            )
            f.write(
                "- mission_precision: {value}\n".format(
                    value="-" if aggregate["mission_precision"] is None else f"{aggregate['mission_precision']:.2f}"
                )
            )
            f.write(
                "- diversity: {value}\n".format(
                    value="-" if aggregate["diversity"] is None else f"{aggregate['diversity']:.2f}"
                )
            )
            f.write(
                "- t_embed_s: {value}\n".format(
                    value="-" if aggregate["t_embed_s"] is None else f"{aggregate['t_embed_s']:.3f}"
                )
            )
            f.write(
                "- t_vector_s: {value}\n".format(
                    value="-" if aggregate["t_vector_s"] is None else f"{aggregate['t_vector_s']:.3f}"
                )
            )
            f.write(
                "- t_rerank_s: {value}\n".format(
                    value="-" if aggregate["t_rerank_s"] is None else f"{aggregate['t_rerank_s']:.3f}"
                )
            )

    return report


def main():
    parser = argparse.ArgumentParser(description="Offline evaluation harness")
    parser.add_argument("--single", type=str, default=None, help="Run a single query")
    parser.add_argument("--debug", action="store_true", help="Print detailed debug breakdown")
    args = parser.parse_args()

    run_eval(single_query=args.single, debug=args.debug)


if __name__ == "__main__":
    main()