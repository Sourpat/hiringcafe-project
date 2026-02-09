# HiringCafe Job Search Engine

An AI-powered job discovery engine that helps users find relevant jobs using natural language queries and iterative refinement.

## The problem (why job search is broken)
Job search is messy for two reasons:
- **Job descriptions are inconsistent**: critical requirements are often implied, not explicitly stated.
- **Users search by intent**, not filters: “remote senior ML at mission-driven companies” is a single thought, not 6 dropdowns.

## The solution (what I built)
This demo is a job discovery engine that:
- Uses **multi-vector semantic search** (explicit requirements, inferred requirements, company characteristics).
- Supports **multi-turn refinement** (follow-up intent narrows results without starting over).
- Stays **token-cheap** via query embedding caching and local scoring.
- Produces **explainable rankings** and **debug artifacts** to prove why results rank.

## Proof it works (what you can verify quickly)
- **FAST_DEMO load** on 20k jobs (memmap embeddings): ~2.6s–2.9s in my runs.
- **Explainable scoring** per result: vector, lexical, mission, role gate, cache source.
- **Offline deterministic mode**: reproducible runs with **no OpenAI calls**.
- **Golden regression checks**: stable top results across changes unless intentionally regenerated.

## Step 4 (proof in one pass)
**One-liner that just works (offline, fast, explainable, trace export):**
```powershell
$env:OFFLINE_MODE="1"; $env:FAST_DEMO="1"; python .\demo.py --explain --trace-out data\trace.jsonl
```

**What good output looks like:**
- Offline mode banner + FAST_DEMO load
- Results include explain blocks (vector/lexical/mission/role_gate/cache)
- Trace file written to `data/trace.jsonl`

**Artifacts to inspect in this repo:**
- `data/trace.jsonl` (per-query timings + top results + cache stats)
- `data/eval_report.md` (quality metrics + timings)
- `data/golden_diff.md` + `data/golden_queries.json` (regression harness inputs/outputs)

## Why this approach (not the alternatives)
Common alternatives and why I didn’t use them for a take-home:
- **Title-only / keyword-only search**: fails on implied requirements and synonyms.
- **LLM rerank everything**: great quality, but costs explode when scoring many candidates.
- **Single embedding for everything**: misses “culture/mission” signal and can overfit to generic wording.

What I chose instead:
- **Precomputed embeddings + local cosine similarity** keep per-query compute cheap.
- **Three vectors** separate different signals:
  - *explicit*: what the job clearly states
  - *inferred*: implied skills/related requirements
  - *company*: culture/mission/values characteristics
- A weighted combination balances relevance without letting “company vibe” dominate.

## Offline vs online (cost + reproducibility)
- **Offline mode (`OFFLINE_MODE=1`)**: deterministic, no network calls, $0 cost.
- **Online mode**: uses OpenAI for query embeddings and intent parsing.
  - Cost stays low due to **query embedding cache hit rates** and local scoring.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Download `jobs.jsonl` from: https://drive.google.com/file/d/1RRVWYAvfb4hUus1hUDY1nPQUJGqpiBiq/view?usp=sharing

Place in: `data/jobs.jsonl`

### 3. Run Demo
```bash
python demo.py
```

## Trade-offs and decisions
This is a demo, so I optimized for **clarity, determinism, and cost control**.

Examples:
- **Lexical boosts + role gates vs trained ranking model**
  - Chosen: simple lexical/title gates (transparent, predictable).
  - Alternative: train a model on click data.
  - Why not: no labeled feedback data in a take-home.
  - When I’d change: once we have 1k+ labeled sessions or click logs.

- **LRU cache vs frequency-based cache**
  - Chosen: LRU (simple, good for “recent repeated queries” in demos).
  - Alternative: LFU/frequency (better when a few queries dominate).
  - When I’d change: if production traffic shows stable “top queries” by frequency.

## Discoveries from the dataset (what I observed)
While running evaluation queries and reviewing results:
- “Mission-driven” queries often return a mix of nonprofit and public sector roles.
- Company and description text contains enough signal for simple mission heuristics to work.
- Outliers happen in real data and need guardrails (see failure handling below).

## Failure modes and how the demo handles them
- **Cache issues**
  - Disk cache can be disabled (`QUERY_CACHE_WRITE=0`) and the system still runs.
  - In offline mode, cached embeddings avoid network dependency entirely.

- **Performance outliers**
  - The eval harness tracks per-phase timings and flags outliers.
  - Candidate caps and timing breakdowns prevent pathological rerank runs from dominating.

## Token and cost story (why this stays cheap)
Two principles keep cost low:
1. **Query embedding cache** prevents repeat embedding calls.
2. **Local vector scoring** avoids LLM reranking over large candidate sets.

In **offline mode**: $0 (no OpenAI calls).
In **online mode**: typical demo sessions remain low-cost because most repeated queries are cache hits.

## What I’d build next (roadmap)
If this were a real product, I’d sequence improvements like this:

**Month 1 (high impact, low effort)**
- Add structured filters (`remote:true`, `seniority:senior`) parsed from intent.
- Salary extraction and range filters.
- Click feedback capture for lightweight relevance tuning.

**Month 2 (high impact, higher effort)**
- Better mission alignment scoring (expand nonprofit/public sector detection).
- Calibrated weighting per query type (skills-heavy vs values-heavy queries).
- Improved dedupe and diversity constraints.

**Month 3 (scale and productization)**
- Expose `/search?q=` API endpoint and stable JSON schema.
- A/B test rank strategies.
- Observability dashboards for latency, cache hit rate, and quality metrics.

## Architecture

### Data Loading (`src/data_loader.py`)
- Loads 100k jobs from jobs.jsonl in batches
- Extracts 3 embedding vectors per job from `v7_processed_job_data`:
  - `embedding_explicit_vector`: Explicit job requirements
  - `embedding_inferred_vector`: Inferred/related qualifications
  - `embedding_company_vector`: Company characteristics
- Caches to pickle for fast subsequent loads

### Search Algorithm (`src/search_engine.py`)
1. User enters natural language query
2. Embed query using OpenAI `text-embedding-3-small` model
3. Compute cosine similarity against all 3 embedding vectors
4. Weighted combination: `0.4*explicit + 0.4*inferred + 0.2*company`
5. Return top-K results ranked by combined score

### Why 3 vectors instead of 1
I kept the model simple but separated signals because job data is messy:
- **Explicit vector** captures stated requirements (“must have Python”).
- **Inferred vector** helps when requirements are implied (“data pipelines” implies SQL, ETL, orchestration).
- **Company vector** adds culture/mission signal without forcing it into skill matching.

**Alternative considered:** a single embedding for everything.
**Why not:** it tends to blur “skills match” and “values match,” which hurts ranking when users refine by mission/values.

### Why these weights (0.4/0.4/0.2)
Explicit and inferred are equally weighted to capture both stated and implied requirements.
Company is lower-weighted so “culture fit” supports ranking but does not dominate core relevance.

### Explainable Ranking
Pass `--explain` in the demo to emit a score breakdown per result. This includes:
- Vector score components
- Title/role boosts
- Mission/remote boosts
- Final composite score

**Why this approach:**
- Avoids re-embedding jobs (expensive)
- Uses pre-computed embeddings (efficient)
- Multi-dimensional search captures explicit + implicit requirements
- Company vector helps match culture fit

### Refinement Strategy (`src/context.py`)
1. Track conversation history and current result pool
2. For each new query:
  - Extract structured intent using GPT-4o-mini (role, remote, seniority, etc.)
  - Apply filters to previous results (don't re-search from scratch)
  - Re-rank filtered results
  - Return top-K
3. Maintains context across turns for smarter filtering

**Why this approach:**
- First query does full search (100-200 results)
- Subsequent queries filter + re-rank (token efficient)
- Intent parsing extracts structured meaning from messy natural language
- Preserves conversation context for multi-turn refinement

### Token Tracking (`src/token_tracker.py`)
- Logs every OpenAI API call
- Tracks input tokens, output tokens, and cost
- Alerts when approaching budget
- Generates detailed report for submission

## Query Embedding Cache (LRU + Disk Persistence)

To reduce embedding API calls and improve latency, the demo uses a normalized query embedding cache with LRU eviction and optional disk persistence.

### Key behaviors
- **Normalization:** queries are normalized (trim, collapse whitespace, lowercase) so variants map to the same key.
- **LRU eviction:** cache is capped via `QUERY_CACHE_MAX`; older entries are evicted (MRU → LRU ordering printed in demo).
- **Disk persistence:** if `QUERY_CACHE_WRITE=1`, embeddings are saved to `data/query_vec_cache.pkl` and reloaded on next run.
- **Prewarm control:** prewarm runs populate embeddings in memory; disk writes are controlled by `QUERY_CACHE_PREWARM_WRITE`.

### Config (env vars)
- `QUERY_CACHE_MAX` (default 500): max LRU entries (e.g., 5 for stress testing)
- `QUERY_CACHE_WRITE` (0/1): enable disk persistence
- `QUERY_CACHE_PREWARM_WRITE` (0/1): allow prewarm embeddings to be persisted
- `QUERY_CACHE_PATH` (default `data/query_vec_cache.pkl`): cache file path

### Repro commands (proof)
```powershell
# Fast demo + cache cap proof
Remove-Item -Force .\data\query_vec_cache.pkl -ErrorAction SilentlyContinue
$env:FAST_DEMO="1"; $env:FAST_DEMO_N="20000"; $env:QUERY_CACHE_MAX="5"; $env:QUERY_CACHE_WRITE="1"; $env:QUERY_CACHE_PREWARM_WRITE="0"
python .\demo.py --stress-cache --stress-n 12 --print-cache-keys

# Disk load proof
python .\demo.py --print-cache-keys

# Debug attribution proof (mem vs disk vs fresh)
python .\demo.py --debug-cache --print-cache-keys
```

## Measured Performance
From recent offline FAST_DEMO runs (20k jobs, memmap embeddings):
- FAST_DEMO load time: 2.40s–2.81s
- Typical query timings (cache hit):
  - embed: 0.000–0.001s
  - vector: ~0.047–0.079s (e.g., 0.079s)
  - rerank: ~0.005–0.010s (e.g., 0.007s)

Outlier note: the “warehouse associate” rerank once hit 1.434s; after candidate cap + timing breakdowns, it is now ~0.004s.

## Caching
Query embeddings flow through three layers:
- **disk**: persisted embeddings from previous runs (`data/query_vec_cache.pkl`)
- **mem**: in-process LRU cache
- **fresh**: new embeddings on cache miss

Observed cache hit rates from demo runs:
- 89% (8/9)
- 80% (4/5)

Prewarm can be disabled (`QUERY_CACHE_PREWARM=0`) and the cache still serves disk hits.

## Explainability + Trace Artifacts
- `--explain` prints per-result score breakdowns: vector, lexical, mission, role_gate, cache source.
- `--trace-out data/trace.jsonl` writes per-query JSONL with timings, cache stats, and top results.

Artifacts in this repo:
- `data/trace.jsonl`
- `data/eval_report.md`
- `data/golden_snapshot.md`
- `data/golden_diff.md`

## Quality and Regression
**Offline eval harness** (`scripts/eval_offline.py`) computes:
- role_intent_hit_rate (title contains ROLE terms)
- remote_precision (when query asks remote)
- mission_precision (when query is mission-focused)
- diversity (unique companies / 10)

**Golden regression** checks expected top-5 job IDs per query:
- Generate/update snapshot: `scripts/generate_golden.py --force`
- Check drift: `scripts/check_golden.py`

## Reproduce My Runs
```powershell
OFFLINE_MODE=1 FAST_DEMO=1 python .\demo.py --explain --trace-out data\trace.jsonl
OFFLINE_MODE=1 FAST_DEMO=1 python .\scripts\eval_offline.py
OFFLINE_MODE=1 FAST_DEMO=1 python .\scripts\generate_golden.py --force
OFFLINE_MODE=1 FAST_DEMO=1 python .\scripts\check_golden.py
python -m pytest -q
```

Artifacts to inspect:
- `data/eval_report.md`
- `data/golden_snapshot.md`
- `data/trace.jsonl`

## Example Queries

### Single-Turn Searches
```
"data science jobs"
→ Returns: Data Scientist, ML Engineer, Analytics Manager roles

"senior roles at startups"
→ Returns: Senior Software Engineer, VP Eng, CTO roles at startups

"remote python developer"
→ Returns: Python Dev, Backend Eng, Full-Stack roles with remote status
```

### Multi-Turn Refinement Flow
```
Turn 1: "data science jobs"
  → [~100 data science roles]

Turn 2: "at mission-driven companies"
  → [~50 roles at nonprofits/mission-driven companies]

Turn 3: "make it remote"
  → [~20 remote roles at mission-driven companies]
```

## Query Types That Work Well
- Specific roles: "senior ML engineer", "product manager"
- Company focus: "startups", "non-profits", "healthcare companies"
- Location/work style: "remote", "hybrid", "NYC based"
- Industry: "climate tech", "fintech", "edtech"
- Combinations: "remote senior ML engineer at climate startups"

## Query Types That Are Tricky
- Very vague: "something interesting"
- Implicit: "not just another job"
- Contradictory: "senior but entry-level"
- Typos and misspellings

**Solution:** The system still returns relevant results, just with slightly lower confidence. Users can refine.

## Token Efficiency

### Pricing (OpenAI Feb 2025)
- `text-embedding-3-small`: $0.02 per 1M tokens
- `gpt-4o-mini`: $0.15 per 1M input + $0.60 per 1M output

### Per-Operation Costs
- Initial search query: ~200 tokens = $0.000004
- Intent parsing (LLM): ~300-500 tokens = $0.00005-0.0001
- Vector search: $0 (local computation)

### Budget Usage
With $10 budget, you can:
- Do ~100+ single searches, OR
- Do ~50 multi-turn conversations with 5 refinements each, OR
- Realistic demo with 20-30 queries

**Estimated usage for this project:** < $0.05 / $10.00

## Files in Submission

```
├── demo.py                 # Run this: python demo.py
├── README.md               # This file
├── requirements.txt        # Dependencies
├── tokens_report.txt       # Generated after running demo
├── .env                    # Your API key
├── src/
│   ├── data_loader.py     # Load + parse jobs.jsonl
│   ├── search_engine.py   # Vector search logic
│   ├── context.py         # Conversation + refinement
│   └── token_tracker.py   # Token monitoring
└── data/
    └── jobs.jsonl         # 100k job postings (not in repo, download separately)
```

## How to Use in Development

### 1. Start with basic search
```python
from src.search_engine import JobSearchEngine
from src.query_cache import QueryEmbeddingCache

cache = QueryEmbeddingCache()
engine = JobSearchEngine(jobs, embeddings, query_cache=cache)
results = engine.search("data science jobs")
```

### 2. Add conversation context
```python
from src.context import SearchContext

context = SearchContext(engine)
results = context.refine("data science jobs")
results = context.refine("remote only")  # Filters previous results
```

### 3. Monitor tokens
```python
from src.token_tracker import tracker

tracker.get_summary()  # See current usage
tracker.save_report('tokens_report.txt')  # Generate report
```

## Improvements With More Time

1. **Better ranking** - Use LLM to re-rank top-20 results for quality
2. **Semantic understanding** - Use embeddings to understand "social impact" ≈ "nonprofit"
3. **Caching** - Cache parsed company metadata to avoid re-parsing
4. **Feedback loop** - Learn from which results users click
5. **Multi-field search** - Search title, description, benefits separately
6. **Salary parsing** - Better extract and filter by salary ranges
7. **Company intelligence** - Use company size, funding, growth to score
8. **Location matching** - Better handling of remote/hybrid/onsite combinations

## Notes

- First run will take 2-5 minutes (loads + embeds 100k jobs)
- Subsequent runs are instant (uses cached pickle)
- Demo runs 10+ queries, costs ~$0.001-0.005
- All API calls are logged and tracked

## Query Cache Controls

Environment variables for query embedding cache behavior:
- `QUERY_CACHE_MAX` (default: 500): LRU cache size.
- `QUERY_CACHE_WRITE` (default: 1): set to 0 to disable disk persistence.
- `QUERY_CACHE_PREWARM_WRITE` (default: 0): set to 1 to persist prewarmed queries.

Debug flags:
- `--debug-cache`: print per-query cache hits.
- `--print-cache-keys`: print cache keys in MRU→LRU order.
- No external dependencies except OpenAI + numpy + scikit-learn
