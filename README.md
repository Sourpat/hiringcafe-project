# HiringCafe Job Search Engine

An AI-powered job discovery engine that helps users find relevant jobs using natural language queries and iterative refinement.

## What this demo proves
- FAST_DEMO load speed on 20k jobs (memmap embeddings) is ~2.67s–2.81s.
- Cached query latency is consistently low in offline runs (see measured timings below).
- Explainable scoring shows *why* a result ranks (vector, lexical, mission, role gate, cache source).
- Offline mode is deterministic and reproducible (no OpenAI calls).
- Golden regression checks guard ranking quality across changes.

## Step 4 (proof in one pass)
**One-liner that just works (offline, fast, explainable, trace export):**
```powershell
$env:OFFLINE_MODE="1"; $env:FAST_DEMO="1"; python .\demo.py --explain --trace-out data\trace.jsonl
```

**What good output looks like:**
- Offline mode banner + FAST_DEMO load
- Results include explain blocks (vector/lexical/mission/role_gate/cache)
- Trace file written to `data/trace.jsonl`

**Proof artifacts to inspect:**
- `data/eval_report.md` (quality + timings)
- `data/golden_snapshot.md` (deterministic top-5 IDs)
- `python -m pytest -q` (regression gate)

**What should not change run-to-run:**
- Golden top-5 IDs in `data/golden_snapshot.md` (unless you intentionally regenerate)

**Step 4 deliverable:**
Use the commands below to reproduce outputs and verify artifacts.

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
- FAST_DEMO load time: 2.67s–2.81s
- Typical query timings (cache hit):
  - embed: 0.000–0.001s
  - vector: ~0.047–0.079s
  - rerank: ~0.005–0.010s

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

## Trade-offs Made

### Optimized For:
- **Token efficiency** - Stay well under $10 budget
- **Speed** - Simple architecture, fast iteration
- **Handling real-world messy data** - Job descriptions are inconsistent
- **User experience** - Multi-turn conversation feels natural

### Not Optimized For:
- Perfect ranking algorithms (good enough is better)
- Comprehensive error handling (this is a demo)
- Production-grade code quality (it works and it's readable)
- UI/UX polish (not the focus)

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
