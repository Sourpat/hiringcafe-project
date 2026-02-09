
import os
import re
import json
import html
import atexit
import time
import hashlib
from src.query_cache import (
    QueryEmbeddingCache,
    normalize_query_key,
    query_cache_prewarm_write,
    should_log_cache_debug,
)
from datetime import datetime, timezone
from openai import OpenAI
import numpy as np
from src.token_tracker import tracker

MISSION_STRONG_KEYWORDS = [
    "nonprofit",
    "foundation",
    "united nations",
    "ngo",
    "humanitarian",
    "public health",
    "global health",
    "climate",
    "sustainability",
    "environmental justice",
    "charity",
    "cdc foundation",
    "nature conservancy",
]
MISSION_TEXT_KEYWORDS = [
    "nonprofit",
    "donation",
    "charity",
    "humanitarian",
    "public health",
    "climate justice",
    "environmental justice",
    "refugee",
    "education equity",
]
MISSION_COMPANY_TOKENS = [
    "foundation",
    "university",
    "hospital",
    "institute",
    "conservancy",
    "united nations",
]
ROLE_KEYWORDS = {
    "data",
    "scientist",
    "science",
    "engineer",
    "engineering",
    "ml",
    "machine",
    "learning",
    "ai",
    "software",
    "backend",
    "frontend",
    "fullstack",
    "full",
    "stack",
    "product",
    "manager",
    "analyst",
    "analytics",
    "developer",
    "devops",
    "sre",
}
ROLE_PHRASES = [
    "data science",
    "data scientist",
    "machine learning",
    "ml engineer",
    "software engineer",
    "backend engineer",
    "frontend engineer",
    "full stack",
    "product manager",
    "data analyst",
    "business analyst",
]
ROLE_NEAR_MISS_TOKENS = {
    "mlops",
    "llmops",
    "platform",
    "applied",
    "scientist",
    "data",
    "engineer",
    "sre",
}

def get_job_id(job):
    meta = job.get("_meta", {}) or {}
    for key in ("job_id", "url", "source_id"):
        value = meta.get(key) or job.get(key)
        if value:
            return str(value)

    title = meta.get("title") or job.get("title") or ""
    company = meta.get("company") or job.get("company") or ""
    location = meta.get("location") or job.get("location") or ""
    workplace = meta.get("workplace_type") or job.get("workplace_type") or ""

    def _norm(text):
        return " ".join(str(text).strip().lower().split())

    base = f"{_norm(title)}|{_norm(company)}|{_norm(location)}|{_norm(workplace)}"
    digest = hashlib.sha1(base.encode("utf-8")).hexdigest()[:12]
    return f"sha1:{digest}"

class JobSearchEngine:
    def __init__(self, jobs, embeddings, debug_cache=False, emoji_ok=True, explain=False, query_cache: QueryEmbeddingCache | None = None):
        self.jobs = jobs
        self.embeddings = embeddings
        self._debug_cache = debug_cache
        self._emoji_ok = emoji_ok
        self._explain = explain
        self._embedding_dim = 1536
        if query_cache is None:
            raise ValueError("JobSearchEngine requires a QueryEmbeddingCache instance (dependency injection).")
        self._query_cache = query_cache
        self._prewarm_write = query_cache_prewarm_write()
        self._prewarm_keys = set()
        self._prewarm_cache = {}
        atexit.register(self._query_cache.save)
        atexit.register(self._log_query_cache_evictions)
        print(
            f"Query cache config: max={self._query_cache.max_size}, "
            f"write={int(self._query_cache.enable_disk)}, "
            f"prewarm_write={int(self._prewarm_write)}, "
            f"path={self._query_cache.path}"
        )
        self.client = None
        if os.getenv("OFFLINE_MODE") == "1":
            self.api_key = None
        else:
            api_key = (
                os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or ""
            ).strip().strip('"').strip("'")

            if os.getenv("DEBUG_AUTH") == "1":
                key_present = bool(api_key)
                key_prefix = api_key[:7] if api_key else ""
                key_len = len(api_key) if api_key else 0
                print(
                    f"Auth debug: key_present={key_present}, key_prefix={key_prefix}, key_len={key_len}"
                )

            self.api_key = api_key
            if not self.api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... or OPENAI_KEY=... and rerun."
                )
            self.client = OpenAI(api_key=self.api_key)

        self.embeddings_norm = {}
        normalized = isinstance(embeddings, dict) and embeddings.get("_normalized") is True
        for key, arr in (embeddings or {}).items():
            if key.startswith("_"):
                continue
            if arr is None:
                continue
            if normalized:
                mat = arr
                if mat.dtype != np.float32:
                    mat = mat.astype(np.float32, copy=False)
                if not mat.flags['C_CONTIGUOUS']:
                    mat = np.ascontiguousarray(mat)
                self.embeddings_norm[key] = mat
            else:
                mat = np.asarray(arr, dtype=np.float32)
                if not mat.flags['C_CONTIGUOUS']:
                    mat = np.ascontiguousarray(mat)
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.embeddings_norm[key] = mat / norms

        self.workplace_type_list = [
            (job.get("_meta", {}) or {}).get("workplace_type") or "" for job in self.jobs
        ]
        self.is_remote_mask = np.array([
            (job.get("_meta", {}) or {}).get("is_remote") is True for job in self.jobs
        ], dtype=bool)
        self.hybrid_mask = np.array([
            ((job.get("_meta", {}) or {}).get("workplace_type") == "Hybrid") for job in self.jobs
        ], dtype=bool)
    
    def embed_query(self, query):
        """Embed a query using OpenAI"""
        cache_key = normalize_query_key(query)
        cached, source = self._query_cache.get(query)
        if cached is not None:
            self._last_embed_source = source
            self._last_embed_cache_hit = True
            self._last_embed_cache_key = cache_key
            self._debug_cache_log(cache_key)
            return cached

        prewarm_cached = self._prewarm_cache.get(cache_key)
        if prewarm_cached is not None:
            self._last_embed_source = "mem"
            self._last_embed_cache_hit = True
            self._last_embed_cache_key = cache_key
            self._debug_cache_log(cache_key)
            return prewarm_cached
        if os.getenv("OFFLINE_MODE") == "1":
            vec = np.asarray(self._offline_query_vector(query), dtype=np.float32)
            if self._prewarm_write or cache_key not in self._prewarm_keys:
                self._query_cache.put(query, vec)
                if self._prewarm_write and cache_key in self._prewarm_keys:
                    self._query_cache.save(force=True)
            else:
                self._prewarm_cache[cache_key] = vec
            self._last_embed_source = "fresh"
            self._last_embed_cache_hit = False
            self._last_embed_cache_key = cache_key
            self._debug_cache_log(cache_key)
            return vec
        response = self.client.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        
        tokens = response.usage.prompt_tokens
        cost = (tokens / 1_000_000) * 0.02
        
        tracker.log_call("embed_query", tokens, 0, cost)

        self._log_usage(
            feature="query_embedding",
            model="text-embedding-3-small",
            input_text=query,
            response=response,
        )
        
        vec = np.asarray(response.data[0].embedding, dtype=np.float32)
        if self._prewarm_write or cache_key not in self._prewarm_keys:
            self._query_cache.put(query, vec)
            if self._prewarm_write and cache_key in self._prewarm_keys:
                self._query_cache.save(force=True)
            else:
                self._query_cache.save()
        else:
            self._prewarm_cache[cache_key] = vec
        self._last_embed_source = "fresh"
        self._last_embed_cache_hit = False
        self._last_embed_cache_key = cache_key
        self._debug_cache_log(cache_key)
        return vec

    def _debug_cache_log(self, cache_key):
        if self._debug_cache:
            entries = self._query_cache.size()
            print(
                f"cache_debug: key={cache_key} hit={str(self._last_embed_cache_hit).lower()} "
                f"source={self._last_embed_source} entries={entries}/{self._query_cache.max_size}"
            )

    def _log_query_cache_evictions(self):
        evictions = self._query_cache.evictions()
        if evictions and should_log_cache_debug():
            print(f"Query cache evicted {evictions} entries (max={self._query_cache.max_size})")

    def mark_prewarm_keys(self, queries):
        for q in queries:
            key = normalize_query_key(q)
            self._prewarm_keys.add(key)

    def get_query_cache_keys_mru(self):
        return self._query_cache.keys_mru()

    def get_query_cache_size(self):
        return self._query_cache.size()

    def get_query_cache_max(self):
        return self._query_cache.max_size

    def get_query_cache_stats(self):
        return self._query_cache.get_stats()

    def _log_usage(self, feature, model, input_text, response):
        """Append a safe JSONL usage record to tokens_report.txt."""
        input_chars = len(input_text) if input_text is not None else 0
        total_tokens = None
        request_id = None

        usage = getattr(response, "usage", None)
        if usage is not None:
            total_tokens = getattr(usage, "total_tokens", None)
        request_id = getattr(response, "id", None)

        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "feature": feature,
            "model": model,
            "input_chars": input_chars,
            "total_tokens": total_tokens,
            "request_id": request_id,
        }

        with open("tokens_report.txt", "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def _offline_query_vector(self, query):
        """Generate a deterministic offline query vector without OpenAI."""
        dim = None
        for key in ("explicit", "inferred", "company"):
            emb = self.embeddings.get(key)
            if emb is not None and hasattr(emb, "shape") and len(emb.shape) == 2:
                dim = emb.shape[1]
                break
        if dim is None:
            raise RuntimeError(
                "OFFLINE_MODE is enabled but embeddings are not available to build offline query vectors."
            )

        digest = hashlib.sha1(str(query).encode("utf-8")).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.normal(size=dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec
    
    def search(self, query, top_k=20):
        """Search for jobs matching query"""
        if self._emoji_ok:
            print(f"\n🔍 Searching for: {query}")
        else:
            print(f"\nSearching for: {query}")
        t_embed_start = datetime.now(timezone.utc)
        query_embedding = self.embed_query(query)
        t_embed = (datetime.now(timezone.utc) - t_embed_start).total_seconds()
        q = query.lower()
        wants_remote = "remote" in q
        wants_python = "python" in q
        query_terms = self._tokenize(query)
        role_tokens = set()
        for phrase in ROLE_PHRASES:
            if phrase in q:
                role_tokens.update(self._tokenize(phrase))
        role_tokens.update(query_terms & ROLE_KEYWORDS)
        role_tokens.difference_update({"job", "jobs", "role", "roles"})
        role_gate_tokens = set()
        if {"data", "science", "scientist"} & role_tokens:
            role_gate_tokens.update({"data", "science", "scientist", "analytics", "analyst", "engineer"})
        if "product" in role_tokens:
            role_gate_tokens.update({"product", "manager", "pm"})
        if {"ml", "machine", "learning"} & role_tokens:
            role_gate_tokens.update({"ml", "machine", "learning", "engineer"})
        if {"engineer", "engineering", "developer", "software"} & role_tokens:
            role_gate_tokens.update({"engineer", "engineering", "developer", "software"})
        if {"data", "science", "scientist", "ml", "machine", "learning"} & role_tokens:
            role_gate_tokens.update(ROLE_NEAR_MISS_TOKENS)
        
        query_vec = np.asarray(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            q_norm = 1.0
        query_vec = query_vec / q_norm

        vector_start = datetime.now(timezone.utc)
        # Search against all 3 embeddings (normalized dot product)
        scores_explicit = self.embeddings_norm['explicit'] @ query_vec
        scores_inferred = self.embeddings_norm['inferred'] @ query_vec
        scores_company = self.embeddings_norm['company'] @ query_vec
        
        # Combine with weights
        combined_scores = (
            0.4 * scores_explicit +
            0.4 * scores_inferred +
            0.2 * scores_company
        )
        vector_scores = combined_scores.copy()

        if role_tokens or query_terms:
            title_scores = np.zeros(len(self.jobs), dtype=np.float32)
            for idx, job in enumerate(self.jobs):
                meta = job.get("_meta", {}) or {}
                title = (meta.get("title") or job.get("title") or "").lower()
                if not title:
                    continue
                if role_tokens:
                    if any(tok in title for tok in role_gate_tokens):
                        title_scores[idx] += 0.25
                    else:
                        title_scores[idx] -= 0.15
                if query_terms:
                    hits = sum(1 for tok in query_terms if tok in title)
                    if hits:
                        title_scores[idx] += min(0.15 * hits, 0.60)
            combined_scores = combined_scores + title_scores

        mission_bonus = np.zeros(len(self.jobs), dtype=np.float32)
        lexical_bonus = np.zeros(len(self.jobs), dtype=np.float32)

        t_filters_start = datetime.now(timezone.utc)
        candidate_mask = np.ones(len(self.jobs), dtype=bool)
        if wants_remote:
            mask = self.is_remote_mask | self.hybrid_mask
            if not mask.any():
                mask = self.hybrid_mask
            candidate_mask = mask
        t_filters = (datetime.now(timezone.utc) - t_filters_start).total_seconds()

        available = int(candidate_mask.sum()) if wants_remote else len(self.jobs)
        if available == 0:
            results = []
            if self._emoji_ok:
                print(f"✅ Found {len(results)} results")
            else:
                print(f"Found {len(results)} results")
            self.last_search_stats = {
                "candidates_k": 0,
                "vector_time_s": 0.0,
                "rerank_time_s": 0.0,
            }
            return results

        top_k_eff = min(top_k, available)
        candidate_k = min(int(os.getenv("CANDIDATE_K", "600")), available)

        if wants_remote:
            combined_scores[~candidate_mask] = -np.inf
        candidate_indices = np.argpartition(combined_scores, -candidate_k)[-candidate_k:]
        vector_time = (datetime.now(timezone.utc) - vector_start).total_seconds()

        rerank_start = datetime.now(timezone.utc)
        t_mission_s = 0.0
        t_lexical_s = 0.0
        t_role_gate_s = 0.0
        why_flags_by_idx = {}
        for idx in candidate_indices:
            job = self.jobs[idx]
            mission_start = time.perf_counter()
            mission_match, mission_reason = self._mission_match(job)
            t_mission_s += time.perf_counter() - mission_start
            if mission_match:
                mission_bonus[idx] = 0.08
                meta = job.get("_meta")
                if isinstance(meta, dict):
                    meta["mission_match"] = True
            lexical_start = time.perf_counter()
            lexical_bonus[idx], flags = self._lexical_boost(
                job,
                query_terms,
                wants_remote=wants_remote,
                wants_python=wants_python,
                mission_reason=mission_reason,
            )
            t_lexical_s += time.perf_counter() - lexical_start
            if role_tokens:
                role_start = time.perf_counter()
                meta = job.get("_meta", {}) or {}
                title = meta.get("title") or job.get("title") or ""
                title_tokens = self._tokenize(str(title))
                if title_tokens:
                    if title_tokens & role_gate_tokens:
                        lexical_bonus[idx] += 0.33
                    else:
                        lexical_bonus[idx] -= 0.55
                    overlap_count = len(title_tokens & role_tokens)
                    if overlap_count:
                        lexical_bonus[idx] += min(0.15 * overlap_count, 0.30)
                else:
                    lexical_bonus[idx] -= 0.25
                flags["role_title_hit"] = bool(title_tokens & role_gate_tokens) if title_tokens else False
                flags["role_title_delta"] = round(float(lexical_bonus[idx]), 3)
                t_role_gate_s += time.perf_counter() - role_start
            if flags:
                why_flags_by_idx[idx] = flags
            if self._explain:
                meta = job.setdefault("_meta", {})
                meta["_score_debug"] = {
                    "vector": round(float(vector_scores[idx]), 3),
                    "lexical": round(float(lexical_bonus[idx]), 3),
                    "mission": round(float(mission_bonus[idx]), 3),
                    "role_gate": (
                        "+" if (role_tokens and why_flags_by_idx.get(idx, {}).get("role_title_hit"))
                        else ("-" if role_tokens else "n/a")
                    ),
                }
        combined_scores = combined_scores + mission_bonus + lexical_bonus
        if wants_remote:
            combined_scores[~candidate_mask] = -np.inf

        top_indices = np.argpartition(combined_scores, -top_k_eff)[-top_k_eff:]
        top_indices = top_indices[np.argsort(combined_scores[top_indices])[::-1]]
        results = [self.jobs[i] for i in top_indices]
        for i in top_indices:
            meta = self.jobs[i].get("_meta")
            if isinstance(meta, dict):
                meta["why_flags"] = why_flags_by_idx.get(i, {})
                if self._explain and "_score_debug" in meta:
                    meta["_score_debug"]["total"] = round(float(combined_scores[i]), 3)
        rerank_time = (datetime.now(timezone.utc) - rerank_start).total_seconds()

        if os.getenv("ROLE_INTENT_DEBUG") == "1" and role_tokens:
            top_hits = []
            top_deltas = []
            for i in top_indices[:10]:
                meta = self.jobs[i].get("_meta", {}) or {}
                title = meta.get("title") or self.jobs[i].get("title") or ""
                title_tokens = self._tokenize(str(title))
                hit = bool(title_tokens & role_gate_tokens) if title_tokens else False
                top_hits.append("H" if hit else "M")
                if len(top_deltas) < 3:
                    delta = why_flags_by_idx.get(i, {}).get("role_title_delta", 0.0)
                    top_deltas.append((title, delta))
            print(f"role_intent: tokens={sorted(role_tokens)} gate={sorted(role_gate_tokens)}")
            print(f"role_intent: top10_title_hits={' '.join(top_hits)}")
            for title, delta in top_deltas:
                print(f"role_intent: top3_delta={delta} title={title}")
        
        if self._emoji_ok:
            print(f"✅ Found {len(results)} results")
        else:
            print(f"Found {len(results)} results")
        self.last_search_stats = {
            "candidates_k": int(candidate_k),
            "candidate_count": int(len(candidate_indices)),
            "vector_time_s": vector_time,
            "rerank_time_s": rerank_time,
            "t_embed_s": t_embed,
            "t_vector_s": vector_time,
            "t_lexical_s": rerank_time,
            "t_mission_s": t_mission_s,
            "t_lexical_inner_s": t_lexical_s,
            "t_role_gate_s": t_role_gate_s,
            "t_filters_s": t_filters,
            "t_format_s": 0.0,
            "embed_cache_hit": getattr(self, "_last_embed_cache_hit", False),
            "embed_cache_key": getattr(self, "_last_embed_cache_key", ""),
            "embed_source": getattr(self, "_last_embed_source", "fresh"),
        }
        return results
    
    def format_result(self, job, rank=1):
        """Format a job result for display"""
        meta = job.get("_meta", {}) or {}
        title = html.unescape(meta.get("title") or job.get("title") or "Unknown Title")
        company_raw = meta.get("company")
        company = html.unescape(company_raw) if company_raw else "Unknown Company"
        location = html.unescape(meta.get("location") or "")
        workplace = html.unescape(meta.get("workplace_type") or "")

        bracket = None
        workplace_lc = workplace.lower()
        if any(k in workplace_lc for k in ["remote", "hybrid", "onsite", "on-site", "on site"]):
            bracket = workplace
        elif location:
            bracket = location
        else:
            bracket = "Location not specified"

        suffix = " (mission-aligned)" if meta.get("mission_match") else ""
        why_flags = meta.get("why_flags") or {}
        why_parts = []
        if why_flags.get("remote_match"):
            why_parts.append("remote match")
        if why_flags.get("python_match"):
            why_parts.append("python match")
        if why_flags.get("mission_reason"):
            why_parts.append(why_flags.get("mission_reason"))
        why_text = f" (why: {', '.join(why_parts)})" if why_parts else ""

        explain_block = ""
        if self._explain:
            dbg = meta.get("_score_debug", {})
            if dbg:
                explain_block = (
                    f"\n     ↳ score={dbg.get('total')} "
                    f"(vector={dbg.get('vector')}, "
                    f"lexical={dbg.get('lexical')}, "
                    f"mission={dbg.get('mission')}, "
                    f"role_gate={dbg.get('role_gate')}, "
                    f"cache={getattr(self, '_last_embed_source', 'fresh')})"
                )

        return f"{rank}. {title} @ {company} [{bracket}]{suffix}{why_text}{explain_block}"

    def _mission_match(self, job):
        nonprofit = self._is_nonprofit_signal(job)
        if nonprofit:
            return True, "nonprofit signal"
        if self._mission_in_description(job):
            return True, "mission in description"
        return False, None

    def _is_nonprofit_signal(self, job):
        meta = job.get("_meta", {}) or {}
        if meta.get("company_is_nonprofit") is True:
            return True

        v7 = job.get("v7_processed_job_data", {}) or {}
        company_profile = v7.get("company_profile", {}) or {}
        org_types = company_profile.get("organization_types") or []
        if isinstance(org_types, str):
            org_types = [org_types]
        org_types_text = " ".join([str(t) for t in org_types]).lower()
        if any(k in org_types_text for k in ["nonprofit", "non-profit", "charity", "ngo", "not for profit"]):
            return True

        company_name = (meta.get("company") or "").lower()
        if any(k in company_name for k in MISSION_COMPANY_TOKENS):
            return True

        return False

    def _mission_in_description(self, job):
        meta = job.get("_meta", {}) or {}
        desc_text = meta.get("description_lc") or ""
        return any(k in desc_text for k in [
            "nonprofit",
            "charity",
            "donation",
            "humanitarian",
            "public health",
            "climate justice",
            "environmental justice",
            "refugee",
            "education equity",
            "social impact",
        ])

    def _tokenize(self, text):
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _job_text(self, job):
        meta = job.get("_meta", {}) or {}
        return meta.get("text_blob_lc") or ""

    def _workplace_type(self, job):
        meta = job.get("_meta", {}) or {}
        wt = meta.get("workplace_type") or ""
        if wt:
            return wt
        fields = [
            job.get("workplace_type"),
            job.get("workplace"),
            job.get("remote_status"),
            job.get("location_type"),
            job.get("v7_processed_job_data", {}).get("remote_status"),
            job.get("v7_processed_job_data", {}).get("work_arrangement", {}).get("workplace_type"),
            job.get("v7_processed_job_data", {}).get("work_arrangement", {}).get("workplace_model"),
        ]
        text = " ".join(str(f) for f in fields if f).lower()
        if any(k in text for k in ["remote", "work from home", "wfh"]):
            return "Remote"
        if "hybrid" in text:
            return "Hybrid"
        if any(k in text for k in ["onsite", "on-site", "on site"]):
            return "Onsite"
        return ""

    def _is_remote_job(self, job):
        wt = self._workplace_type(job).lower()
        return wt in ("remote", "hybrid")

    def _lexical_boost(self, job, query_terms, wants_remote=False, wants_python=False, mission_reason=None):
        if not query_terms:
            return 0.0, {}
        text = self._job_text(job)
        boost = 0.0
        flags = {}

        wt = self._workplace_type(job)
        if wants_remote:
            if wt == "Remote":
                boost += 0.20
                flags["remote_match"] = True
            elif wt == "Hybrid":
                boost += 0.05
                flags["remote_match"] = True
            elif wt == "Onsite":
                boost -= 0.50
            else:
                boost -= 0.15

        if wants_python:
            if "python" in text:
                boost += 0.12
                flags["python_match"] = True
            else:
                boost -= 0.08

        if any(k in query_terms for k in ["backend", "api", "platform"]):
            if any(k in text for k in ["backend", "api", "fastapi", "django", "flask"]):
                boost += 0.05

        if mission_reason:
            flags["mission_reason"] = mission_reason

        return boost, flags
    
    def print_results(self, results):
        """Pretty print results"""
        t_format_start = datetime.now(timezone.utc)
        for i, job in enumerate(results, 1):
            print(f"  {self.format_result(job, i)}")
        t_format = (datetime.now(timezone.utc) - t_format_start).total_seconds()
        if getattr(self, "last_search_stats", None) is not None:
            self.last_search_stats["t_format_s"] = t_format
