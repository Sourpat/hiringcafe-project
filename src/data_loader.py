import json
import pickle
import os
import numpy as np

CACHE_VERSION = 2
COMPANY_PATHS = [
    "v7_processed_job_data.company_profile.name",
    "job_information.company_name",
    "v7_processed_job_data.company_name",
    "company.name",
    "company",
    "company_name",
]
LOCATION_PATHS = [
    "v5_processed_job_data.formatted_workplace_location",
    "v7_processed_job_data.work_arrangement.workplace_locations[0].city",
    "v7_processed_job_data.work_arrangement.workplace_locations[0].state",
    "v7_processed_job_data.work_arrangement.workplace_locations[0].country_code",
]
WORKPLACE_PATHS = [
    "v7_processed_job_data.work_arrangement.workplace_type",
    "v7_processed_job_data.work_arrangement.workplace_model",
    "v7_processed_job_data.work_arrangement.workplace_locations[0].kind",
    "v7_processed_job_data.remote_status",
    "job_information.remote_status",
    "remote_status",
    "workplace_type",
    "workplace",
    "work_style",
]

class JobDataLoader:
    def __init__(self, jsonl_path, cache_path='data/jobs_index.pkl'):
        self.jsonl_path = jsonl_path
        self.cache_path = cache_path
        self.jobs = []
        self.embeddings = {
            'explicit': None,
            'inferred': None,
            'company': None
        }
        self.load_source = ""
        self.loaded_limit = None
    
    def load(self, fast_demo=False, fast_n=None):
        """Load jobs from cache or JSONL"""
        if os.getenv("REBUILD_CACHE") == "1":
            print("REBUILD_CACHE=1 set, rebuilding cache from JSONL...")
        elif fast_demo:
            limit = int(fast_n) if fast_n else 20000
            fast_meta_path = self._fast_meta_path(limit)
            fast_emb_path = self._fast_emb_path(limit)
            old_fast_cache_path = self._fast_cache_path(limit)

            if os.path.exists(fast_meta_path) and os.path.exists(fast_emb_path):
                print(f"Loading FAST_DEMO meta from {fast_meta_path}...")
                with open(fast_meta_path, 'rb') as f:
                    self.jobs = pickle.load(f)
                emb_all = np.load(fast_emb_path, mmap_mode="r")
                self.embeddings = {
                    "explicit": emb_all[0],
                    "inferred": emb_all[1],
                    "company": emb_all[2],
                    "_normalized": True,
                }
                self.load_source = "fast cache v2"
                self.loaded_limit = limit
                print(f"✅ Loaded {len(self.jobs)} jobs from fast cache v2")
                return

            if os.path.exists(old_fast_cache_path):
                print(f"Loading legacy FAST_DEMO cache from {old_fast_cache_path}...")
                with open(old_fast_cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if isinstance(cached, dict) and "jobs" in cached and "embeddings" in cached:
                    self.jobs = cached.get("jobs", [])
                    self.embeddings = cached.get("embeddings", {})
                else:
                    self.jobs, self.embeddings = cached

                self._strip_embeddings_from_jobs(self.jobs)
                self.jobs = self._slim_jobs(self.jobs)
                emb_all = np.stack([
                    np.asarray(self.embeddings["explicit"], dtype=np.float32),
                    np.asarray(self.embeddings["inferred"], dtype=np.float32),
                    np.asarray(self.embeddings["company"], dtype=np.float32),
                ], axis=0)
                emb_all = self._normalize_emb_stack(emb_all)
                os.makedirs(os.path.dirname(fast_meta_path), exist_ok=True)
                with open(fast_meta_path, 'wb') as f:
                    pickle.dump(self.jobs, f)
                np.save(fast_emb_path, emb_all)
                self.embeddings = {
                    "explicit": emb_all[0],
                    "inferred": emb_all[1],
                    "company": emb_all[2],
                    "_normalized": True,
                }
                self.load_source = "fast cache v1->v2"
                self.loaded_limit = limit
                print(f"✅ Migrated FAST_DEMO cache to v2 ({fast_meta_path}, {fast_emb_path})")
                return

            print(f"Loading FAST_DEMO jobs from {self.jsonl_path} (limit={limit})...")
            jobs, emb_all = self.load_jobs_from_jsonl(self.jsonl_path, limit=limit, return_embeddings=True)
            self.jobs = self._slim_jobs(jobs)
            emb_all = self._normalize_emb_stack(emb_all)
            self.embeddings = {
                "explicit": emb_all[0],
                "inferred": emb_all[1],
                "company": emb_all[2],
                "_normalized": True,
            }
            self._strip_embeddings_from_jobs(self.jobs)
            os.makedirs(os.path.dirname(fast_meta_path), exist_ok=True)
            with open(fast_meta_path, 'wb') as f:
                pickle.dump(self.jobs, f)
            np.save(fast_emb_path, emb_all)
            print(f"✅ Cached FAST_DEMO v2 to {fast_meta_path} and {fast_emb_path}")
            self.load_source = "fast jsonl build v2"
            self.loaded_limit = limit
            return
        elif os.path.exists(self.cache_path):
            print(f"Loading cached jobs from {self.cache_path}...")
            with open(self.cache_path, 'rb') as f:
                cached = pickle.load(f)
            if isinstance(cached, dict) and "jobs" in cached and "embeddings" in cached:
                self.jobs = cached.get("jobs", [])
                self.embeddings = cached.get("embeddings", {})
            else:
                self.jobs, self.embeddings = cached

            migrated = 0
            for job in self.jobs:
                meta = job.get("_meta") or {}
                if (
                    not meta
                    or not meta.get("title")
                    or not meta.get("company")
                    or not meta.get("location")
                    or meta.get("text_blob_lc") is None
                    or meta.get("workplace_type") is None
                    or meta.get("is_remote") is None
                    or meta.get("company_is_nonprofit") is None
                ):
                    job["_meta"] = self._build_meta(job)
                    migrated += 1
            if migrated:
                print(f"✅ Cache migrated: added _meta to {migrated} jobs")
                with open(self.cache_path, 'wb') as f:
                    pickle.dump({"version": CACHE_VERSION, "jobs": self.jobs, "embeddings": self.embeddings}, f)
            print(f"✅ Loaded {len(self.jobs)} jobs from cache")
            self.load_source = "cache"
            return
        
        print(f"Loading jobs from {self.jsonl_path}...")
        self.jobs = self.load_jobs_from_jsonl(self.jsonl_path, limit=None)
        print(f"✅ Loaded {len(self.jobs)} jobs total")
        self._build_embeddings()
        
        # Cache for future runs
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump({"version": CACHE_VERSION, "jobs": self.jobs, "embeddings": self.embeddings}, f)
        print(f"✅ Cached to {self.cache_path}")
        self.load_source = "jsonl"
    
    def get_jobs(self):
        return self.jobs
    
    def get_embeddings(self):
        return self.embeddings

    def load_jobs_from_jsonl(self, path, limit=None, return_embeddings=False):
        jobs = []
        exp_list = []
        inf_list = []
        comp_list = []
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                if limit is not None and len(jobs) >= limit:
                    break
                try:
                    job = json.loads(line.strip())
                    if 'v7_processed_job_data' in job:
                        v7 = job['v7_processed_job_data']
                        if all(k in v7 for k in ['embedding_explicit_vector', 'embedding_inferred_vector', 'embedding_company_vector']):
                            job["_meta"] = self._build_meta(job)
                            jobs.append(job)
                            if return_embeddings:
                                exp_list.append(v7['embedding_explicit_vector'])
                                inf_list.append(v7['embedding_inferred_vector'])
                                comp_list.append(v7['embedding_company_vector'])
                except:
                    pass

                if (i + 1) % 10000 == 0:
                    print(f"  Processed {i + 1} jobs...")
        if return_embeddings:
            emb_all = np.stack([
                np.asarray(exp_list, dtype=np.float32),
                np.asarray(inf_list, dtype=np.float32),
                np.asarray(comp_list, dtype=np.float32),
            ], axis=0)
            return jobs, emb_all
        return jobs

    def _fast_cache_path(self, limit):
        base = os.path.dirname(self.cache_path)
        return os.path.join(base, f"jobs_index.fast.{limit}.pkl")

    def _fast_meta_path(self, limit):
        base = os.path.dirname(self.cache_path)
        return os.path.join(base, f"jobs_meta.fast.{limit}.pkl")

    def _fast_emb_path(self, limit):
        base = os.path.dirname(self.cache_path)
        return os.path.join(base, f"jobs_emb.fast.{limit}.npy")

    def _build_embeddings(self):
        self.embeddings['explicit'] = np.array([
            job['v7_processed_job_data']['embedding_explicit_vector']
            for job in self.jobs
        ])
        self.embeddings['inferred'] = np.array([
            job['v7_processed_job_data']['embedding_inferred_vector']
            for job in self.jobs
        ])
        self.embeddings['company'] = np.array([
            job['v7_processed_job_data']['embedding_company_vector']
            for job in self.jobs
        ])
        print("✅ Extracted all embeddings into memory")

    def _normalize_emb_stack(self, emb_all):
        emb_all = emb_all.astype(np.float32, copy=False)
        for i in range(emb_all.shape[0]):
            norms = np.linalg.norm(emb_all[i], axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb_all[i] = emb_all[i] / norms
        return emb_all

    def _slim_jobs(self, jobs):
        slim = []
        for job in jobs:
            meta = job.get("_meta", {}) or {}
            slim.append({
                "job_id": meta.get("job_id") or job.get("job_id") or job.get("id"),
                "title": meta.get("title") or job.get("title"),
                "company": meta.get("company") or job.get("company") or job.get("company_name"),
                "location": meta.get("location") or job.get("location"),
                "workplace_type": meta.get("workplace_type") or job.get("workplace_type"),
                "url": meta.get("url") or job.get("url") or job.get("apply_url"),
                "_meta": meta,
            })
        return slim

    def _strip_embeddings_from_jobs(self, jobs):
        for job in jobs:
            v7 = job.get("v7_processed_job_data")
            if isinstance(v7, dict):
                v7.pop("embedding_explicit_vector", None)
                v7.pop("embedding_inferred_vector", None)
                v7.pop("embedding_company_vector", None)
            v5 = job.get("v5_processed_job_data")
            if isinstance(v5, dict):
                v5.pop("embedding_explicit_vector", None)
                v5.pop("embedding_inferred_vector", None)
                v5.pop("embedding_company_vector", None)

    def _first_non_empty(self, *values):
        for value in values:
            if value is None:
                continue
            if isinstance(value, str):
                if value.strip():
                    return value.strip()
            else:
                return value
        return None

    def _pick_first(self, job, keys):
        for k in keys:
            v = job.get(k)
            if not v:
                continue
            if isinstance(v, dict):
                for kk in ["name", "title", "value", "text", "display"]:
                    nested = v.get(kk)
                    if nested:
                        return nested
            return v
        return None

    def _get_path(self, job, path):
        cur = job
        for part in path.split("."):
            if "[" in part and part.endswith("]"):
                name, idx_part = part.split("[", 1)
                idx = idx_part[:-1]
                if name:
                    if not isinstance(cur, dict):
                        return None
                    cur = cur.get(name)
                if not isinstance(cur, list):
                    return None
                try:
                    cur = cur[int(idx)]
                except (ValueError, IndexError):
                    return None
            else:
                if not isinstance(cur, dict):
                    return None
                cur = cur.get(part)
            if cur is None:
                return None
        return cur

    def get_by_path(self, job, dotted_path):
        return self._get_path(job, dotted_path)

    def _pick_first_path(self, job, paths):
        for path in paths:
            v = self._get_path(job, path)
            if not v:
                continue
            if isinstance(v, dict):
                for kk in ["name", "title", "value", "text", "display"]:
                    nested = v.get(kk)
                    if nested:
                        return nested
            return v
        return None

    def _build_meta(self, job):
        job_info = job.get("job_information", {}) or {}
        v7 = job.get("v7_processed_job_data", {}) or {}

        title = (
            self._pick_first_path(job, ["job_information.title", "v7_processed_job_data.title"])
            or self._pick_first(job, ["title", "job_title", "position_title", "role", "position"])
        )

        company = self._pick_first_path(job, COMPANY_PATHS)

        location = None
        formatted_location = self._get_path(job, "v5_processed_job_data.formatted_workplace_location")
        city = self._get_path(job, "v7_processed_job_data.work_arrangement.workplace_locations[0].city")
        state = self._get_path(job, "v7_processed_job_data.work_arrangement.workplace_locations[0].state")
        country = self._get_path(job, "v7_processed_job_data.work_arrangement.workplace_locations[0].country_code")
        city = str(city).strip() if city else None
        state = str(state).strip() if state else None
        country = str(country).strip() if country else None
        if city or state or country:
            parts = [p for p in [city, state, country] if p]
            location = ", ".join(parts)
        if not location:
            location = self._pick_first_path(job, LOCATION_PATHS)

        workplace_type = self._pick_first_path(job, WORKPLACE_PATHS)

        def _normalize_workplace_type(raw):
            if not raw:
                return ""
            text = str(raw).lower()
            if any(k in text for k in ["remote", "work from home", "wfh"]):
                return "Remote"
            if "hybrid" in text:
                return "Hybrid"
            if any(k in text for k in ["onsite", "on-site", "on site"]):
                return "Onsite"
            return ""

        normalized_workplace = _normalize_workplace_type(workplace_type)
        is_remote = normalized_workplace in ("Remote", "Hybrid")

        v5_company = job.get("v5_processed_company_data", {}) or {}
        company_is_nonprofit = v5_company.get("is_non_profit") is True
        if not company_is_nonprofit:
            company_profile = v7.get("company_profile", {}) or {}
            org_types = company_profile.get("organization_types") or []
            if isinstance(org_types, str):
                org_types = [org_types]
            org_text = " ".join([str(t) for t in org_types]).lower()
            company_is_nonprofit = any(k in org_text for k in ["nonprofit", "non-profit", "charity", "ngo", "not for profit"])

        url = (
            self._pick_first_path(job, ["job_information.apply_url", "job_information.url"])
            or self._pick_first(job, ["apply_url", "url", "apply_link", "job_url", "application_url"])
        )

        job_id = self._pick_first(job, ["job_id", "id", "job_uuid"]) or self._pick_first(v7, ["job_id", "id", "job_uuid"])

        def _to_str(val):
            return str(val).strip() if val is not None else None

        description = (
            job.get("description")
            or job.get("job_description")
            or job.get("description_text")
            or job_info.get("description")
            or v7.get("job_description")
            or ""
        )
        industry = (
            v7.get("company_sector_and_industry")
            or v7.get("industry")
            or job.get("industry")
            or job_info.get("industry")
            or ""
        )
        text_blob = " ".join([
            _to_str(title) or "",
            _to_str(company) or "",
            _to_str(industry) or "",
            str(description)[:2000],
        ]).strip()
        text_blob_lc = text_blob.lower()[:4000]
        description_lc = str(description).lower()[:1200]

        return {
            "job_id": _to_str(job_id),
            "title": _to_str(title),
            "company": _to_str(company),
            "location": _to_str(location),
            "workplace_type": _to_str(normalized_workplace) or _to_str(workplace_type),
            "url": _to_str(url),
            "is_remote": is_remote,
            "company_is_nonprofit": company_is_nonprofit,
            "text_blob_lc": text_blob_lc,
            "description_lc": description_lc,
        }
