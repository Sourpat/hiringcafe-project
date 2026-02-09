import os
import json
from datetime import datetime, timezone

def workplace_type(job):
    meta = job.get("_meta") or {}
    fields = [
        meta.get("workplace_type"),
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

def is_remote(job):
    wt = workplace_type(job).lower()
    return wt in ("remote", "hybrid")

def is_hybrid(job):
    return workplace_type(job) == "Hybrid"
MISSION_KEYWORDS = [
    "social good",
    "mission-driven",
    "nonprofit",
    "climate",
    "sustainability",
    "public health",
    "education",
    "equity",
    "charity",
    "foundation",
    "impact",
]
from openai import OpenAI
from src.token_tracker import tracker

class SearchContext:
    def __init__(self, search_engine, initial_top_k=100, mission_top_k=200, emoji_ok=True):
        self.search_engine = search_engine
        self._emoji_ok = emoji_ok
        self.conversation = []
        self.current_results = []
        self.intent = {}
        self.initial_top_k = initial_top_k
        self.mission_top_k = mission_top_k
        self.client = None
        if os.getenv("OFFLINE_MODE") != "1":
            api_key = (
                os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY") or ""
            ).strip().strip('"').strip("'")
            if not api_key:
                raise RuntimeError(
                    "OPENAI_API_KEY is not set. Create a .env with OPENAI_API_KEY=... or OPENAI_KEY=... and rerun."
                )
            self.client = OpenAI(api_key=api_key)
        self.filters = {
            'role': None,
            'remote': None,
            'company_mission': None,
            'experience_level': None,
            'industry': None,
            'company_type': None
        }
    
    def parse_intent(self, query):
        """Use LLM to extract structured intent"""
        if os.getenv("OFFLINE_MODE") == "1":
            return {}
        prompt = f"""Extract job search intent from: "{query}"

Return ONLY valid JSON (no markdown):
{{
    "role": "job title/role or null",
    "remote": true/false/null,
    "company_mission": "mission focus or null",
    "experience_level": "junior/mid/senior or null",
    "industry": "industry or null",
    "company_type": "startup/enterprise/nonprofit or null"
}}"""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        tokens_in = response.usage.prompt_tokens
        tokens_out = response.usage.completion_tokens
        cost = (tokens_in / 1_000_000) * 0.15 + (tokens_out / 1_000_000) * 0.60
        
        tracker.log_call("parse_intent", tokens_in, tokens_out, cost)

        self._log_usage(
            feature="intent_parse",
            model="gpt-4o-mini",
            input_text=prompt,
            response=response,
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except:
            return {}

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
    
    def apply_filters(self, results):
        """Filter results based on current filters"""
        filtered = results
        
        # Filter by remote status
        if self.filters['remote'] is not None:
            remote_capable = [j for j in filtered if is_remote(j) or is_hybrid(j)]
            print(f"Remote-capable jobs in current set: {len(remote_capable)}")

            if self.filters['remote']:
                filtered = [j for j in filtered if is_remote(j) or is_hybrid(j)]
            else:
                filtered = [j for j in filtered if workplace_type(j) == "Onsite"]
        
        # Filter by experience level
        if self.filters['experience_level']:
            filtered = [j for j in filtered 
                       if self.filters['experience_level'].lower() in 
                          j.get('v7_processed_job_data', {}).get('seniority_level', '').lower()]
        
        return filtered
    
    def refine(self, query):
        """Refine search based on conversation"""
        self.conversation.append(query)
        query_lower = query.lower()
        if any(k in query_lower for k in MISSION_KEYWORDS):
            self.intent["mission_focus"] = True

        intent = self.parse_intent(query)
        
        # Update filters
        for key in self.filters:
            if intent.get(key) is not None:
                self.filters[key] = intent[key]
        
        base_query = self.conversation[0] if self.conversation else query
        if self.intent.get("mission_focus"):
            expanded_query = (
                base_query
                + " mission-driven nonprofit social impact climate public health education"
            )
            self.current_results = self.search_engine.search(expanded_query, top_k=self.mission_top_k)
        elif not self.current_results:
            self.current_results = self.search_engine.search(query, top_k=self.initial_top_k)
        
        # Apply filters
        filtered = self.apply_filters(self.current_results)
        
        active_filters = [f for f, v in self.filters.items() if v is not None]
        if self._emoji_ok:
            print(f"💡 Active filters: {active_filters}")
            print(f"📊 Results: {len(filtered)} matching")
        else:
            print(f"Active filters: {active_filters}")
            print(f"Results: {len(filtered)} matching")
        
        # Update results
        self.current_results = filtered
        
        return filtered[:20]
