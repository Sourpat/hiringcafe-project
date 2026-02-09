import json
import os
from collections import defaultdict, Counter

COMPANY_HINTS = [
    "company",
    "employer",
    "organization",
    "org",
    "hiring_organization",
    "hiringorganization",
]

LOCATION_HINTS = [
    "location",
    "city",
    "state",
    "country",
    "region",
    "joblocation",
    "job_location",
]

def _walk(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{prefix}.{k}" if prefix else k
            yield from _walk(v, path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            path = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _walk(v, path)
    else:
        yield prefix, obj

def _as_str(val):
    if val is None:
        return None
    s = str(val).strip()
    return s if s else None

def _is_url(text):
    return "http://" in text or "https://" in text

def _probe(jobs, limit):
    company_counts = Counter()
    location_counts = Counter()
    company_samples = defaultdict(list)
    location_samples = defaultdict(list)

    for job in jobs[:limit]:
        for path, value in _walk(job):
            val = _as_str(value)
            if not val or _is_url(val):
                continue
            val_len = len(val)
            path_lc = path.lower()

            if any(h in path_lc for h in COMPANY_HINTS) and 2 <= val_len <= 80:
                company_counts[path] += 1
                if len(company_samples[path]) < 3 and val not in company_samples[path]:
                    company_samples[path].append(val[:60])

            if any(h in path_lc for h in LOCATION_HINTS) and 2 <= val_len <= 120:
                location_counts[path] += 1
                if len(location_samples[path]) < 3 and val not in location_samples[path]:
                    location_samples[path].append(val[:60])

    return company_counts, location_counts, company_samples, location_samples

def main():
    jsonl_path = os.path.join("data", "jobs.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"Missing data file: {jsonl_path}")
        return

    jobs = []
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i >= 1000:
                break
            try:
                jobs.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                continue

    company_counts, location_counts, company_samples, location_samples = _probe(jobs, 1000)

    print("\nCompany field frequency (non-empty):")
    for path, cnt in company_counts.most_common(15):
        print(f"  {path}: {cnt}/1000")

    for path, _ in company_counts.most_common(5):
        print(f"\nSamples for {path}:")
        for sample in company_samples[path]:
            print(f"  - {sample}")

    print("\nLocation field frequency (non-empty):")
    for path, cnt in location_counts.most_common(15):
        print(f"  {path}: {cnt}/1000")

    for path, _ in location_counts.most_common(5):
        print(f"\nSamples for {path}:")
        for sample in location_samples[path]:
            print(f"  - {sample}")

if __name__ == "__main__":
    main()
