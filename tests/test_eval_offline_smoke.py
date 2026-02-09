import json
import os

from scripts.eval_offline import run_eval


def test_eval_offline_smoke(tmp_path, monkeypatch):
    monkeypatch.setenv("OFFLINE_MODE", "1")
    monkeypatch.setenv("FAST_DEMO", "1")
    monkeypatch.setenv("FAST_DEMO_N", "20000")

    output_dir = tmp_path / "data"
    report = run_eval(output_dir=str(output_dir))

    json_path = output_dir / "eval_report.json"
    md_path = output_dir / "eval_report.md"

    assert json_path.exists()
    assert md_path.exists()

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "passes" in data
    assert isinstance(data["passes"], list)
    assert data["passes"]
    assert "results" in data["passes"][0]
    assert data["passes"][0]["results"]
    assert isinstance(report, dict)

    for row in data["passes"][0]["results"]:
        rerank = (row.get("timings") or {}).get("rerank")
        notes = row.get("notes") or []
        if rerank is not None and rerank > 0.2:
            assert notes
        else:
            assert rerank is None or rerank <= 0.2