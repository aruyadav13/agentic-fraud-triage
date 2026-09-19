import json

import pytest

from fraud_triage.benchmark import cost_summary, metrics, run
from fraud_triage.data import demo
from fraud_triage.schemas import Investigation


def test_abstention_is_not_free_recall():
    m = metrics([1, 1, 0, 0], ["fraud", "review", "legitimate", "fraud"])
    assert m["precision"] == 0.5
    assert m["recall_all_fraud"] == 0.5
    assert m["recall_decided_fraud"] == 1.0
    assert m["fraud_in_review"] == 1
    assert m["automatic_coverage"] == 0.75


def test_all_review_undefined_metrics_are_null():
    m = metrics([0, 1], ["review", "review"])
    assert m["precision"] is None and m["selective_accuracy"] is None
    assert m["recall_all_fraud"] == 0


def test_unknown_usage_not_zero_cost():
    result = Investigation(
        transaction_id="a",
        decision="review",
        confidence=0,
        reason="Failed call",
        evidence=[],
        usage_complete=False,
    )
    m = cost_summary([result], 3, 1, 2)
    assert m["estimated_human_usd"] == 3
    assert m["estimated_model_usd"] is None
    assert m["estimated_usd_per_investigation"] is None


def test_negative_cost_rejected():
    with pytest.raises(ValueError):
        cost_summary([], -1)


def test_end_to_end_reproducible_cohort(tmp_path):
    frame = demo(1500)
    a = run(frame, tmp_path / "a", max_cases=8, provenance={"kind": "test"})
    b = run(frame, tmp_path / "b", max_cases=8, provenance={"kind": "test"})
    assert a["cohort"] == b["cohort"]
    assert a["baseline_same_cohort"] == b["baseline_same_cohort"]
    assert a["agent_same_cohort"] == b["agent_same_cohort"]
    assert a["llm_benchmark"] is False
    assert a["cohort"]["investigated"] <= 8
    report = json.loads((tmp_path / "a" / "benchmark.json").read_text())
    assert report["split"]["train"]["rows"] == 900
    assert report["baseline_same_cohort"]["n"] == report["agent_same_cohort"]["n"]
    assert len((tmp_path / "a" / "investigations.jsonl").read_text().splitlines()) == 8


def test_single_class_rejected(tmp_path):
    frame = demo(100)
    frame["is_fraud"] = 0
    with pytest.raises(ValueError, match="both labels"):
        run(frame, tmp_path)
