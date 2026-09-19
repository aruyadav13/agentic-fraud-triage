"""Chronological benchmark with explicit selective metrics and cost assumptions."""

import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve

from .agent import TriageAgent
from .features import FEATURES, EvidenceStore
from .models import RuleReplayModel, huggingface_model


def metrics(labels, decisions):
    y, d = np.asarray(labels), np.asarray(decisions)
    if len(y) != len(d) or not len(y):
        raise ValueError("Metrics require equally sized nonempty arrays")
    fraud, legit, review = d == "fraud", d == "legitimate", d == "review"
    if not np.all(fraud | legit | review):
        raise ValueError("Unknown decision")
    tp, fp = int(np.sum(fraud & (y == 1))), int(np.sum(fraud & (y == 0)))
    fn, tn = int(np.sum(legit & (y == 1))), int(np.sum(legit & (y == 0)))
    positives = int(y.sum())

    def safe(a, b):
        return float(a / b) if b else None

    return {
        "n": len(y),
        "fraud_cases": positives,
        "automatic_true_positives": tp,
        "automatic_false_positives": fp,
        "automatic_false_negatives": fn,
        "automatic_true_negatives": tn,
        "reviews": int(review.sum()),
        "fraud_in_review": int(np.sum(review & (y == 1))),
        "precision": safe(tp, tp + fp),
        "recall_all_fraud": safe(tp, positives),
        "recall_decided_fraud": safe(tp, tp + fn),
        "automatic_coverage": float((~review).mean()),
        "review_rate": float(review.mean()),
        "selective_accuracy": safe(tp + tn, int((~review).sum())),
    }


def choose_threshold(y, scores):
    p, r, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1))])


def cost_summary(results, human_review_usd, input_per_million=None, output_per_million=None):
    if human_review_usd < 0 or any(
        x is not None and x < 0 for x in [input_per_million, output_per_million]
    ):
        raise ValueError("Cost assumptions cannot be negative")
    n = len(results)
    human = sum(r.decision == "review" for r in results) * human_review_usd
    complete = all(r.usage_complete for r in results)
    known = complete and input_per_million is not None and output_per_million is not None
    model = (
        (
            sum(r.input_tokens for r in results) * input_per_million / 1e6
            + sum(r.output_tokens for r in results) * output_per_million / 1e6
        )
        if known
        else None
    )
    return {
        "input_tokens": sum(r.input_tokens for r in results),
        "output_tokens": sum(r.output_tokens for r in results),
        "usage_complete": complete,
        "estimated_model_usd": model,
        "estimated_human_usd": human,
        "estimated_usd_per_investigation": (model + human) / n if known and n else None,
        "human_review_usd_assumption": human_review_usd,
        "input_usd_per_million_assumption": input_per_million,
        "output_usd_per_million_assumption": output_per_million,
        "excludes": ["hardware", "training", "network", "human outcome quality"],
        "billing_note": "Token estimates are not provider invoices; unknown usage stays null",
    }


def run(
    frame,
    output: Path,
    backend="replay",
    max_cases=100,
    seed=42,
    flag_quantile=0.9,
    model_id=None,
    human_review_usd=3.0,
    input_per_million=None,
    output_per_million=None,
    provenance=None,
):
    if backend not in ("replay", "hf") or max_cases < 1 or not 0 < flag_quantile < 1:
        raise ValueError("Invalid benchmark configuration")
    if len(frame) < 100:
        raise ValueError("At least 100 chronological rows required")
    # Timestamp boundaries keep same-time rows in the same partition.
    times = frame.timestamp
    first, second = times.iloc[int(len(frame) * 0.6)], times.iloc[int(len(frame) * 0.8)]
    train, validation, test = times < first, (times >= first) & (times < second), times >= second
    if min(train.sum(), validation.sum(), test.sum()) < 10:
        raise ValueError("Too few rows or distinct timestamps for chronological split")
    if any(frame.loc[mask, "is_fraud"].nunique() != 2 for mask in [train, validation, test]):
        raise ValueError("Every temporal partition needs both labels; fetch more data")
    store = EvidenceStore(frame)
    x, y = store.features(), frame.is_fraud
    baseline = HistGradientBoostingClassifier(
        max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=seed
    )
    started = time.perf_counter()
    baseline.fit(x.loc[train, FEATURES], y.loc[train])
    training_seconds = time.perf_counter() - started
    validation_scores = baseline.predict_proba(x.loc[validation, FEATURES])[:, 1]
    threshold = choose_threshold(y.loc[validation], validation_scores)
    flag_threshold = float(np.quantile(validation_scores, flag_quantile))
    started = time.perf_counter()
    test_scores = baseline.predict_proba(x.loc[test, FEATURES])[:, 1]
    scoring_seconds = time.perf_counter() - started
    test_indices = frame.index[test].to_numpy()
    eligible = test_indices[test_scores >= flag_threshold]
    if not len(eligible):
        raise ValueError("No flagged test cases; lower flag quantile")
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(eligible, size=min(max_cases, len(eligible)), replace=False))
    score_by_index = dict(zip(test_indices, test_scores))
    model_id = model_id or os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    model = RuleReplayModel() if backend == "replay" else huggingface_model(model_id)
    agent = TriageAgent(model, store)
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with (output / "investigations.jsonl").open("w") as stream:
        for i in selected:
            result = agent.investigate(frame.loc[i, "transaction_id"])
            stream.write(result.model_dump_json() + "\n")
            stream.flush()
            results.append(result)
    with (output / "review_queue.jsonl").open("w") as stream:
        for result in results:
            if result.decision == "review":
                stream.write(result.model_dump_json() + "\n")
    base_all = np.where(test_scores >= threshold, "fraud", "legitimate")
    base_selected = ["fraud" if score_by_index[i] >= threshold else "legitimate" for i in selected]
    cohort_labels = y.loc[selected].to_numpy()
    report = {
        "backend": backend,
        "llm_benchmark": backend == "hf",
        "model_id": (model_id or "Qwen/Qwen2.5-7B-Instruct")
        if backend == "hf"
        else "rule-protocol-fixture",
        "warning": "Replay measures deterministic rules through the agent protocol, NOT an LLM"
        if backend == "replay"
        else "Self-reported confidence is not calibrated; inspect failure and review rates",
        "dataset": provenance or {"kind": "unspecified"},
        "seed": seed,
        "environment": {
            "python": platform.python_version(),
            **{
                name: importlib.metadata.version(name)
                for name in ["scikit-learn", "pandas", "langchain-core", "langchain-huggingface"]
            },
        },
        "split": {
            "method": "chronological 60/20/20; grouped timestamp boundaries",
            "validation_start": str(first),
            "test_start": str(second),
            **{
                name: {"rows": int(mask.sum()), "fraud": int(y.loc[mask].sum())}
                for name, mask in [("train", train), ("validation", validation), ("test", test)]
            },
        },
        "policy": {
            "baseline_threshold": threshold,
            "selection": "validation maximum F1",
            "flag_threshold": flag_threshold,
            "flag_quantile": flag_quantile,
            "min_agent_confidence": 0.8,
            "min_prior_history": 3,
            "features": FEATURES,
        },
        "cohort": {
            "eligible_flagged_test": len(eligible),
            "investigated": len(selected),
            "selection": "uniform sample without replacement, fixed seed",
            "transaction_ids": frame.loc[selected, "transaction_id"].tolist(),
        },
        "baseline_all_test": {
            **metrics(y.loc[test], base_all),
            "average_precision": float(average_precision_score(y.loc[test], test_scores)),
        },
        "baseline_same_cohort": metrics(cohort_labels, base_selected),
        "agent_same_cohort": metrics(cohort_labels, [r.decision for r in results]),
        "cost": cost_summary(
            results,
            human_review_usd,
            0.0 if backend == "replay" else input_per_million,
            0.0 if backend == "replay" else output_per_million,
        ),
        "baseline_cost": {
            "estimated_usd_per_investigation": 0.0,
            "basis": "No LLM tokens or human review in the forced binary baseline",
            "excludes": ["hardware", "training", "network", "cost of incorrect decisions"],
        },
        "runtime": {
            "baseline_training_seconds": training_seconds,
            "baseline_test_scoring_seconds": scoring_seconds,
            "baseline_seconds_per_test_transaction": scoring_seconds / len(test_indices),
            "agent_total_seconds": sum(r.elapsed_seconds for r in results),
            "agent_mean_seconds": float(np.mean([r.elapsed_seconds for r in results])),
        },
        "failures": {
            "review_reasons": sorted({r.reason for r in results if r.decision == "review"}),
            "provider_error_cases": sum(
                any(t["event"] == "provider_error" for t in r.trace) for r in results
            ),
        },
        "limitations": [
            "Public data are simulated, not real bank transactions",
            "Temporal history uses prior unlabeled events, including earlier test rows",
            "No human outcomes are simulated or credited as correct",
            "No confidence calibration or production fraud guarantees",
            "Agent metrics concern sampled flags, not all transactions",
        ],
    }
    # CSV hash ties the full report to the exact canonical input even without a source manifest.
    report["canonical_frame_sha256"] = hashlib.sha256(
        frame.to_csv(index=False).encode()
    ).hexdigest()
    (output / "benchmark.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    write_markdown(report, output / "benchmark.md")
    return report


def write_markdown(report, path):
    def fmt(value):
        return "N/A" if value is None else f"{value:.3f}"

    lines = [
        "# Benchmark",
        "",
        f"Backend: **{report['backend']}**. {report['warning']}",
        "",
        "| System / population | N | Precision | Recall (all fraud) | Coverage | Review rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key in ["baseline_all_test", "baseline_same_cohort", "agent_same_cohort"]:
        m = report[key]
        lines.append(
            f"| {key} | {m['n']} | {fmt(m['precision'])} | {fmt(m['recall_all_fraud'])} | "
            f"{fmt(m['automatic_coverage'])} | {fmt(m['review_rate'])} |"
        )
    lines += [
        "",
        "Recall counts abstained fraud as not automatically detected. No credit is assigned to human review.",
        "Undefined metrics are N/A; see JSON for confusion counts, cohort IDs, provenance and assumptions.",
        "",
        f"Estimated cost/investigation: {report['cost']['estimated_usd_per_investigation']} USD.",
        "Baseline token + human-review cost/investigation: 0 USD (forced binary decisions).",
        "Costs are scenario estimates excluding compute/training, not actual bills.",
        "",
        "## Limitations",
        "",
        *[f"- {x}" for x in report["limitations"]],
    ]
    path.write_text("\n".join(lines) + "\n")
