"""Pinned public Sparkov CSV; only canonical non-PII fields leave the loader."""

import csv
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE = "dazzle-nu/CIS435-CreditCardFraudDetection"
REVISION = "f5d3f80f0f1dfdab4b8a5537156fb0cfae82c6d8"
URL = f"https://huggingface.co/datasets/{SOURCE}/resolve/{REVISION}/fraudTrain.csv"
COLUMNS = ["transaction_id", "account_id", "timestamp", "amount", "is_fraud"]


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame[COLUMNS].copy()
    if frame.isna().any().any():
        raise ValueError("Null transaction fields are not allowed")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True, errors="raise", format="mixed")
    for col in ["transaction_id", "account_id"]:
        frame[col] = frame[col].astype(str)
        if frame[col].str.len().eq(0).any():
            raise ValueError(f"Empty {col}")
    frame["amount"] = pd.to_numeric(frame.amount, errors="raise")
    if not np.isfinite(frame.amount).all() or (frame.amount < 0).any():
        raise ValueError("Amounts must be finite and nonnegative")
    if not frame.is_fraud.isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    frame["is_fraud"] = frame.is_fraud.astype(int)
    if frame.transaction_id.duplicated().any():
        raise ValueError("Transaction IDs must be unique")
    return frame.sort_values(["timestamp", "transaction_id"]).reset_index(drop=True)


def fetch(path: Path, rows: int = 100_000) -> dict:
    if rows < 100:
        raise ValueError("Fetch at least 100 rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    # Stream a chronological prefix; no pickle execution and no full dataset download.
    with urllib.request.urlopen(URL, timeout=90) as response:
        reader = csv.DictReader(io.TextIOWrapper(response, encoding="utf-8"))
        for row in reader:
            records.append(
                {
                    "transaction_id": row["trans_num"],
                    "account_id": hashlib.sha256(row["cc_num"].encode()).hexdigest()[:24],
                    "timestamp": row["trans_date_trans_time"],
                    "amount": float(row["amt"]),
                    "is_fraud": int(row["is_fraud"]),
                }
            )
            if len(records) >= rows:
                break
    frame = validate(pd.DataFrame(records))
    frame.to_csv(path, index=False)
    manifest = {
        "source": SOURCE,
        "revision": REVISION,
        "url": URL,
        "rows": len(frame),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "kind": "public_simulated_sparkov",
        "selection": "first N source rows, then time sorted",
        "time_min": str(frame.timestamp.min()),
        "time_max": str(frame.timestamp.max()),
        "fraud_count": int(frame.is_fraud.sum()),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def demo(rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Local synthetic fixture, explicitly not a public-dataset benchmark."""
    if rows < 100:
        raise ValueError("Generate at least 100 rows")
    rng = np.random.default_rng(seed)
    amount = rng.lognormal(3.5, 0.8, rows)
    fraud = rng.random(rows) < 0.04
    amount[fraud] *= rng.uniform(3, 12, fraud.sum())
    return validate(
        pd.DataFrame(
            {
                "transaction_id": [f"demo-{i}" for i in range(rows)],
                "account_id": [f"acct-{i}" for i in rng.integers(0, 80, rows)],
                "timestamp": pd.date_range("2025-01-01", periods=rows, freq="min", tz="UTC"),
                "amount": amount.round(2),
                "is_fraud": fraud.astype(int),
            }
        )
    )


def read(path: Path) -> pd.DataFrame:
    return validate(pd.read_csv(path, dtype={"transaction_id": str, "account_id": str}))
