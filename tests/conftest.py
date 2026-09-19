import pandas as pd
import pytest

from fraud_triage.data import validate
from fraud_triage.features import EvidenceStore


@pytest.fixture
def frame():
    return validate(
        pd.DataFrame(
            {
                "transaction_id": ["a", "b", "c", "d", "e", "f"],
                "account_id": ["one"] * 6,
                "timestamp": [
                    "2025-01-01T00:00Z",
                    "2025-01-01T01:00Z",
                    "2025-01-01T02:00Z",
                    "2025-01-02T00:00Z",
                    "2025-01-02T00:00Z",
                    "2025-01-03T00:00Z",
                ],
                "amount": [10.0, 20.0, 30.0, 100.0, 900.0, 5000.0],
                "is_fraud": [0, 0, 0, 1, 1, 1],
            }
        )
    )


@pytest.fixture
def store(frame):
    return EvidenceStore(frame)
