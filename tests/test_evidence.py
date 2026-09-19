import pandas as pd
import pytest
from pydantic import ValidationError

from fraud_triage.data import validate
from fraud_triage.features import EvidenceStore


def test_strictly_prior_same_time_and_future_excluded(store):
    assert store.history("d") == {
        "prior_count": 3,
        "prior_mean_amount": 20.0,
        "current_to_mean_ratio": 5.0,
    }
    assert store.history("e")["prior_count"] == 3
    assert store.velocity("d", 24)["prior_total_amount"] == 60.0
    assert store.velocity("d", 1)["prior_count"] == 0


def test_window_includes_left_boundary_excludes_current(store):
    assert store.velocity("b", 1)["prior_count"] == 1
    assert store.history("a")["current_to_mean_ratio"] is None


def test_labels_and_future_changes_do_not_change_features(frame):
    before = EvidenceStore(frame).features()
    frame["is_fraud"] = 1 - frame.is_fraud
    frame.loc[frame.transaction_id == "f", "amount"] = 1e9
    after = EvidenceStore(frame).features()
    pd.testing.assert_frame_equal(before.iloc[:5], after.iloc[:5])


def test_cross_case_query_and_bad_arguments_rejected(store):
    history, velocity = store.tools("d")
    with pytest.raises(ValueError, match="active transaction"):
        history.invoke({"transaction_id": "a"})
    for args in [{"transaction_id": 3}, {"transaction_id": "d", "extra": True}]:
        with pytest.raises(ValidationError):
            history.invoke(args)
    with pytest.raises(ValidationError):
        velocity.invoke({"transaction_id": "d", "window_hours": 72})


@pytest.mark.parametrize(
    "column,value",
    [("amount", float("nan")), ("amount", -1), ("amount", float("inf")), ("is_fraud", 2)],
)
def test_bad_input_rejected(frame, column, value):
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        validate(frame)


def test_duplicate_id_rejected(frame):
    frame.loc[1, "transaction_id"] = "a"
    with pytest.raises(ValueError, match="unique"):
        validate(frame)
