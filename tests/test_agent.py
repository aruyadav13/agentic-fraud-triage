import pytest
from langchain_core.messages import AIMessage

from fraud_triage.agent import TriageAgent
from fraud_triage.models import RuleReplayModel


def call(name, args, id="1"):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": id}],
        usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
    )


def verdict(confidence=0.9, evidence=None):
    return call(
        "submit_verdict",
        {
            "decision": "fraud",
            "confidence": confidence,
            "reason": "Large amount relative to prior account activity",
            "evidence": evidence if evidence is not None else ["account_history", "velocity"],
        },
    )


class ScriptedModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.seen = []

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        self.seen.append(list(messages))
        value = next(self.responses)
        if isinstance(value, Exception):
            raise value
        return value


def evidence_calls(case="d"):
    return [
        call("account_history", {"transaction_id": case}),
        call("velocity", {"transaction_id": case, "window_hours": 24}),
    ]


def test_valid_evidence_gated_verdict_and_usage(store):
    result = TriageAgent(ScriptedModel([*evidence_calls(), verdict()]), store).investigate("d")
    assert result.decision == "fraud"
    assert result.input_tokens == 30 and result.output_tokens == 15
    assert result.usage_complete


def test_schema_retry_recovers(store):
    model = ScriptedModel(
        [
            call("velocity", {"transaction_id": "d", "window_hours": "24"}),
            *evidence_calls(),
            verdict(),
        ]
    )
    result = TriageAgent(model, store).investigate("d")
    assert result.decision == "fraud"
    assert any(t["event"] == "tool_error" for t in result.trace)
    assert "repair arguments" in model.seen[1][-1].content


@pytest.mark.parametrize("confidence", [0.79, 0.0])
def test_low_confidence_abstains(store, confidence):
    result = TriageAgent(
        ScriptedModel([*evidence_calls(), verdict(confidence)]), store
    ).investigate("d")
    assert result.decision == "review" and "confidence" in result.reason


def test_insufficient_history_abstains(store):
    result = TriageAgent(ScriptedModel([*evidence_calls("a"), verdict()]), store).investigate("a")
    assert result.decision == "review" and "history" in result.reason


def test_missing_evidence_cannot_force_verdict(store):
    result = TriageAgent(ScriptedModel([verdict(evidence=[])]), store).investigate("d")
    assert result.decision == "review" and "Missing" in result.reason


def test_hallucinated_evidence_never_accepted(store):
    result = TriageAgent(ScriptedModel([verdict()] * 3), store).investigate("d")
    assert result.decision == "review" and "retry budget" in result.reason


def test_provider_failure_is_redacted_and_usage_unknown(store):
    result = TriageAgent(ScriptedModel([RuntimeError("secret-token")]), store).investigate("d")
    assert result.decision == "review" and not result.usage_complete
    assert "secret-token" not in result.model_dump_json()


def test_invalid_json_repair(store):
    bad = AIMessage(
        content="",
        invalid_tool_calls=[{"name": "velocity", "args": "{", "id": "bad", "error": "bad json"}],
    )
    result = TriageAgent(ScriptedModel([bad, *evidence_calls(), verdict()]), store).investigate("d")
    assert result.decision == "fraud"
    assert not result.usage_complete


def test_step_budget(store):
    result = TriageAgent(ScriptedModel(evidence_calls()), store, max_steps=2).investigate("d")
    assert result.decision == "review" and "step budget" in result.reason


def test_unknown_tool_and_cross_case_retries_exhaust(store):
    responses = [
        call("shell", {}),
        call("account_history", {"transaction_id": "f"}),
        call("shell", {}),
    ]
    result = TriageAgent(ScriptedModel(responses), store).investigate("d")
    assert result.decision == "review" and not result.evidence


def test_replay_is_runnable(store):
    result = TriageAgent(RuleReplayModel(), store).investigate("f")
    assert result.model_calls == 2
    assert result.evidence == ["account_history", "velocity"]


def test_mixed_verdict_batch_rejected(store):
    mixed = AIMessage(
        content="",
        tool_calls=[
            {"name": "account_history", "args": {"transaction_id": "d"}, "id": "a"},
            {"name": "submit_verdict", "args": verdict().tool_calls[0]["args"], "id": "b"},
        ],
    )
    result = TriageAgent(ScriptedModel([mixed]), store, max_errors=0).investigate("d")
    assert result.decision == "review" and not result.evidence
