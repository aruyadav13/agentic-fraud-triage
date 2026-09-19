import json
import os

from langchain_core.messages import AIMessage, ToolMessage


class RuleReplayModel:
    """Deterministic protocol fixture. This is NOT an LLM or evidence of LLM accuracy."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        case = json.loads(messages[1].content)
        results = [json.loads(m.content) for m in messages if isinstance(m, ToolMessage)]
        if len(results) < 2:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "account_history",
                        "args": {"transaction_id": case["transaction_id"]},
                        "id": "history",
                    },
                    {
                        "name": "velocity",
                        "args": {"transaction_id": case["transaction_id"], "window_hours": 24},
                        "id": "velocity",
                    },
                ],
                usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            )
        history, velocity = results[:2]
        ratio = history["current_to_mean_ratio"] or 0
        if ratio >= 5 and case["amount"] >= 250:
            decision, confidence = "fraud", 0.9
        elif ratio < 2 and velocity["prior_count"] < 10:
            decision, confidence = "legitimate", 0.85
        else:
            decision, confidence = "review", 0.5
        return AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "submit_verdict",
                    "id": "verdict",
                    "args": {
                        "decision": decision,
                        "confidence": confidence,
                        "reason": f"Rule fixture: amount/mean={ratio:.2f}; prior 24h count={velocity['prior_count']}",
                        "evidence": ["account_history", "velocity"],
                    },
                }
            ],
            usage_metadata={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        )


def huggingface_model(model_id=None):
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Set HUGGINGFACEHUB_API_TOKEN (or HF_TOKEN) for the HF backend")
    endpoint = HuggingFaceEndpoint(
        repo_id=model_id or os.getenv("HF_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        task="text-generation",
        max_new_tokens=512,
        do_sample=False,
        timeout=60,
        huggingfacehub_api_token=token,
    )
    return ChatHuggingFace(llm=endpoint)
