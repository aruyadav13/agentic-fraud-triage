from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class CaseQuery(StrictModel):
    transaction_id: str = Field(min_length=1, max_length=100)


class VelocityQuery(CaseQuery):
    window_hours: Literal[1, 24] = 24


class Verdict(StrictModel):
    decision: Literal["fraud", "legitimate", "review"]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: str = Field(min_length=8, max_length=1500)
    evidence: list[Literal["account_history", "velocity"]] = Field(max_length=2)


class Investigation(BaseModel):
    transaction_id: str
    decision: Literal["fraud", "legitimate", "review"]
    confidence: float
    reason: str
    evidence: list[str]
    trace: list[dict] = Field(default_factory=list)
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usage_complete: bool = True
    elapsed_seconds: float = 0
