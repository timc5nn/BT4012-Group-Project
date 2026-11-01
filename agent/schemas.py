"""What a reviewer is shown (Case) and what it must return (Decision)."""

from dataclasses import asdict, dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Case:
    """One flagged transaction, as a reviewer sees it. No label, no card number."""

    case_id: str  # the transaction's trans_num
    customer_id: str  # KYC record ID, e.g. KYC-00042
    time: pd.Timestamp
    amount: float
    merchant: str
    category: str
    xgb_score: float

    def to_dict(self, include_score=True):
        d = {
            "case_id": self.case_id,
            "customer_id": self.customer_id,
            "time": self.time.isoformat(),
            "amount": round(self.amount, 2),
            "merchant": self.merchant,
            "category": self.category,
        }
        if include_score:
            d["xgb_score"] = round(self.xgb_score, 4)
        return d


# JSON Schema for the reviewer's answer; usable as a structured-output schema by most LLM APIs.
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["fraud", "legitimate"]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "case_note": {"type": "string"},
    },
    "required": ["decision", "confidence", "evidence", "case_note"],
    "additionalProperties": False,
}


@dataclass
class Decision:
    decision: str  # "fraud" (raise the alert) or "legitimate" (clear the flag)
    confidence: float  # 0–1, the reviewer's confidence that the transaction is fraud
    evidence: list = field(default_factory=list)
    case_note: str = ""

    def __post_init__(self):
        if self.decision not in ("fraud", "legitimate"):
            raise ValueError(f"decision must be 'fraud' or 'legitimate', got {self.decision!r}")
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")

    @classmethod
    def from_dict(cls, d):
        return cls(
            decision=d["decision"],
            confidence=float(d["confidence"]),
            evidence=list(d.get("evidence", [])),
            case_note=d.get("case_note", ""),
        )

    def to_dict(self):
        return asdict(self)
