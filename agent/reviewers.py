"""Reviewers decide whether a flagged transaction is fraud.

Every reviewer implements the same interface, so the pipeline and the
evaluation do not change when the model behind it does:

    class MyReviewer:
        name = "my-reviewer"
        def review(self, case: Case, tools: CaseTools) -> Decision: ...

The LLM reviewer is in llm.py. To support an API it doesn't cover, write a reviewer
(or an llm.py backend) whose review() method:
  1. sends prompt.SYSTEM_PROMPT, prompt.render_case(case) and tools.definitions
     to the model (definitions are JSON Schema, accepted by most tool-calling APIs);
  2. runs each tool call the model makes with tools.call(name, arguments) and
     returns the result to the model, repeating until it stops calling tools;
  3. parses the final JSON answer with Decision.from_dict(...)
     (schemas.DECISION_SCHEMA can be passed as a structured-output schema).
"""

from typing import Protocol

from .schemas import Case, Decision
from .tools import CaseTools


class Reviewer(Protocol):
    name: str

    def review(self, case: Case, tools: CaseTools) -> Decision: ...


class ScoreThresholdReviewer:
    """No-LLM stand-in: confirms a flag when the XGBoost score clears a threshold.

    With the standalone threshold it reproduces "XGBoost alone" inside the
    pipeline, which checks the plumbing and the evaluation before any model is
    plugged in. It also calls a tool, so the tool path is exercised.
    """

    def __init__(self, threshold):
        self.threshold = threshold
        self.name = f"score-threshold-{threshold:.3f}"

    def review(self, case, tools):
        summary = tools.call("get_spending_summary")
        fraud = case.xgb_score >= self.threshold
        recent = summary.get("recent_activity", {})
        return Decision(
            decision="fraud" if fraud else "legitimate",
            confidence=case.xgb_score,
            evidence=[
                f"XGBoost score {case.xgb_score:.3f} vs threshold {self.threshold:.3f}",
                f"{recent.get('transactions_last_24h', 0)} transactions in the previous 24 hours",
            ],
            case_note="Decided by the XGBoost score alone (baseline reviewer, no LLM).",
        )
