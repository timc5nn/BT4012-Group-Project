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

from .llm import LLMReviewer
from .prompt import TACTIC_SYSTEM_PROMPT
from .retriever import LexicalRetriever
from .schemas import Case, Decision
from .tactics import HISTORY_LIMIT, TACTIC_TOOLS, match_tactics, render_tactic_check, shown_tactics
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
            confidence=case.xgb_score if fraud else 1 - case.xgb_score,
            evidence=[
                f"XGBoost score {case.xgb_score:.3f} vs threshold {self.threshold:.3f}",
                f"{recent.get('transactions_last_24h', 0)} transactions in the previous 24 hours",
            ],
            case_note="Decided by the XGBoost score alone (baseline reviewer, no LLM).",
        )


class RAGReviewer(LLMReviewer):
    """LLM reviewer augmented with retrieved, label-free investigation guidance."""

    def __init__(self, backend, retriever=None, top_k=3, **kwargs):
        super().__init__(backend, **kwargs)
        self.retriever = retriever or LexicalRetriever()
        self.top_k = top_k
        self.name = f"rag-{self.name}"
        self.last_retrieval = []

    def review(self, case, tools):
        query = (
            f"transaction category {case.category} merchant {case.merchant} "
            f"amount {case.amount} customer fraud review xgboost score {case.xgb_score}"
        )
        self.last_retrieval = self.retriever.retrieve(query, self.top_k)
        # Only the guidance text goes to the model. Document ids and retrieval scores stay
        # in the run log: the pilot showed the model citing a retrieval score as evidence.
        self.additional_context = "\n\n".join(doc.text for doc in self.last_retrieval)
        try:
            decision = super().review(case, tools)
            decision_index = next(
                index for index, event in enumerate(self.last_transcript)
                if event.get("role") == "assistant" and event.get("decision")
            )
            self.last_transcript.insert(
                decision_index,
                {
                    "role": "note",
                    "text": "Retrieved guidance: " + ", ".join(
                        f"{document.document_id} ({document.score:.2f})"
                        for document in self.last_retrieval
                    ),
                },
            )
            return decision
        finally:
            self.additional_context = ""


class TacticReviewer(LLMReviewer):
    """LLM reviewer guided by fraud-tactic documents chosen by checklist (see tactics.py).

    Code runs the tools the checklists need, ticks every tactic's checklist, and sends
    the model up to `top_k` matched tactics with their ticks, plus the innocent-
    explanations note. The model then judges the fit; it is not required to call tools.
    """

    MODEL_HISTORY_ROWS = 20  # the checklists read 100 rows; the model sees the latest 20

    def __init__(self, backend, top_k=3, **kwargs):
        kwargs.setdefault("system_prompt", TACTIC_SYSTEM_PROMPT)
        kwargs.setdefault("require_secondary_tool", False)
        super().__init__(backend, **kwargs)
        self.top_k = top_k
        self.name = f"tactic-{self.name}"
        self.last_retrieval = []  # TacticMatch objects; the pipeline logs them per case
        self._history_rows = []

    def _collect_required_tools(self, tools):
        evidence = {}
        calls = [{"name": name, "arguments": dict(arguments)}
                 for name, arguments in TACTIC_TOOLS if name in self.allowed_tools]
        self.last_transcript.append({
            "role": "assistant",
            "text": "Collected the evidence the tactic checklists need before analysis.",
            "tool_calls": calls,
        })
        self._history_rows = []
        for call in calls:
            name, arguments = call["name"], call["arguments"]
            result = tools.call(name, arguments)
            self.last_transcript.append({"role": "tool", "name": name, "arguments": arguments, "result": result})
            if name == "get_transaction_history" and isinstance(result, dict) and "transactions" in result:
                self._history_rows = result["transactions"]
                result = {
                    "transactions": self._history_rows[-self.MODEL_HISTORY_ROWS:],
                    "note": f"latest {self.MODEL_HISTORY_ROWS} of the {len(self._history_rows)} rows "
                            f"(up to {HISTORY_LIMIT}) used by the tactic check",
                }
            evidence[name] = result
        return evidence

    def review(self, case, tools):
        decision = super().review(case, tools)
        if decision.decision == "fraud" and not any(m.matched for m in self.last_retrieval):
            # A fraud alert must name a known tactic. In a fit-row check, transactions matching
            # no tactic were mostly legitimate, so an unexplained alert goes to a person instead.
            note = ("No known fraud tactic matched, so this alert needs an analyst's decision. "
                    + decision.case_note)
            decision = Decision(decision="needs_human_review", confidence=decision.confidence,
                                evidence=decision.evidence, case_note=" ".join(note.split()[:120]))
            self.last_transcript.append({"role": "note", "text": "Fraud call without a matched tactic "
                                         "changed to needs_human_review."})
            for event in reversed(self.last_transcript):
                if event.get("decision"):
                    event["decision"] = decision.to_dict()
                    break
        return decision

    def _evidence_intro(self):
        return (
            "Tool results for this case, with deterministic findings. They are the only sources "
            "of customer-specific facts. If a result is missing or reports an error, say so "
            "instead of guessing:\n"
        )

    def _guidance_message(self, case, evidence):
        self.last_retrieval = match_tactics(case, evidence, self._history_rows)
        best, others = shown_tactics(self.last_retrieval, self.top_k)
        self.last_transcript.append({
            "role": "note",
            "text": "Tactic check: " + (", ".join(f"{m.title} ({m.strength})" for m in [best, *others])
                                         if best else "no known tactic matched"),
        })
        return render_tactic_check(self.last_retrieval, self.top_k)
