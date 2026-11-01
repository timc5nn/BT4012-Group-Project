"""Tests for the LLM reviewer's agent loop, using a scripted backend (no API calls).

Run with: python -m pytest agent -q
"""

import json

import pandas as pd
import pytest

from agent.data import RAW_FILES, TransactionStore
from agent.config import DATA_DIR
from agent.llm import HISTORY_TOOLS, ChatResponse, LLMReviewer, parse_decision
from agent.schemas import Case
from agent.tools import CaseTools

ANSWER = {"decision": "fraud", "confidence": 0.9, "evidence": ["$900 vs usual $40"], "case_note": "Out of character."}


class ScriptedBackend:
    """Replays a fixed list of replies; records what the reviewer sent."""

    provider, model = "scripted", "test-model"

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []

    system = staticmethod(lambda text: {"role": "system", "content": text})
    user = staticmethod(lambda text: {"role": "user", "content": text})
    tool_result = staticmethod(lambda call_id, result: {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)})

    def chat(self, messages, tool_definitions):
        self.sent.append({"messages": list(messages), "tools": [d["name"] for d in tool_definitions]})
        return self.replies.pop(0)


def tool_call(name, arguments, call_id="c1"):
    return ChatResponse(text="", tool_calls=[{"id": call_id, "name": name, "arguments": arguments}],
                        assistant_message={"role": "assistant", "content": ""}, input_tokens=100, output_tokens=10)


def answer(text):
    return ChatResponse(text=text, tool_calls=[], assistant_message={"role": "assistant", "content": text},
                        input_tokens=100, output_tokens=50)


@pytest.fixture(scope="module")
def store():
    return TransactionStore.load()


@pytest.fixture(scope="module")
def case_and_tools(store):
    row = pd.read_csv(DATA_DIR / RAW_FILES[0], usecols=["cc_num", "trans_num", "trans_date_trans_time", "amt",
                                                         "merchant", "category"], nrows=50_000).iloc[-1]
    time = pd.Timestamp(row.trans_date_trans_time)
    case = Case(row.trans_num, store.customer_id(row.cc_num), time, row.amt,
                row.merchant.removeprefix("fraud_"), row.category, 0.7)
    return case, lambda: CaseTools(store, row.cc_num, time)


def test_tool_loop_then_decision(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([
        tool_call("get_spending_summary", {}),
        tool_call("get_transaction_history", {"limit": 5}, call_id="c2"),
        answer(json.dumps(ANSWER)),
    ])
    reviewer = LLMReviewer(backend)
    tools = make_tools()
    decision = reviewer.review(case, tools)

    assert decision.decision == "fraud" and decision.confidence == 0.9
    assert [c["name"] for c in tools.calls] == ["get_spending_summary", "get_transaction_history"]
    assert reviewer.last_usage == {"model": "test-model", "turns": 3, "input_tokens": 300, "output_tokens": 70}
    # tool results went back to the model, and contain no labels
    tool_messages = [m for m in backend.sent[-1]["messages"] if m["role"] == "tool"]
    assert len(tool_messages) == 2 and all("fraud" not in m["content"].lower() for m in tool_messages)


def test_score_withheld_when_asked(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([answer(json.dumps(ANSWER))])
    LLMReviewer(backend, include_score=False).review(case, make_tools())
    assert "xgb_score" not in backend.sent[0]["messages"][1]["content"]


def test_kyc_tool_removed_and_refused(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([tool_call("get_kyc_profile", {}), answer(json.dumps(ANSWER))])
    reviewer = LLMReviewer(backend, allowed_tools=HISTORY_TOOLS)
    tools = make_tools()
    reviewer.review(case, tools)
    assert "get_kyc_profile" not in backend.sent[0]["tools"]
    assert tools.calls == []  # the refused call never reached the tools
    assert "not available" in backend.sent[1]["messages"][-1]["content"]


def test_malformed_answer_retried_once(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([answer("I think it's fraud."), answer("```json\n" + json.dumps(ANSWER) + "\n```")])
    assert LLMReviewer(backend).review(case, make_tools()).decision == "fraud"

    backend = ScriptedBackend([answer("no idea"), answer("still no idea")])
    with pytest.raises(ValueError):
        LLMReviewer(backend).review(case, make_tools())


def test_gives_up_after_max_turns(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([tool_call("get_spending_summary", {}, call_id=str(i)) for i in range(3)])
    with pytest.raises(RuntimeError):
        LLMReviewer(backend, max_turns=3).review(case, make_tools())


def test_parse_decision_rejects_bad_values():
    with pytest.raises(ValueError):
        parse_decision(json.dumps({**ANSWER, "decision": "maybe"}))
    with pytest.raises(ValueError):
        parse_decision(json.dumps({**ANSWER, "confidence": 3}))
