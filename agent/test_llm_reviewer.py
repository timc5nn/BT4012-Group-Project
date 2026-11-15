"""Tests for the LLM reviewer's agent loop, using a scripted backend (no API calls).

Run with: python -m pytest agent -q
"""

import json

import pandas as pd
import pytest

from agent.data import RAW_FILES, TransactionStore
from agent.config import DATA_DIR
from agent.llm import ALL_TOOLS, HISTORY_TOOLS, ChatResponse, LLMReviewer, parse_decision, prepare_tool_arguments, validate_evidence
from agent.schemas import Case
from agent.tools import CaseTools

ANSWER = {"decision": "fraud", "confidence": 0.9, "evidence": ["$900 vs usual $40"], "case_note": "Out of character."}
PRECOLLECTED = [{"name": name, "arguments": {"limit": 20} if name == "get_transaction_history" else {}}
                for name in ALL_TOOLS]  # every enabled tool runs before the first model turn


def answer_for(case):
    return {
        **ANSWER,
        "evidence": [f"The flagged amount was {case.amount:.2f}."],
    }


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


def test_all_tools_are_collected_before_decision(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([answer(json.dumps(answer_for(case)))])
    reviewer = LLMReviewer(backend)
    tools = make_tools()
    decision = reviewer.review(case, tools)

    assert decision.decision == "fraud" and decision.confidence == 0.9
    assert tools.calls == PRECOLLECTED
    assert reviewer.last_usage == {"model": "test-model", "turns": 1, "input_tokens": 100, "output_tokens": 50}
    evidence_message = backend.sent[0]["messages"][2]["content"]
    assert "required_tool_results" in evidence_message
    assert all(name in evidence_message for name in ALL_TOOLS)
    assert "is_fraud" not in evidence_message


def test_score_withheld_when_asked(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([answer(json.dumps(answer_for(case)))])
    LLMReviewer(backend, include_score=False).review(case, make_tools())
    assert "xgb_score" not in backend.sent[0]["messages"][1]["content"]


def test_model_can_select_additional_tool(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([
        tool_call("get_transaction_history", {"limit": 5}),
        answer(json.dumps(answer_for(case))),
    ])
    reviewer = LLMReviewer(backend)
    tools = make_tools()
    reviewer.review(case, tools)
    assert tools.calls == PRECOLLECTED + [{"name": "get_transaction_history", "arguments": {"limit": 5}}]


def test_unsupported_numeric_claim_prompts_for_targeted_tool(case_and_tools):
    case, make_tools = case_and_tools
    invalid = {
        **answer_for(case),
        "evidence": ["The amount was 86.6 times the merchant median."],
    }
    backend = ScriptedBackend([
        answer(json.dumps(invalid)),
        tool_call("get_amount_deviation", {}),
        answer(json.dumps(answer_for(case))),
    ])
    reviewer = LLMReviewer(backend)
    tools = make_tools()
    reviewer.review(case, tools)
    assert tools.calls == PRECOLLECTED + [{"name": "get_amount_deviation", "arguments": {}}]


def test_kyc_tool_removed_from_model_choices(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([answer(json.dumps(answer_for(case)))])
    reviewer = LLMReviewer(backend, allowed_tools=HISTORY_TOOLS)
    tools = make_tools()
    reviewer.review(case, tools)
    assert "get_kyc_profile" not in backend.sent[0]["tools"]
    assert [c["name"] for c in tools.calls] == HISTORY_TOOLS  # only the enabled tools are pre-collected
    assert "annual_income" not in backend.sent[0]["messages"][2]["content"]


def test_malformed_answer_retried_once(case_and_tools):
    case, make_tools = case_and_tools
    valid = answer_for(case)
    backend = ScriptedBackend([answer("I think it's fraud."), answer("```json\n" + json.dumps(valid) + "\n```")])
    assert LLMReviewer(backend).review(case, make_tools()).decision == "fraud"

    backend = ScriptedBackend([answer("no idea"), answer("still no idea"), answer("still no idea")])
    with pytest.raises(ValueError):
        LLMReviewer(backend).review(case, make_tools())


def test_prepare_tool_arguments_drops_identity_and_dates():
    cleaned, dropped = prepare_tool_arguments("get_kyc_profile", {"customer_id": "KYC-00001"})
    assert cleaned == {} and dropped == ["customer_id"]
    cleaned, dropped = prepare_tool_arguments(
        "get_transaction_history", {"customer_id": "KYC-1", "before": "2099-01-01", "limit": 1000}
    )
    assert cleaned == {"limit": 100}
    assert set(dropped) == {"customer_id", "before"}


def test_customer_id_argument_still_returns_the_bound_profile(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([
    answer(json.dumps(answer_for(case))),
    tool_call("get_transaction_history", {"customer_id": "KYC-99999", "limit": 2}),
    answer("Here are the recent transactions."),
    ])
    reviewer = LLMReviewer(backend)
    tools = make_tools()
    reviewer.review(case, tools)
    reviewer.ask(reviewer.last_messages, tools, "Show me two recent transactions.")
    assert tools.calls[:len(PRECOLLECTED)] == PRECOLLECTED
    assert tools.calls[-1] == {"name": "get_transaction_history", "arguments": {"limit": 2}}
    assert "transactions" in backend.sent[1]["messages"][-1]["content"]


def test_follow_up_uses_the_same_case_tools(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([
        answer(json.dumps(answer_for(case))),
        tool_call("get_kyc_profile", {"customer_id": "KYC-99999"}),
        answer("The credit limit is on the KYC profile."),
    ])
    reviewer = LLMReviewer(backend)
    tools = make_tools()
    reviewer.review(case, tools)
    text = reviewer.ask(reviewer.last_messages, tools, "What is the credit limit?")
    assert "credit limit" in text.lower()
    assert {"name": "get_kyc_profile", "arguments": {}} in tools.calls


def test_decision_uses_precollected_tools_without_nudge(case_and_tools):
    case, make_tools = case_and_tools
    backend = ScriptedBackend([answer(json.dumps(answer_for(case)))])
    reviewer = LLMReviewer(backend)
    assert reviewer.review(case, make_tools()).decision == "fraud"
    assert all(message["role"] != "user" or message["content"] != "Call the tools before you decide."
               for message in backend.sent[0]["messages"])


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


def test_parse_decision_accepts_qwen_evidence_objects():
    response = json.dumps({
        **ANSWER,
        "evidence": [{"statement": "The amount was 22.50.", "amount": 22.5}],
    })
    assert parse_decision(response).evidence == ["The amount was 22.50."]


def test_derived_numeric_evidence_is_allowed_only_when_grounded():
    decision = parse_decision(json.dumps({
        **ANSWER,
        "evidence": ["The amount was 1.26 times the category median and the share was 60%."],
    }))
    validate_evidence(decision, {"derived_numeric_values": [1.27, 60]})
    with pytest.raises(ValueError):
        validate_evidence(decision, {"derived_numeric_values": [2.5]})


def test_amount_deviation_ratio_is_grounded():
    decision = parse_decision(json.dumps({
        **ANSWER,
        "evidence": ["The amount was 5.39 times the merchant median."],
    }))
    validate_evidence(decision, {
        "flagged_transaction": {"amount": 1039.23},
        "get_amount_deviation": {"merchant": {"median": 194.22}},
        "derived_numeric_values": [1039.23 / 194.22],
    })
