"""Tests for the tactic checklists and the tactic reviewer (no data files or API calls)."""

import json
from types import SimpleNamespace

import pandas as pd

from agent.llm import ChatResponse
from agent.reviewers import CombinedReviewer, TacticReviewer
from agent.schemas import Case
from agent.tactics import TACTIC_TOOLS, TACTICS, TacticMatch, match_tactics, render_tactic_check, shown_tactics

NOW = pd.Timestamp("2020-04-02 01:15:00")


def make_case(amount=940.0, hour=1, category="shopping_pos", merchant="NewShop"):
    return Case("case-1", "KYC-00001", NOW.replace(hour=hour), amount, merchant, category, 0.9)


def history(rows):
    return [{"time": t.isoformat(), "amount": a, "merchant": m, "category": c} for t, a, m, c in rows]


def quiet_customer():
    """45 daytime purchases at 3 regular merchants over the last 45 days, all under $80."""
    rows = []
    for day in range(45, 0, -1):
        rows.append((NOW - pd.Timedelta(days=day) + pd.Timedelta(hours=13), 30 + day % 40,
                     ["Grocer", "Cafe", "Fuel"][day % 3], "grocery_pos"))
    return rows


def evidence(rows, first_time=True, merchant_visits=0, merchant_max=None):
    amounts = pd.Series([r[1] for r in rows], dtype=float)
    merchant = ({"merchant": "NewShop", "first_time": True, "earlier_transactions": 0} if first_time else
                {"merchant": "NewShop", "first_time": False, "earlier_transactions": merchant_visits,
                 "amounts": {"median": merchant_max, "p95": merchant_max, "max": merchant_max}})
    return {
        "get_spending_summary": {"amount_overall": {"median": float(amounts.median()),
                                                    "p95": float(amounts.quantile(0.95)),
                                                    "max": float(amounts.max())}},
        "get_amount_deviation": {"overall": {"median": float(amounts.median()), "p95": float(amounts.quantile(0.95))}},
        "get_merchant_history": merchant,
        "get_transaction_history": {"transactions": history(rows)},
    }


def by_id(matches):
    return {m.document_id: m for m in matches}


def test_documents_are_label_free_and_complete():
    for tactic in TACTICS:
        assert "is_fraud" not in tactic.text and tactic.core and tactic.text
        assert "Example (illustrative" in tactic.text


def test_late_night_out_of_character_is_a_strong_match():
    rows = quiet_customer()
    matches = match_tactics(make_case(), evidence(rows))
    late = by_id(matches)["tactic_late_night_out_of_character"]
    assert matches[0].document_id == late.document_id
    assert late.matched and late.strength == "strong"
    assert all(item["present"] for item in late.supporting)  # new merchant, rarely late, largest ever
    assert not any(item["present"] for item in late.against)


def test_core_signals_gate_the_match():
    """A large purchase in the afternoon does not match the late-night tactic, however many extras fit."""
    matches = match_tactics(make_case(hour=14), evidence(quiet_customer()))
    assert not by_id(matches)["tactic_late_night_out_of_character"].matched


def test_innocent_explanation_weakens_the_match():
    rows = quiet_customer()
    late_match = by_id(match_tactics(make_case(), evidence(rows, first_time=False, merchant_visits=5,
                                                           merchant_max=900.0)))
    late = late_match["tactic_late_night_out_of_character"]
    assert late.matched and late.strength in {"moderate", "weak"}
    assert any(i["signal"] == "regular_merchant_similar_amounts" and i["present"] for i in late.against)


def test_ordinary_purchase_matches_nothing():
    matches = match_tactics(make_case(amount=35.0, hour=13), evidence(quiet_customer()))
    assert not any(m.matched for m in matches)
    assert "No known tactic matched" in render_tactic_check(matches)


def test_online_tactic_needs_an_online_category():
    rows = quiet_customer()
    assert by_id(match_tactics(make_case(category="shopping_net"), evidence(rows)))["tactic_online_purchase"].matched
    assert not by_id(match_tactics(make_case(), evidence(rows)))["tactic_online_purchase"].matched


def test_spree_needs_new_merchants_and_high_recent_spend():
    rows = quiet_customer() + [
        (NOW - pd.Timedelta(hours=h), 300.0, f"Spree{h}", cat)
        for h, cat in [(9, "shopping_net"), (7, "misc_pos"), (5, "gas_transport"), (3, "travel")]
    ]
    spree = by_id(match_tactics(make_case(), evidence(rows)))["tactic_stolen_card_spree"]
    assert spree.matched
    assert any(i["signal"] == "many_categories_24h" and i["present"] for i in spree.supporting)


def test_missing_tool_results_are_unknown_not_present():
    matches = match_tactics(make_case(), {})
    assert not any(m.matched for m in matches)
    text = render_tactic_check(matches)
    assert "Best-matching tactic" not in text and "No known tactic matched" in text  # unmatched tactics are not shown
    late = by_id(matches)["tactic_late_night_out_of_character"]
    assert late.core[0]["present"] is None


class OneAnswerBackend:
    provider, model = "scripted", "test-model"
    system = staticmethod(lambda text: {"role": "system", "content": text})
    user = staticmethod(lambda text: {"role": "user", "content": text})
    tool_result = staticmethod(lambda call_id, result: {"role": "tool", "content": json.dumps(result)})

    def __init__(self, reply):
        self.reply, self.sent = reply, []

    def chat(self, messages, tool_definitions):
        self.sent.append(list(messages))
        return ChatResponse(text=self.reply, tool_calls=[], assistant_message={"role": "assistant", "content": self.reply},
                            input_tokens=100, output_tokens=20)


class FakeTools:
    def __init__(self, data):
        self.data, self.calls = data, []

    def call(self, name, arguments=None):
        self.calls.append({"name": name, "arguments": arguments or {}})
        return self.data[name]


class FakeRetriever:
    last_error = None

    def __init__(self):
        self.query = None

    def retrieve(self, query, top_k):
        self.query = query
        return [SimpleNamespace(
            document_id="policy_evidence",
            text="Use only customer-specific tool evidence.",
            score=0.9,
        )]


def test_tactic_reviewer_sends_matched_tactics_and_decides_without_extra_tools():
    reply = json.dumps({"decision": "fraud", "confidence": 0.9, "evidence": ["late and large"],
                        "case_note": "Late-night out-of-character spending."})
    backend = OneAnswerBackend(reply)
    rows = quiet_customer()
    tools = FakeTools(evidence(rows))
    reviewer = TacticReviewer(backend, require_secondary_tool=False)
    decision = reviewer.review(make_case(), tools)

    assert decision.decision == "fraud"
    assert [c["name"] for c in tools.calls] == [name for name, _ in TACTIC_TOOLS]
    messages = backend.sent[0]
    assert "known fraud tactics" in messages[0]["content"]  # tactic system prompt
    guidance = messages[-1]["content"]
    assert "Best-matching tactic: Out-of-character late-night spending - strong match" in guidance
    assert "Common innocent explanations" in guidance
    evidence_message = messages[2]["content"]
    assert evidence_message.count('"merchant"') <= TacticReviewer.MODEL_HISTORY_ROWS + 2  # history trimmed for the model
    assert reviewer.last_retrieval and reviewer.last_retrieval[0].matched
    assert "is_fraud" not in guidance and "is_fraud" not in evidence_message


def test_combined_reviewer_runs_tactics_before_embedding_rag():
    reply = json.dumps({"decision": "fraud", "confidence": 0.9, "evidence": ["late and large"],
                        "case_note": "Late-night out-of-character spending."})
    retriever = FakeRetriever()
    backend = OneAnswerBackend(reply)
    decision = CombinedReviewer(backend, retriever=retriever, require_secondary_tool=False).review(
        make_case(), FakeTools(evidence(quiet_customer()))
    )

    assert decision.decision == "fraud"
    assert "Out-of-character late-night spending" in retriever.query
    guidance = backend.sent[0][-1]["content"]
    assert "Best-matching tactic" in guidance
    assert "Use only customer-specific tool evidence." in guidance


def test_small_charges_capped_without_a_spree():
    rows = quiet_customer() + [(NOW - pd.Timedelta(hours=2), 4.2, "TinyShop", "misc_pos")]
    matches = by_id(match_tactics(make_case(), evidence(rows)))
    small = matches["tactic_small_charges_in_spree"]
    assert small.matched and not matches["tactic_stolen_card_spree"].matched
    assert small.strength == "moderate" and "capped" in small.capped


def test_fraud_call_without_a_tactic_goes_to_human_review():
    reply = json.dumps({"decision": "fraud", "confidence": 0.9, "evidence": ["odd"], "case_note": "Looks odd."})
    tools = FakeTools(evidence(quiet_customer()))
    reviewer = TacticReviewer(OneAnswerBackend(reply), require_secondary_tool=False)
    decision = reviewer.review(make_case(amount=35.0, hour=13), tools)
    assert decision.decision == "needs_human_review"
    assert decision.case_note.startswith("No known fraud tactic matched")


def test_legitimate_call_without_a_tactic_is_kept():
    reply = json.dumps({"decision": "legitimate", "confidence": 0.9, "evidence": ["usual"], "case_note": "Normal."})
    tools = FakeTools(evidence(quiet_customer()))
    decision = TacticReviewer(OneAnswerBackend(reply), require_secondary_tool=False).review(
        make_case(amount=35.0, hour=13), tools
    )
    assert decision.decision == "legitimate"


def _fake_match(document_id, strength, score):
    core = [{"signal": "x", "present": True, "detail": f"core sign of {document_id}"}]
    return TacticMatch(document_id, document_id, True, strength, score, core, [], [])


def test_only_best_tactic_in_full_and_weak_secondaries_dropped():
    matches = [
        _fake_match("tactic_stolen_card_spree", "strong", 3),
        _fake_match("tactic_small_charges_in_spree", "strong", 2),
        _fake_match("tactic_online_purchase", "moderate", 1),
        _fake_match("tactic_late_night_out_of_character", "weak", 0),
    ]
    best, others = shown_tactics(matches, top_k=3)
    assert best.document_id == "tactic_stolen_card_spree"
    assert [m.document_id for m in others] == ["tactic_small_charges_in_spree", "tactic_online_purchase"]
    text = render_tactic_check(matches)
    assert text.count("Best-matching tactic:") == 1 and text.count("Also matched:") == 2
    assert "core sign of tactic_late_night_out_of_character" not in text
    assert "How it works: once a card is stolen" in text  # full text only for the best match
    assert "How it works: stolen card numbers are easiest" not in text


def test_single_weak_match_is_still_shown():
    best, others = shown_tactics([_fake_match("tactic_online_purchase", "weak", 0)])
    assert best.document_id == "tactic_online_purchase" and others == []
