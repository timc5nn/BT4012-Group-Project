"""Evaluate the fraud reviewer's investigation path and write-up (Lecture 12 metrics).

Reads run logs only. Nothing in agent/ or the pipeline changes. The tools are
deterministic and point-in-time, so each logged tool call is replayed to recover
exactly what the reviewer saw. Labels are not read.

Path metrics, on every case of every run:
    tool recall        required checks that returned data / required checks
    argument accuracy  tool calls with valid, case-bound arguments / tool calls made
    order violations   cases where a step ran before the step it depends on
    gate adherence     cases that had to go to a human and did / cases that had to

Write-up rubric, graded by a local LLM judge on a sample of each run:
    GROUNDING (blocking), HEDGING, COMPLETENESS

Run from the repo root:

    python results/agent_eval/scripts/evaluate_agent.py                    # path metrics, saved runs
    python results/agent_eval/scripts/evaluate_agent.py --judge 20         # + judge on 20 cases per judged run
    python results/agent_eval/scripts/evaluate_agent.py --fresh 20 --seed 2033 --judge 20
        # first review 20 new validation cases with the current reviewer (the dashboard's
        # CombinedReviewer), through agent.pipeline.run, keeping each case's transcript
"""

import argparse
import json
import random
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from agent.config import DATA_DIR, PROCESSED_DIR, RUNS_DIR  # noqa: E402
from agent.data import RAW_FILES, TransactionStore  # noqa: E402
from agent.llm import (  # noqa: E402
    ALL_TOOLS, HUMAN_REVIEW_THRESHOLD, NUMBER_PATTERN, derived_numeric_values, rule_checks,
)
from agent.schemas import Case  # noqa: E402
from agent.tactics import LATE_HOURS, TACTIC_TOOLS, TACTICS_BY_ID  # noqa: E402
from agent.tools import TOOL_DEFINITIONS, CaseTools  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parents[1]
SAVED_RUNS_DIR = EVAL_DIR / "runs"
REPORTS_DIR = EVAL_DIR / "reports"
JUDGE_CACHE = REPORTS_DIR / "judge_cache.jsonl"
FRESH_NAME = "current-combined-qwen8k"
TACTIC_TOOL_NAMES = [name for name, _ in TACTIC_TOOLS]
SCHEMAS = {d["name"]: d["input_schema"] for d in TOOL_DEFINITIONS}

# What code ran before the model's first turn, and whether the design required the model to
# make a lookup of its own. Saved runs come from earlier commits (see results/agent_eval/README.md).
DESIGNS = {
    "pilot-qwen-rag": {"code_tools": ["get_spending_summary"], "own_lookup_required": False},
    "pilot-qwen8k-alltools": {"code_tools": [
        "get_kyc_profile", "get_transaction_history", "get_spending_summary", "get_merchant_history",
        "get_amount_deviation", "get_velocity_profile", "get_location_history", "get_model_explanation",
    ], "own_lookup_required": False},
    "pilot-qwen8k-tactics": {"code_tools": TACTIC_TOOL_NAMES, "own_lookup_required": False},
    "eval100-qwen8k-tactics-v2": {"code_tools": TACTIC_TOOL_NAMES, "own_lookup_required": False},
    "eval100-qwen8k-tactics-v3": {"code_tools": TACTIC_TOOL_NAMES, "own_lookup_required": False},
}
CURRENT_DESIGN = {"code_tools": TACTIC_TOOL_NAMES, "own_lookup_required": True}  # CombinedReviewer today

# The four signals the system prompts tell the reviewer to weigh, and the tools that cover each.
REQUIRED_CHECKS = {
    "amount vs usual": {"get_amount_deviation", "get_spending_summary"},
    "merchant familiarity": {"get_merchant_history"},
    "recent velocity": {"get_velocity_profile", "get_spending_summary"},
    "time-of-day habit": {"get_spending_summary", "get_transaction_history"},
}
OWN_LOOKUP = "model's own lookup"

# Text that code, not the model, adds to a decision. Stripped before grading the write-up.
CODE_NOTES = [
    re.compile(r"Confidence \d+% is below the \d+% automatic-review threshold\. "
               r"A human analyst must make the final decision\.\s*"),
    re.compile(r"No known fraud tactic matched, so this alert needs an analyst's decision\.\s*"),
    re.compile(r"No secondary case tool was called before this decision\.\s*"),
]
# Claims that the flagged merchant itself is new. Plural "new merchants" (the 24-hour spree sign)
# and the small/large charge of a pair are about other transactions, so they do not count.
NEW_MERCHANT_CLAIM = re.compile(
    r"\b(?:a|the) new merchant\b(?!s)|\bmerchant is new to (?:the|this) customer\b(?!:)"
    r"|\bfirst (?:transaction|purchase|visit) (?:with|at) (?:this|the) merchant"
    r"|\bnever (?:used|visited|shopped at) (?:this|the) merchant", re.IGNORECASE)
OTHER_TRANSACTION = re.compile(r"\b(charge|small|large|not|no|without)\b[^.]{0,25}$", re.IGNORECASE)
LATE_CLAIM = re.compile(r"\b(late[- ]night|late at night|after midnight|overnight)\b", re.IGNORECASE)
LOOSE_NUMBER = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Loading runs and rebuilding what the reviewer saw
# ---------------------------------------------------------------------------

def load_run(path):
    records = [json.loads(line) for line in Path(path).open(encoding="utf-8") if line.strip()]
    transcripts_path = Path(path).with_name(Path(path).stem + "_transcripts.jsonl")
    transcripts = {}
    if transcripts_path.exists():
        for line in transcripts_path.open(encoding="utf-8"):
            item = json.loads(line)
            transcripts[item["case_id"]] = item["transcript"]
    return records, transcripts


def run_name(path):
    return Path(path).stem.removesuffix("_val").removesuffix("_test")


def design_for(name):
    return DESIGNS.get(name, CURRENT_DESIGN)


def load_cases(case_ids, split):
    """Raw rows for the logged case ids. Stage 1 has been re-run since some runs, so the
    current flagged_{split}.csv does not contain every logged case; the raw files do."""
    columns = ["trans_num", "cc_num", "trans_date_trans_time", "amt", "merchant", "category"]
    rows = pd.concat([pd.read_csv(DATA_DIR / name, usecols=columns) for name in RAW_FILES], ignore_index=True)
    rows = rows[rows["trans_num"].isin(case_ids)].set_index("trans_num")
    scores = pd.read_csv(PROCESSED_DIR / f"scores_{split}.csv", usecols=["trans_num", "xgb_score"])
    rows = rows.join(scores.set_index("trans_num"), how="left")
    rows["trans_date_trans_time"] = pd.to_datetime(rows["trans_date_trans_time"])
    rows["merchant"] = rows["merchant"].str.removeprefix("fraud_")
    return rows


def make_case(store, row, case_id):
    return Case(case_id=case_id, customer_id=store.customer_id(row.cc_num), time=row.trans_date_trans_time,
                amount=row.amt, merchant=row.merchant, category=row.category,
                xgb_score=row.xgb_score if pd.notna(row.xgb_score) else 0.0)


def trajectory(record, transcript, store, row):
    """Every tool call in order: name, arguments as the model sent them, result, and who made it."""
    if transcript:
        calls = []
        precollect_done = False
        for event in transcript:
            if (event.get("role") == "assistant" and "tool_calls" in event
                    and not event.get("text", "").startswith("Collected")):
                precollect_done = True
            if event.get("role") == "tool":
                calls.append({"name": event["name"], "arguments": event.get("arguments"),
                              "dropped": event.get("dropped", []), "result": event.get("result"),
                              "origin": "model" if precollect_done else "code"})
        return calls

    tools = CaseTools(store, row.cc_num, row.trans_date_trans_time,
                      xgb_score=row.xgb_score if pd.notna(row.xgb_score) else None)
    code_tools = design_for(record["_run"])["code_tools"]
    calls = []
    for i, call in enumerate(record["tool_calls"]):
        origin = "code" if i < len(code_tools) and call["name"] == code_tools[i] else "model"
        calls.append({"name": call["name"], "arguments": call["arguments"], "dropped": [],
                      "result": tools.call(call["name"], call["arguments"]), "origin": origin})
    return calls


def full_facts(store, row):
    """All eight tool results, used only to list the case's red flags."""
    tools = CaseTools(store, row.cc_num, row.trans_date_trans_time,
                      xgb_score=row.xgb_score if pd.notna(row.xgb_score) else None)
    return {name: tools.call(name, {"limit": 100} if name == "get_transaction_history" else {})
            for name in ALL_TOOLS}


def tactic_matches(record):
    return [d for d in record.get("retrieved_documents") or [] if "matched" in d]


def ok(result):
    return isinstance(result, dict) and "error" not in result


# ---------------------------------------------------------------------------
# Path metrics
# ---------------------------------------------------------------------------

def tool_recall(calls, design):
    required = dict(REQUIRED_CHECKS)
    done = {}
    for check, tools in required.items():
        done[check] = any(c["name"] in tools and ok(c["result"]) for c in calls)
    if design["own_lookup_required"]:
        done[OWN_LOOKUP] = any(c["origin"] == "model" and c["name"] not in design["code_tools"]
                               and ok(c["result"]) for c in calls)
    return done


def argument_ok(call):
    schema = SCHEMAS.get(call["name"])
    arguments = call["arguments"]
    if schema is None or not isinstance(arguments, dict) or call.get("dropped"):
        return False
    if set(arguments) - set(schema.get("properties", {})):
        return False
    if "limit" in arguments:
        limit = arguments["limit"]
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            return False
    return not (isinstance(call["result"], dict) and "unexpected arguments" in str(call["result"].get("error", "")))


def order_violations(record, calls, transcript, design):
    found = []
    code_tools = design["code_tools"]
    prefix = [c["name"] for c in calls[:len(code_tools)]]
    if prefix != code_tools:
        found.append("code evidence was not collected first, in order")
    if tactic_matches(record) and not set(TACTIC_TOOL_NAMES) <= set(prefix):
        found.append("tactic check ran without all of its evidence")
    for event in transcript or []:
        if event.get("role") == "note" and event.get("text", "").startswith(
                ("Immediate decision rejected", "No tool result yet")):
            found.append("model tried to decide before the required lookup")
            break
    return found


def gates(record, calls, recall, design):
    """Escalations the case required, and whether it was sent to a person."""
    decision = record.get("decision") or {}
    final = decision.get("decision")
    escalated = final == "needs_human_review"
    required = []
    if record.get("error") or not decision:
        required.append("no decision recorded (crash)")
    else:
        if decision["confidence"] < HUMAN_REVIEW_THRESHOLD:
            required.append(f"confidence below {HUMAN_REVIEW_THRESHOLD:.2f}")
        matches = tactic_matches(record)
        if matches and not any(m["matched"] for m in matches) and (
                final == "fraud" or decision.get("case_note", "").startswith("No known fraud tactic matched")):
            required.append("fraud call with no matched tactic")
    if any(not ok(c["result"]) for c in calls):
        required.append("a lookup returned an error")
    if design["own_lookup_required"] and not recall.get(OWN_LOOKUP, True):
        required.append("required lookup missing")
    return {gate: escalated for gate in required}


# ---------------------------------------------------------------------------
# Write-up: what the judge sees, and a rule-based grounding cross-check
# ---------------------------------------------------------------------------

def model_writeup(decision):
    evidence = []
    for item in decision.get("evidence", []):
        for pattern in CODE_NOTES:
            item = pattern.sub("", item)
        if item.strip():
            evidence.append(item.strip())
    note = decision.get("case_note", "")
    for pattern in CODE_NOTES:
        note = pattern.sub("", note)
    return {"decision": decision.get("decision"), "confidence": decision.get("confidence"),
            "evidence": evidence, "case_note": note.strip()}


def seen_evidence(record, calls, case, include_score):
    """Tool results as the model received them (tactic runs saw the latest 20 history rows)."""
    seen = {}
    for call in calls:
        result = call["result"]
        if (call["name"] == "get_transaction_history" and ok(result)
                and record["reviewer"].startswith("tactic")):
            rows = result.get("transactions", [])
            result = {"transactions": rows[-20:], "note": f"latest 20 of {len(rows)} rows"}
        seen[call["name"]] = result
    seen["flagged_transaction"] = case.to_dict(include_score=include_score)
    return seen


def checklist_lines(record):
    matches = tactic_matches(record)
    lines = []
    for m in matches:
        if not m["matched"]:
            continue
        lines.append(f"Tactic: {m['title']} - {m['strength']} match")
        for group in ("core", "supporting", "against"):
            for item in m[group]:
                mark = {True: "[yes]", False: "[no]"}.get(item["present"], "[unknown]")
                lines.append(f"  {group}: {mark} {item['detail']}")
    if matches and not lines:
        lines.append("No known tactic matched (no tactic had all of its core signs present).")
    return lines


def red_flags(case, facts, record):
    """Triage red flags (and innocent explanations) the write-up should give a verdict on."""
    flags = []
    summary = facts.get("get_spending_summary") or {}
    median = (summary.get("amount_overall") or {}).get("median")
    top = (summary.get("amount_overall") or {}).get("max")
    if median and case.amount / median >= 5:
        flags.append(f"amount {case.amount:.2f} is {case.amount / median:.1f}x the customer's median {median:.2f}")
    if top is not None and case.amount > top:
        flags.append(f"largest purchase so far (earlier maximum {top:.2f})")
    if case.time.hour in LATE_HOURS:
        flags.append(f"late-night purchase at {case.time:%H:%M}")
    if case.category.endswith("_net"):
        flags.append(f"online category {case.category}")
    merchant = facts.get("get_merchant_history") or {}
    if merchant.get("first_time"):
        flags.append(f"first purchase at {case.merchant}")
    recent = summary.get("recent_activity") or {}
    if recent.get("transactions_last_1h", 0) >= 3:
        flags.append(f"{recent['transactions_last_1h']} transactions in the previous hour")
    if recent.get("transactions_last_24h", 0) >= 8:
        flags.append(f"{recent['transactions_last_24h']} transactions in the previous 24 hours")
    hours = recent.get("hours_since_previous_transaction")
    if hours is not None and hours < 0.25:
        flags.append(f"only {hours:.2f} hours since the previous transaction")
    matched = [m for m in tactic_matches(record) if m["matched"]]
    if matched:
        flags.append(f"matched tactic: {matched[0]['title']} ({matched[0]['strength']})")
        for item in matched[0]["against"]:
            if item["present"] is True:
                flags.append(f"innocent explanation present: {item['detail']}")
    return flags


def available_numbers(seen, checklist, findings, case):
    values = []

    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                collect(str(key))  # field names carry windows and thresholds: last_24h, p95
                collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
        elif isinstance(value, str):
            values.extend(float(n) for n in LOOSE_NUMBER.findall(value.replace(",", "")))

    collect(seen)
    collect(checklist)
    collect(findings)
    values += derived_numeric_values(seen)
    values += [case.time.hour, case.time.minute, case.time.year, case.time.month, case.time.day]
    return values


def rule_grounding(writeup, seen, checklist, findings, case, facts):
    """Rule-based grounding problems: numbers absent from the evidence, and two common contradictions."""
    available = available_numbers(seen, checklist, findings, case)
    problems = []
    text = " ".join(writeup["evidence"] + [writeup["case_note"]])
    for number in NUMBER_PATTERN.findall(text.replace(",", "")):
        value = float(number)
        if not any(abs(value - c) <= max(0.05, abs(c) * 0.01) for c in available):
            problems.append(f"number {number} not in any tool result")
    merchant = facts.get("get_merchant_history") or {}
    if merchant.get("first_time") is False and any(
            not OTHER_TRANSACTION.search(text[max(0, m.start() - 40):m.start()])
            for m in NEW_MERCHANT_CLAIM.finditer(text)):
        problems.append(f"calls the merchant new, but there were {merchant.get('earlier_transactions')} earlier visits")
    if case.time.hour not in LATE_HOURS and LATE_CLAIM.search(text):
        problems.append(f"calls it late night, but it was at {case.time:%H:%M}")
    return sorted(set(problems))


JUDGE_INSTRUCTIONS = """You grade a fraud analyst's write-up for one flagged card transaction. You do not decide whether it is fraud. You check the write-up against the evidence the analyst had.

Score each criterion and quote the write-up for each score.

GROUNDING (blocking)
 2 = every factual claim traces to the evidence below (tool results, the flagged transaction, the checklist, or the deterministic findings)
 1 = one claim is unsourced but plausible
 0 = any claim contradicts the evidence, states a sign marked [no] as true, uses numbers from a tactic's illustrative example as this customer's facts, or describes a lookup that is not in the evidence

HEDGING
 2 = unconfirmed items are explicitly marked unconfirmed (failed lookups, [unknown] signs, mixed evidence), or there is nothing unconfirmed and every claim is verified
 1 = hedged vaguely ("possibly", "may be") without saying what is missing
 0 = asserts an unverified claim as fact

COMPLETENESS
 2 = every red flag listed below gets a verdict in the write-up (supports fraud, or explained away), and innocent explanations or negative results are reported too
 0 = at least one red flag below is not addressed in the write-up

Ignore the decision itself and the XGBoost score unless the write-up states something about them that contradicts the evidence. Matching a number within rounding counts as grounded. Ratios calculated from two values in the evidence count as grounded.

Reply with JSON only:
{"grounding": {"score": 0-2, "quote": "...", "reason": "..."},
 "hedging": {"score": 0-2, "quote": "...", "reason": "..."},
 "completeness": {"score": 0 or 2, "unaddressed_red_flags": ["..."], "reason": "..."}}
Keep each quote under 25 words and each reason under 30 words."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounding": {"type": "object", "properties": {
            "score": {"type": "integer", "enum": [0, 1, 2]}, "quote": {"type": "string"},
            "reason": {"type": "string"}}, "required": ["score", "quote", "reason"]},
        "hedging": {"type": "object", "properties": {
            "score": {"type": "integer", "enum": [0, 1, 2]}, "quote": {"type": "string"},
            "reason": {"type": "string"}}, "required": ["score", "quote", "reason"]},
        "completeness": {"type": "object", "properties": {
            "score": {"type": "integer", "enum": [0, 2]},
            "unaddressed_red_flags": {"type": "array", "items": {"type": "string"}},
            "reason": {"type": "string"}}, "required": ["score", "unaddressed_red_flags", "reason"]},
    },
    "required": ["grounding", "hedging", "completeness"],
}


def judge_message(writeup, seen, checklist, findings, flags, best_tactic_text, score_shown):
    parts = ["EVIDENCE THE ANALYST HAD", json.dumps(seen, default=str)]
    if not score_shown:
        parts.append("(The XGBoost score the analyst saw is not available here; do not grade claims about it.)")
    parts += ["Deterministic findings:", *findings]
    if checklist:
        parts += ["Tactic checklist:", *checklist]
    if best_tactic_text:
        parts += ["Best-matching tactic document (its example is illustrative, not this customer):",
                  best_tactic_text]
    parts += ["", "RED FLAGS FROM TRIAGE (each needs a verdict):", *(f"- {f}" for f in flags or ["(none)"])]
    parts += ["", "WRITE-UP TO GRADE", json.dumps(writeup)]
    return "\n".join(parts)


def call_judge(model, message, url="http://localhost:11434/api/chat"):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": JUDGE_INSTRUCTIONS}, {"role": "user", "content": message}],
        "stream": False, "think": False, "format": JUDGE_SCHEMA,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    reply = json.loads(urllib.request.urlopen(request, timeout=900).read())
    return json.loads(reply["message"]["content"])


def load_judge_cache():
    cache = {}
    if JUDGE_CACHE.exists():
        for line in JUDGE_CACHE.open(encoding="utf-8"):
            item = json.loads(line)
            cache[(item["run"], item["case_id"], item["judge_model"])] = item["grade"]
    return cache


# ---------------------------------------------------------------------------
# Fresh run of the current reviewer, through the existing pipeline
# ---------------------------------------------------------------------------

class TranscriptRecorder:
    """Passes every attribute through to the reviewer; also saves each case's transcript,
    because the run log keeps only the cleaned tool arguments, not what the model sent."""

    def __init__(self, reviewer, path):
        self._reviewer, self._path = reviewer, path

    def __getattr__(self, name):
        return getattr(self._reviewer, name)

    def review(self, case, tools):
        try:
            return self._reviewer.review(case, tools)
        finally:
            with self._path.open("a", encoding="utf-8") as out:
                out.write(json.dumps({"case_id": case.case_id, "transcript": self._reviewer.last_transcript},
                                     default=str) + "\n")


def run_fresh(n, seed, split, store, provider, model):
    from agent.llm import OpenAICompatibleBackend
    from agent.pipeline import run
    from agent.rag_corpus import load_corpus
    from agent.retriever import EmbeddingRetriever
    from agent.reviewers import CombinedReviewer

    reviewer = CombinedReviewer(OpenAICompatibleBackend(provider, model),
                                retriever=EmbeddingRetriever(load_corpus(None)), allowed_tools=ALL_TOOLS)
    name = f"{FRESH_NAME}-s{seed}"
    transcripts = RUNS_DIR / f"{name}_{split}_transcripts.jsonl"
    path = run(split, TranscriptRecorder(reviewer, transcripts), sample=n, seed=seed, run_name=name, store=store)
    for source in (path, transcripts):
        shutil.copy(source, SAVED_RUNS_DIR / source.name)
    return SAVED_RUNS_DIR / path.name


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_run(path, store, cases, judge_n, judge_model, cache, judge_seed=0):
    name = run_name(path)
    design = design_for(name)
    records, transcripts = load_run(path)
    fresh = bool(transcripts)
    judged_ids = set()
    if judge_n:
        decided = [r["case_id"] for r in records if r.get("decision")]
        judged_ids = set(random.Random(judge_seed).sample(decided, min(judge_n, len(decided))))

    rows = []
    for record in records:
        record["_run"] = name
        case_id = record["case_id"]
        row = cases.loc[case_id]
        case = make_case(store, row, case_id)
        transcript = transcripts.get(case_id)
        calls = trajectory(record, transcript, store, row)
        recall = tool_recall(calls, design)
        violations = order_violations(record, calls, transcript, design)
        gate = gates(record, calls, recall, design)
        model_calls = [c for c in calls if c["origin"] == "model"]
        out = {
            "run": name, "case_id": case_id,
            "decision": (record.get("decision") or {}).get("decision", "ERROR"),
            "confidence": (record.get("decision") or {}).get("confidence"),
            "tools_called": " > ".join(c["name"].removeprefix("get_") for c in calls),
            "checks_required": len(recall), "checks_done": sum(recall.values()),
            "checks_missed": "; ".join(k for k, v in recall.items() if not v),
            "calls": len(calls), "calls_args_ok": sum(argument_ok(c) for c in calls),
            "model_calls": len(model_calls), "model_calls_args_ok": sum(argument_ok(c) for c in model_calls),
            "order_violations": "; ".join(violations),
            "gates_required": "; ".join(gate), "gates_held": sum(gate.values()), "gates_total": len(gate),
            "gates_breached": "; ".join(g for g, held in gate.items() if not held),
        }
        decision = record.get("decision")
        if decision:
            facts = full_facts(store, row)
            seen = seen_evidence(record, calls, case, include_score=fresh)
            findings = rule_checks(case, seen)
            checklist = checklist_lines(record)
            writeup = model_writeup(decision)
            problems = rule_grounding(writeup, seen, checklist, findings, case, facts)
            out["rule_grounding_problems"] = "; ".join(problems)
            out["rule_grounding_ok"] = not problems
            if case_id in judged_ids:
                key = (name, case_id, judge_model)
                if key not in cache:
                    matched = [m for m in tactic_matches(record) if m["matched"]]
                    tactic_text = TACTICS_BY_ID[matched[0]["document_id"]].text if matched else ""
                    message = judge_message(writeup, seen, checklist, findings, red_flags(case, facts, record),
                                            tactic_text, score_shown=fresh)
                    started = time.perf_counter()
                    try:
                        grade = call_judge(judge_model, message)
                    except Exception as exc:  # a failed grade is reported, not fatal
                        grade = {"error": repr(exc)}
                    print(f"  judged {name} {case_id[:8]} in {time.perf_counter() - started:.0f}s: "
                          + (json.dumps({k: v.get('score') for k, v in grade.items() if isinstance(v, dict)})
                             if "error" not in grade else grade["error"]))
                    cache[key] = grade
                    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                    with JUDGE_CACHE.open("a", encoding="utf-8") as f:
                        f.write(json.dumps({"run": name, "case_id": case_id, "judge_model": judge_model,
                                            "grade": grade}) + "\n")
                grade = cache[key]
                if "error" not in grade:
                    for criterion in ("grounding", "hedging", "completeness"):
                        out[criterion] = grade[criterion]["score"]
                        out[f"{criterion}_quote"] = grade[criterion].get("quote", "")
                        out[f"{criterion}_reason"] = grade[criterion]["reason"]
                    out["unaddressed_red_flags"] = "; ".join(grade["completeness"].get("unaddressed_red_flags", []))
                    out["rubric_pass"] = grade["grounding"]["score"] > 0
        rows.append(out)
    return pd.DataFrame(rows)


def summarise(cases):
    rows = []
    for name, df in cases.groupby("run", sort=False):
        judged = df.dropna(subset=["grounding"]) if "grounding" in df else df.iloc[0:0]
        decided = df[df["decision"] != "ERROR"]
        model_calls = df["model_calls"].sum()
        rows.append({
            "run": name,
            "cases": len(df),
            "tool recall": df["checks_done"].sum() / df["checks_required"].sum(),
            "cases with every check": f"{(df['checks_done'] == df['checks_required']).sum()}/{len(df)}",
            "argument accuracy (all calls)": df["calls_args_ok"].sum() / df["calls"].sum(),
            "argument accuracy (model's own calls)": (df["model_calls_args_ok"].sum() / model_calls
                                                      if model_calls else None),
            "model's own calls": int(model_calls),
            "order violations (cases)": int((df["order_violations"] != "").sum()),
            "gate adherence": (f"{int(df['gates_held'].sum())}/{int(df['gates_total'].sum())}"
                               if df["gates_total"].sum() else "no case needed one"),
            "rule grounding ok": f"{int(decided['rule_grounding_ok'].sum())}/{len(decided)}",
            "judged": len(judged),
            "GROUNDING mean (0-2)": judged["grounding"].mean() if len(judged) else None,
            "GROUNDING = 0": int((judged["grounding"] == 0).sum()) if len(judged) else None,
            "HEDGING mean (0-2)": judged["hedging"].mean() if len(judged) else None,
            "COMPLETENESS = 2": f"{int((judged['completeness'] == 2).sum())}/{len(judged)}" if len(judged) else None,
            "rubric pass (GROUNDING > 0)": f"{int(judged['rubric_pass'].sum())}/{len(judged)}" if len(judged) else None,
        })
    return pd.DataFrame(rows).set_index("run")


def gate_breakdown(cases):
    rows = []
    for _, r in cases.iterrows():
        for gate in filter(None, r["gates_required"].split("; ")):
            rows.append({"run": r["run"], "gate": gate, "held": gate not in r["gates_breached"].split("; ")})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.groupby(["run", "gate"], sort=False)["held"].agg(required="size", held="sum")


def judge_agreement(cases):
    """How often the judge's GROUNDING agrees with the rule-based check (judge 2 = rules found nothing)."""
    if "grounding" not in cases:
        return None
    judged = cases.dropna(subset=["grounding"])
    if judged.empty:
        return None
    agree = (judged["grounding"] == 2) == judged["rule_grounding_ok"].astype(bool)
    return pd.crosstab(judged["rule_grounding_ok"].map({True: "rules: no problem", False: "rules: problem"}),
                       judged["grounding"].astype(int).rename("judge GROUNDING")), agree.mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run .jsonl files (default: every run in results/agent_eval/runs)")
    parser.add_argument("--split", default="val", choices=["val"])
    parser.add_argument("--fresh", type=int, help="first review this many new flagged cases with the current reviewer")
    parser.add_argument("--seed", type=int, default=2033, help="sample seed for --fresh")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b-8k", help="reviewer model for --fresh")
    parser.add_argument("--judge", type=int, default=0, help="grade this many write-ups per judged run")
    parser.add_argument("--judge-runs", nargs="*",
                        help=f"runs to judge (default: eval100-qwen8k-tactics-v3 and any {FRESH_NAME} run)")
    parser.add_argument("--judge-model", default="qwen3.5:4b", help="Ollama model used as the judge")
    args = parser.parse_args()

    store = TransactionStore.load()
    if args.fresh:
        fresh_path = run_fresh(args.fresh, args.seed, args.split, store, args.provider, args.model)
        print(f"Fresh run saved to {fresh_path}")

    paths = [Path(p) for p in args.runs] or sorted(
        p for p in SAVED_RUNS_DIR.glob(f"*_{args.split}.jsonl") if not p.stem.endswith("_transcripts"))
    judge_runs = set(args.judge_runs or ["eval100-qwen8k-tactics-v3"] +
                     [run_name(p) for p in paths if run_name(p).startswith(FRESH_NAME)])
    all_ids = set()
    for path in paths:
        all_ids |= {r["case_id"] for r in load_run(path)[0]}
    cases = load_cases(all_ids, args.split)
    cache = load_judge_cache()

    frames = []
    for path in paths:
        name = run_name(path)
        n = args.judge if name in judge_runs else 0
        print(f"Scoring {name}" + (f" (judging {n} write-ups with {args.judge_model})" if n else ""))
        frames.append(score_run(path, store, cases, n, args.judge_model, cache))
    per_case = pd.concat(frames, ignore_index=True)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    per_case.to_csv(REPORTS_DIR / "agent_eval_cases.csv", index=False)
    summary = summarise(per_case)
    summary.to_csv(REPORTS_DIR / "agent_eval_summary.csv")

    pd.set_option("display.width", 250, "display.max_columns", 30)
    print("\nPath metrics and write-up rubric\n")
    print(summary.round(3).T.to_string())
    breakdown = gate_breakdown(per_case)
    if not breakdown.empty:
        print("\nGates: cases that had to go to a person, and how many did\n")
        print(breakdown.to_string())
    agreement = judge_agreement(per_case)
    if agreement:
        table, rate = agreement
        print(f"\nJudge vs rule-based grounding check (agreement {rate:.0%})\n")
        print(table.to_string())
    print(f"\nPer-case results: {REPORTS_DIR / 'agent_eval_cases.csv'}")


if __name__ == "__main__":
    main()
