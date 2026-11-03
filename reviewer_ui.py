"""Dark chat page for the local fraud reviewer."""

import time

import pandas as pd
import streamlit as st

from agent.case_query import route_message
from agent.config import PROCESSED_DIR
from agent.data import TransactionStore
from agent.llm import LLMReviewer, OpenAICompatibleBackend
from agent.schemas import Case
from agent.tools import CaseTools

MODEL = "qwen2.5:3b"
PROVIDER = "ollama"
SAMPLE_COUNT = 6

TOOL_LABELS = {
    "get_kyc_profile": "KYC profile",
    "get_transaction_history": "Transaction history",
    "get_spending_summary": "Spending summary",
}

GENERAL_PROMPT = (
    "You are the analyst assistant for a card-fraud reviewer. No transaction is open, "
    "so you cannot look up a customer. Answer in two or three sentences. "
    "If they want a case reviewed, tell them to name a customer id and an amount, "
    "or a merchant and an amount, from a transaction the model already flagged."
)

WELCOME = (
    "Describe a flagged transaction to open it, or ask how this reviewer works. "
    "A customer id and amount is enough, and so is a merchant plus the amount. "
    "Once a case is open, questions stay on it until you name a different one."
)


def apply_theme():
    st.markdown(
        """
        <style>
        .stApp { background: #101014; overflow-x: hidden; }
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] { overflow-x: hidden !important; }
        header[data-testid="stHeader"] { display: none; }
        [data-testid="stToolbar"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"],
        [data-testid="stHeaderActionElements"], [data-testid="stDecoration"] { display: none; }
        footer { visibility: hidden; }
        [data-testid="stMainBlockContainer"],
        [data-testid="stMain"] .block-container,
        .main .block-container,
        [data-testid="stBottom"] > div,
        [data-testid="stBottomBlockContainer"] {
          width: min(740px, 100%) !important;
          max-width: min(740px, 100%) !important;
          margin-left: auto !important;
          margin-right: auto !important;
          padding-left: 1rem !important;
          padding-right: 1rem !important;
          box-sizing: border-box !important;
        }
        [data-testid="stMainBlockContainer"],
        [data-testid="stMain"] .block-container,
        .main .block-container { padding-top: 0.85rem !important; }
        h1 {
          font-weight: 600;
          letter-spacing: -0.03em;
          font-size: 1.45rem;
          line-height: 1.2;
          margin: 0.2rem 0 0.15rem;
        }
        [data-testid="stCaptionContainer"] { font-size: 0.8rem; line-height: 1.35; }
        [data-testid="stChatMessage"] { background: transparent; }
        [data-testid="stChatInput"] textarea {
          background: #18181d;
          border: 1px solid #2a2a32;
          border-radius: 14px;
        }
        [data-testid="stButtonGroup"] [role="radiogroup"] {
          display: flex !important;
          flex-wrap: wrap !important;
          justify-content: flex-start !important;
          align-items: flex-start !important;
          gap: 0.45rem !important;
        }
        [data-testid="stButtonGroup"] [role="radio"] {
          width: auto !important;
          flex: 0 0 auto !important;
          border-radius: 999px;
          background: #18181d;
          border: 1px solid #2e2e38;
        }
        [data-testid="stLayoutWrapper"]:has([class*="st-key-assistant-"]),
        [data-testid="stLayoutWrapper"]:has([class*="st-key-user-"]) {
          width: 100% !important;
          max-width: 100% !important;
          min-width: 0 !important;
          margin-left: 0 !important;
          margin-right: 0 !important;
          align-self: stretch !important;
          overflow: visible !important;
        }
        [class*="st-key-assistant-"],
        [class*="st-key-user-"] {
          min-width: 0 !important;
          margin-bottom: 0.55rem !important;
          padding: 1.5rem 1.6rem !important;
          box-sizing: border-box !important;
          overflow: visible !important;
          overflow-wrap: break-word;
        }
        [class*="st-key-assistant-"] {
          width: 100% !important;
          max-width: 100% !important;
          margin-right: auto !important;
          margin-left: 0 !important;
          background: #1c1c24;
          border: 1px solid #2e2e38;
          border-radius: 16px 16px 16px 5px;
        }
        [class*="st-key-user-"] {
          width: fit-content !important;
          max-width: min(24rem, 80%) !important;
          margin-left: auto !important;
          margin-right: 0 !important;
          background: #243044;
          border: 1px solid #3d5278;
          border-radius: 16px 16px 5px 16px;
        }
        [class*="st-key-assistant-"] [data-testid="stMarkdown"],
        [class*="st-key-assistant-"] [data-testid="stMarkdownContainer"],
        [class*="st-key-user-"] [data-testid="stMarkdown"],
        [class*="st-key-user-"] [data-testid="stMarkdownContainer"],
        [class*="st-key-assistant-"] [data-testid="stElementContainer"],
        [class*="st-key-user-"] [data-testid="stElementContainer"] {
          width: auto !important;
          max-width: 100% !important;
          min-width: 0 !important;
          margin: 0 !important;
          overflow: visible !important;
          box-sizing: border-box !important;
        }
        [data-testid="stChatInput"] textarea {
          padding: 0.85rem 1.15rem !important;
        }
        [class*="st-key-assistant-"] ul {
          margin: 0.4rem 0 0.15rem;
          padding-left: 1.15rem;
        }
        [class*="st-key-assistant-"] li { margin: 0.35rem 0; }
        [class*="st-key-assistant-"] [data-testid="stMarkdown"] p,
        [class*="st-key-user-"] [data-testid="stMarkdown"] p {
          margin: 0;
          font-size: 0.95rem;
          line-height: 1.45;
        }
        [class*="st-key-assistant-"] [data-testid="stCaptionContainer"],
        [class*="st-key-user-"] [data-testid="stCaptionContainer"] {
          margin-top: 0.35rem;
        }
        [class*="st-key-assistant-"] .stButton > button {
          width: 100%;
          text-align: left;
          justify-content: flex-start;
          white-space: normal;
          height: auto;
          min-height: 2.6rem;
          background: #16161c;
          color: #ececf1;
          border: 1px solid #3a3a48;
          border-radius: 12px;
          padding: 0.65rem 0.8rem;
          line-height: 1.35;
          transition: border-color 0.15s ease, background 0.15s ease, transform 0.15s ease;
          cursor: pointer;
        }
        [class*="st-key-assistant-"] .stButton > button:hover {
          border-color: #8eb4ff;
          background: #2a3348;
          transform: translateY(-1px);
        }
        [class*="st-key-assistant-"] .stButton > button p {
          text-align: left;
          white-space: normal;
        }
        [data-testid="stExpander"] {
          background: transparent;
          border: none;
        }
        [data-testid="stExpander"] details {
          background: transparent;
          border: none;
        }
        [data-testid="stExpander"] summary {
          color: #9eb6e0;
          cursor: pointer;
        }
        [data-testid="stExpander"] summary:hover {
          color: #d6e4ff;
        }
        .verdict {
          display: inline-block;
          padding: 0.15rem 0.55rem;
          border-radius: 999px;
          font-size: 0.82rem;
          letter-spacing: 0.01em;
          margin-bottom: 0.35rem;
        }
        .verdict.fraud { color: #ffb088; background: rgba(235, 104, 52, 0.16); }
        .verdict.clear { color: #9ee0c3; background: rgba(61, 190, 138, 0.14); }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Loading transactions and KYC…")
def get_store():
    return TransactionStore.load()


@st.cache_resource(show_spinner=False)
def get_flagged():
    """Every validation transaction the scoring model flagged, with its customer id."""
    store = get_store()
    flagged = pd.read_csv(PROCESSED_DIR / "flagged_val.csv", parse_dates=["trans_date_trans_time"])
    customer_ids = {cc_num: store.customer_id(cc_num) for cc_num in flagged["cc_num"].unique()}
    flagged["customer_id"] = flagged["cc_num"].map(customer_ids)
    return flagged


@st.cache_data(show_spinner=False)
def get_samples():
    flagged = pd.read_csv(PROCESSED_DIR / "flagged_val.csv", parse_dates=["trans_date_trans_time"])
    ranked = flagged.sort_values(["xgb_score", "trans_num"], kind="mergesort").reset_index(drop=True)
    positions = [int(round(i * (len(ranked) - 1) / (SAMPLE_COUNT - 1))) for i in range(SAMPLE_COUNT)]
    return ranked.iloc[positions].reset_index(drop=True)


def _reviewer():
    return LLMReviewer(OpenAICompatibleBackend(PROVIDER, MODEL))


def investigate(row, store):
    case = Case(
        case_id=row.trans_num,
        customer_id=store.customer_id(row.cc_num),
        time=row.trans_date_trans_time,
        amount=float(row.amt),
        merchant=row.merchant,
        category=row.category,
        xgb_score=float(row.xgb_score),
    )
    tools = CaseTools(store, row.cc_num, row.trans_date_trans_time)
    reviewer = _reviewer()
    started = time.perf_counter()
    try:
        decision = reviewer.review(case, tools)
        error = None
    except Exception as exc:
        decision, error = None, str(exc)
    return {
        "case": case.to_dict(),
        "events": reviewer.last_transcript,
        "decision": None if decision is None else decision.to_dict(),
        "error": error,
        "usage": reviewer.last_usage,
        "seconds": round(time.perf_counter() - started, 1),
        "messages": reviewer.last_messages,
        "cc_num": row.cc_num,
        "time": row.trans_date_trans_time,
    }


def answer_general(question):
    backend = OpenAICompatibleBackend(PROVIDER, MODEL)
    started = time.perf_counter()
    response = backend.chat(
        [backend.system(GENERAL_PROMPT), backend.user(question)],
        [],
    )
    return {
        "events": [{"role": "assistant", "text": response.text or "Choose a sample case to open a transaction."}],
        "usage": {"turns": 1, "input_tokens": response.input_tokens, "output_tokens": response.output_tokens},
        "seconds": round(time.perf_counter() - started, 1),
        "error": None,
    }


def ask_about_case(record, store, question):
    tools = CaseTools(store, record["cc_num"], record["time"])
    reviewer = _reviewer()
    started = time.perf_counter()
    try:
        reviewer.ask(record["messages"], tools, question)
        error = None
    except Exception as exc:
        error = str(exc)
    record["messages"] = reviewer.last_messages
    return {
        "events": reviewer.last_transcript,
        "error": error,
        "usage": reviewer.last_usage,
        "seconds": round(time.perf_counter() - started, 1),
    }


def chip_label(row):
    category = str(row.category).replace("_", " ")
    return f"${row.amt:,.0f} · {category} · {row.xgb_score:.2f}"


def choice_option(row_index, flagged):
    row = flagged.iloc[int(row_index)]
    when = pd.Timestamp(row.trans_date_trans_time).strftime("%d %b %Y, %H:%M")
    category = str(row.category).replace("_", " ")
    return {
        "row": int(row_index),
        "label": f"${row.amt:,.2f}  ·  {category}  ·  {row.merchant}  ·  {when}",
    }


def _bubble(role, turn_index):
    """Assistant fills the column. A user message stays as wide as its text."""
    width = "content" if role == "user" else "stretch"
    return st.container(key=f"{role}-{turn_index}", width=width)


def render_tool(event):
    name = event.get("name") or "tool"
    result = event.get("result") or {}
    with st.expander(TOOL_LABELS.get(name, name), expanded=False):
        dropped = event.get("dropped") or []
        if dropped:
            st.caption(f"Ignored {', '.join(dropped)}. This lookup stayed on the flagged case.")
        if name == "get_transaction_history" and result.get("transactions"):
            frame = pd.DataFrame(result["transactions"])
            st.caption(f"{len(frame)} earlier transactions")
            st.dataframe(frame, hide_index=True, width="stretch")
        else:
            st.json(result)


def render_decision(decision):
    kind = "fraud" if decision["decision"] == "fraud" else "clear"
    title = "Raise a fraud alert" if kind == "fraud" else "Clear the flag"
    st.markdown(f"<span class='verdict {kind}'>{title}</span>", unsafe_allow_html=True)
    st.markdown(f"{decision['confidence']:.0%} confidence it is fraud")
    for item in decision.get("evidence") or []:
        st.markdown(f"- {item}")
    if decision.get("case_note"):
        st.caption(decision["case_note"])


def _visible_answer(events):
    answers = [event for event in events if event["role"] == "assistant" and not event.get("tool_calls")]
    if not answers:
        return None
    return answers[-1]


def render_steps(events):
    """The lookups and notes that led to the reply, in order."""
    step = 1
    index = 0
    while index < len(events):
        event = events[index]
        if event["role"] == "user":
            index += 1
            continue
        if event["role"] == "note":
            st.markdown(f"**{step}.** {event['text']}")
            step += 1
            index += 1
            continue
        if event["role"] == "assistant" and event.get("tool_calls"):
            names = [TOOL_LABELS.get(call["name"], call["name"]) for call in event["tool_calls"]]
            st.markdown(f"**{step}.** Looked up {', '.join(names)}.")
            thought = (event.get("text") or "").strip()
            if thought:
                st.markdown(thought)
            step += 1
            index += 1
            while index < len(events) and events[index]["role"] == "tool":
                render_tool(events[index])
                index += 1
            continue
        if event["role"] == "assistant" and event.get("decision"):
            title = "raise a fraud alert" if event["decision"]["decision"] == "fraud" else "clear the flag"
            st.markdown(f"**{step}.** Decided to {title}.")
            step += 1
            index += 1
            continue
        if event["role"] == "assistant" and (event.get("text") or "").strip():
            st.markdown(f"**{step}.** Wrote a reply.")
            st.markdown(event["text"])
            step += 1
        index += 1


def render_turn(item, turn_index):
    if item["role"] == "user" and item.get("kind") == "case":
        case = item["case"]
        when = pd.Timestamp(case["time"]).strftime("%d %b %Y, %H:%M")
        category = case["category"].replace("_", " ")
        with _bubble("user", turn_index):
            st.markdown(f"**${case['amount']:,.2f}** · {category} · {case['merchant']}")
            st.caption(f"{when} · {case['customer_id']} · score {case['xgb_score']:.2f}")
        return
    if item["role"] == "user":
        with _bubble("user", turn_index):
            st.markdown(item["text"])
        return

    events = item.get("events") or []
    opened = next((event for event in events if event.get("role") == "note" and str(event.get("text", "")).startswith("Opened ")), None)
    back = next((event for event in events if event.get("role") == "note" and str(event.get("text", "")).startswith("Back on ")), None)
    trace = [event for event in events if event is not opened and event is not back]
    has_steps = any(event.get("role") == "tool" or event.get("tool_calls") or event.get("role") == "note" for event in trace)
    answer = _visible_answer(events)
    tool_events = [event for event in events if event["role"] == "tool"]
    rejected = [event for event in tool_events if isinstance(event.get("result"), dict) and "error" in event["result"]]

    with _bubble("assistant", turn_index):
        if tool_events and len(rejected) == len(tool_events):
            st.warning("Every tool call was refused, so this reply is not grounded in a lookup.")
        if answer and answer.get("decision"):
            render_decision(answer["decision"])
        elif answer and answer.get("text"):
            shown = answer["text"]
            if item.get("options"):
                shown = shown.split("\n")[0]
            st.markdown(shown)
        elif back:
            st.markdown(back["text"])
        for option in item.get("options") or []:
            if st.button(option["label"], key=f"choice-{turn_index}-{option['row']}", width="stretch"):
                st.session_state.pick_row = option["row"]
                st.session_state.pick_label = option["label"]
                st.rerun()
        if item.get("options"):
            st.caption("Click a transaction, or reply with a number.")
        if opened or has_steps:
            label = opened["text"].rstrip(".") if opened else "How this reply was reached"
            with st.expander(label):
                render_steps(trace)
        if item.get("error"):
            st.markdown(f"That turn stopped: {item['error']}")
        usage = item.get("usage") or {}
        if item.get("seconds") is not None:
            st.caption(f"{item['seconds']}s · {usage.get('turns', 0)} turns · {MODEL}")


def _welcome_turn():
    return {"role": "assistant", "events": [{"role": "assistant", "text": WELCOME}]}


def _ensure_state():
    if "chat" not in st.session_state:
        st.session_state.chat = [_welcome_turn()]
        st.session_state.cases = {}
        st.session_state.active_id = None
        st.session_state.pending_choices = None
    opening = st.session_state.chat[0]["events"][0].get("text", "")
    if opening.startswith("Choose a sample case"):
        st.session_state.chat[0] = _welcome_turn()


def chat_page():
    if st.session_state.pop("do_clear", False):
        for key in ("sample_pick", "chat", "handled_pick", "pick_row", "pick_label"):
            st.session_state.pop(key, None)
        st.session_state.cases = {}
        st.session_state.active_id = None
        st.session_state.pending_choices = None
    _ensure_state()
    if "pending_choices" not in st.session_state:
        st.session_state.pending_choices = None

    st.title("Fraud reviewer")
    active = st.session_state.cases.get(st.session_state.active_id)
    if active:
        case = active["case"]
        st.caption(f"Open case · {case['customer_id']} · ${case['amount']:,.2f} at {case['merchant']}")
    else:
        st.caption(f"{MODEL} on Ollama · one flagged transaction at a time · the fraud label stays hidden")

    samples = get_samples()
    store = get_store()
    labels = {row.trans_num: chip_label(row) for _, row in samples.iterrows()}
    if st.session_state.get("pick_row") is not None:
        _consume_pick(store)
    thread = st.container()
    with thread:
        for turn_index, item in enumerate(st.session_state.chat):
            render_turn(item, turn_index)

    picked = st.pills(
        "Sample cases",
        list(labels),
        format_func=lambda case_id: labels[case_id],
        selection_mode="single",
        key="sample_pick",
        width="stretch",
        wrap=True,
    )
    clear_col, _ = st.columns([1, 4])
    if clear_col.button("Clear chat", type="tertiary"):
        st.session_state.do_clear = True
        st.rerun()

    if picked and picked != st.session_state.get("handled_pick"):
        st.session_state.handled_pick = picked
        st.session_state.pending_choices = None
        row = samples.loc[samples["trans_num"] == picked].iloc[0]
        _open_case(row, store, thread, announce=False)

    if st.session_state.pending_choices:
        placeholder = "Click a transaction, or reply with a number"
    elif st.session_state.active_id:
        placeholder = "Ask about this case, or name another"
    else:
        placeholder = "Describe a transaction, or ask a question"
    question = st.chat_input(placeholder)
    if not question:
        return

    st.session_state.chat.append({"role": "user", "text": question})
    flagged = get_flagged()
    routed = route_message(
        question,
        flagged,
        active_id=st.session_state.active_id,
        pending=st.session_state.pending_choices,
    )
    if routed["action"] == "investigate":
        st.session_state.pending_choices = None
        _open_case(flagged.iloc[routed["row"]], store, thread, announce=True)
        return
    if routed["action"] == "clarify":
        choices = routed.get("choices") or []
        st.session_state.pending_choices = choices or None
        reply = {
            "events": [{"role": "assistant", "text": routed["text"]}],
            "options": [choice_option(index, flagged) for index in choices],
            "usage": {},
            "seconds": None,
            "error": None,
        }
    else:
        st.session_state.pending_choices = None
        with st.spinner("Thinking…"):
            reply = _answer(question, store)
    st.session_state.chat.append({"role": "assistant", **reply})
    with thread:
        render_turn(st.session_state.chat[-2], len(st.session_state.chat) - 2)
        render_turn(st.session_state.chat[-1], len(st.session_state.chat) - 1)


def _open_case(row, store, thread, announce):
    """Run the reviewer on one flagged row and show it in the thread."""
    case_id = row.trans_num
    cached = case_id in st.session_state.cases
    rerun = cached and st.session_state.active_id == case_id
    if not cached or rerun:
        with st.spinner(f"Reviewing ${row.amt:,.2f} at {row.merchant}…"):
            record = investigate(row, store)
        st.session_state.cases[case_id] = record
        if announce:
            events = [{
                "role": "note",
                "text": f"Opened {record['case']['customer_id']} · ${record['case']['amount']:,.2f} at {record['case']['merchant']}.",
            }]
            events.extend(record["events"])
        else:
            st.session_state.chat.append({"role": "user", "kind": "case", "case": record["case"]})
            events = record["events"]
        turn = {
            "role": "assistant",
            "events": events,
            "error": record["error"],
            "usage": record["usage"],
            "seconds": record["seconds"],
        }
    else:
        case = st.session_state.cases[case_id]["case"]
        turn = {
            "role": "assistant",
            "events": [{
                "role": "note",
                "text": f"Back on {case['customer_id']} · ${case['amount']:,.2f} at {case['merchant']}.",
            }],
            "usage": {},
            "seconds": None,
            "error": None,
        }
    st.session_state.chat.append(turn)
    if thread is not None:
        with thread:
            if announce or not cached:
                render_turn(st.session_state.chat[-2], len(st.session_state.chat) - 2)
            render_turn(st.session_state.chat[-1], len(st.session_state.chat) - 1)
    st.session_state.active_id = case_id
    st.session_state.handled_pick = case_id


def _consume_pick(store):
    """Open the transaction chosen from a clickable list."""
    row_index = st.session_state.pop("pick_row")
    label = st.session_state.pop("pick_label", "Selected transaction")
    st.session_state.pending_choices = None
    st.session_state.chat.append({"role": "user", "text": label})
    _open_case(get_flagged().iloc[int(row_index)], store, None, announce=True)


def _answer(question, store):
    try:
        if st.session_state.active_id:
            record = st.session_state.cases[st.session_state.active_id]
            return ask_about_case(record, store, question)
        return answer_general(question)
    except Exception as exc:
        return {
            "events": [{"role": "assistant", "text": f"That question did not go through: {exc}"}],
            "usage": {},
            "seconds": 0,
            "error": None,
        }
