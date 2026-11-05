"""LLM reviewer: an agent loop over any OpenAI-compatible chat API.

OpenAI, Google Gemini, OpenRouter and local Ollama all accept the same
OpenAI-style chat-completions request, so one backend covers them; the
provider is a setting. Requires `pip install openai` (only when a run starts).

    backend = OpenAICompatibleBackend("openrouter", "<model name>")
    reviewer = LLMReviewer(backend)                          # system B
    reviewer = LLMReviewer(backend, include_score=False)     # system C: score withheld
    reviewer = LLMReviewer(backend, allowed_tools=HISTORY_TOOLS)  # system D: no KYC tool

A backend is any object with `provider`, `model`, and the four methods used
below (chat, system, user, tool_result), so another API can be added by
writing one class.
"""

import json
import os
import re
from dataclasses import dataclass, field

from .prompt import SYSTEM_PROMPT, render_case
from .schemas import Decision
from .tools import TOOL_DEFINITIONS

ALL_TOOLS = [d["name"] for d in TOOL_DEFINITIONS]
HISTORY_TOOLS = ["get_transaction_history", "get_spending_summary"]

# Base URLs and key variables per provider. Check them against the provider's
# current docs before the first run; they are the only provider-specific part.
PROVIDERS = {
    "openai": {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "api_key_env": "GEMINI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    "ollama": {"base_url": "http://localhost:11434/v1", "api_key_env": None},
}


@dataclass
class ChatResponse:
    text: str  # the model's text reply ("" when it only called tools)
    tool_calls: list  # [{"id", "name", "arguments": dict, or None if not valid JSON}]
    assistant_message: dict  # the reply, in the backend's format, to append to the conversation
    input_tokens: int = 0
    output_tokens: int = 0


class OpenAICompatibleBackend:
    def __init__(self, provider, model, temperature=None, timeout=120, max_retries=3):
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; choose from {sorted(PROVIDERS)}")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The LLM reviewer needs the openai package: pip install openai") from exc

        config = PROVIDERS[provider]
        key_env = config["api_key_env"]
        api_key = os.environ.get(key_env) if key_env else "ollama"  # Ollama ignores the key but the client needs one
        if not api_key:
            raise RuntimeError(f"Set the {key_env} environment variable for provider {provider!r}")

        self.provider, self.model, self.temperature = provider, model, temperature
        self.client = OpenAI(api_key=api_key, base_url=config["base_url"], timeout=timeout, max_retries=max_retries)

    @staticmethod
    def system(text):
        return {"role": "system", "content": text}

    @staticmethod
    def user(text):
        return {"role": "user", "content": text}

    @staticmethod
    def tool_result(call_id, result):
        return {"role": "tool", "tool_call_id": call_id, "content": json.dumps(result)}

    def chat(self, messages, tool_definitions):
        tools = [{"type": "function", "function": {"name": d["name"], "description": d["description"],
                                                   "parameters": d["input_schema"]}} for d in tool_definitions]
        options = {} if self.temperature is None else {"temperature": self.temperature}
        if tools:
            options["tools"] = tools
        response = self.client.chat.completions.create(model=self.model, messages=messages, **options)
        message = response.choices[0].message

        calls = []
        for call in message.tool_calls or []:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                arguments = None
            calls.append({"id": call.id, "name": call.function.name, "arguments": arguments})

        assistant = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls
            ]
        usage = response.usage
        return ChatResponse(
            text=message.content or "",
            tool_calls=calls,
            assistant_message=assistant,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


TOOL_NUDGE = (
    "Call the tools before you decide. get_kyc_profile and get_spending_summary take {}, "
    "and get_transaction_history takes {\"limit\": 20} only. Do not pass a customer id or a date."
)
HUMAN_REVIEW_THRESHOLD = 0.60
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])(?:\d+(?:\.\d+)?)(?![A-Za-z])")


def prepare_tool_arguments(name, arguments):
    """Drop anything the tool is not allowed to use, and clamp `limit`.

    A customer id or a timestamp cannot select a different customer or time.
    The case binding already fixed both. Returns (arguments to pass, dropped keys).
    """
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    spec = next((d for d in TOOL_DEFINITIONS if d["name"] == name), None)
    if spec is None:
        raise ValueError(f"unknown tool {name!r}")
    allowed = set(spec["input_schema"].get("properties", {}))
    cleaned, dropped = {}, []
    for key, value in arguments.items():
        if key in allowed:
            cleaned[key] = value
        else:
            dropped.append(key)
    if "limit" in cleaned:
        try:
            limit = int(cleaned["limit"])
        except (TypeError, ValueError):
            limit = 20
        cleaned["limit"] = min(100, max(1, limit))
    return cleaned, dropped


def parse_decision(text):
    """Parse Qwen's JSON object, including harmless prose, fences, or evidence objects."""
    text = re.sub(r"```(?:json)?|```", "", text.strip(), flags=re.IGNORECASE).strip()
    decoder = json.JSONDecoder()
    parsed = None
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[start:])
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(parsed, dict):
        raise ValueError("no valid JSON object in the reply")
    if isinstance(parsed.get("evidence"), list):
        parsed["evidence"] = [
            item if isinstance(item, str) else item.get("statement", "")
            for item in parsed["evidence"]
        ]
    return Decision.from_dict(parsed)


def rule_checks(case, evidence):
    """Create deterministic findings before the model is asked to analyse."""
    findings = []
    summary = evidence.get("get_spending_summary", {})
    overall = summary.get("amount_overall", {})
    median = overall.get("median")
    if median and case.amount / median >= 5:
        findings.append(f"Current amount {case.amount:.2f} is at least five times the historical median {median:.2f}.")
    recent = summary.get("recent_activity", {})
    if recent.get("transactions_last_1h", 0) >= 3:
        findings.append(
            f"There were {recent['transactions_last_1h']} transactions in the previous hour."
        )
    if recent.get("transactions_last_24h", 0) >= 8:
        findings.append(
            f"There were {recent['transactions_last_24h']} transactions in the previous 24 hours."
        )
    hours = recent.get("hours_since_previous_transaction")
    if hours is not None and hours < 0.25:
        findings.append(f"Only {hours:.2f} hours elapsed since the previous transaction.")
    if not findings:
        findings.append("The deterministic checks found no high-risk velocity or amount rule match.")
    return findings


def validate_evidence(decision, evidence):
    """Reject numeric claims that are absent from the case or collected results."""
    corpus = json.dumps(evidence, default=str)
    for statement in decision.evidence:
        for number in NUMBER_PATTERN.findall(statement):
            if number not in corpus:
                raise ValueError(f"evidence contains unsupported number {number!r}")


@dataclass
class LLMReviewer:
    backend: object
    include_score: bool = True  # False = system C: the XGBoost score is withheld
    allowed_tools: list = field(default_factory=lambda: list(ALL_TOOLS))  # HISTORY_TOOLS = system D
    max_turns: int = 8  # model calls per case before giving up
    human_review_threshold: float = HUMAN_REVIEW_THRESHOLD
    name: str = ""

    def __post_init__(self):
        unknown = set(self.allowed_tools) - set(ALL_TOOLS)
        if unknown:
            raise ValueError(f"unknown tools: {sorted(unknown)}")
        if not 0 <= self.human_review_threshold <= 1:
            raise ValueError("human_review_threshold must be between 0 and 1")
        self.tool_definitions = [d for d in TOOL_DEFINITIONS if d["name"] in self.allowed_tools]
        if not self.name:
            variant = ("" if self.include_score else "-noscore") + ("" if set(self.allowed_tools) == set(ALL_TOOLS) else "-nokyc")
            self.name = f"llm-{self.backend.provider}-{self.backend.model}{variant}".replace("/", "_")
        self.last_usage = None
        self.last_transcript = []  # display events for one review: user, assistant, tool, note

    def _collect_required_tools(self, tools):
        """Collect every enabled tool before the first LLM analysis turn."""
        evidence = {}
        for definition in self.tool_definitions:
            name = definition["name"]
            arguments = {"limit": 20} if name == "get_transaction_history" else {}
            result = tools.call(name, arguments)
            evidence[name] = result
            self.last_transcript.append({"role": "tool", "name": name, "arguments": arguments, "result": result})
        return evidence

    def review(self, case, tools):
        backend = self.backend
        opening = render_case(case, self.include_score)
        self.last_transcript = [{"role": "user", "text": opening}]
        evidence = self._collect_required_tools(tools)
        findings = rule_checks(case, evidence)
        evidence_context = {
            "flagged_transaction": case.to_dict(include_score=self.include_score),
            "required_tool_results": evidence,
            "deterministic_rule_findings": findings,
        }
        messages = [backend.system(SYSTEM_PROMPT), backend.user(opening)]
        messages.append(backend.user(
            "The required evidence has already been collected before analysis. "
            "Use it as the primary source of truth:\n" + json.dumps(evidence_context, default=str)
        ))
        usage = {"model": backend.model, "turns": 0, "input_tokens": 0, "output_tokens": 0}
        self.last_usage = usage  # filled in as the loop runs, so a failed case still reports its spend
        self.last_messages = messages
        retried_format = False
        nudged_tools = False
        saw_tool_data = True

        for _ in range(self.max_turns):
            response = backend.chat(messages, self.tool_definitions)
            usage["turns"] += 1
            usage["input_tokens"] += response.input_tokens
            usage["output_tokens"] += response.output_tokens
            messages.append(response.assistant_message)

            if response.tool_calls:
                if self._dispatch_tool_calls(response, tools, messages):
                    saw_tool_data = True
                continue

            if not saw_tool_data and not nudged_tools:
                nudged_tools = True
                messages.append(backend.user(TOOL_NUDGE))
                self.last_transcript.append({
                    "role": "note",
                    "text": "No tool result yet. Asking for the three lookups before a decision.",
                })
                continue

            self.last_transcript.append({"role": "assistant", "text": response.text})
            try:
                decision = parse_decision(response.text)
                validate_evidence(decision, evidence_context)
            except (ValueError, KeyError, TypeError) as error:
                if retried_format:
                    raise
                retried_format = True
                messages.append(backend.user(
                    f"That reply could not be read as a decision ({error}). "
                    "Reply with only the JSON object described in your instructions."
                ))
                self.last_transcript.append({
                    "role": "note",
                    "text": f"That reply was not a valid decision ({error}). Asking again.",
                })
                continue
            if decision.confidence < self.human_review_threshold:
                note = (
                    f"Confidence {decision.confidence:.0%} is below the "
                    f"{self.human_review_threshold:.0%} automatic-review threshold. "
                    "A human analyst must make the final decision. " + decision.case_note
                )
                decision = Decision(
                    decision="needs_human_review",
                    confidence=decision.confidence,
                    evidence=decision.evidence,
                    case_note=" ".join(note.split()[:120]),
                )
            self.last_transcript[-1]["decision"] = decision.to_dict()
            self.last_messages = messages
            return decision

        self.last_messages = messages
        raise RuntimeError(f"no decision after {self.max_turns} model turns")

    def ask(self, messages, tools, question):
        """Answer a follow-up about the same case. Tools stay bound to that case."""
        backend = self.backend
        messages = list(messages)
        messages.append(backend.user(
            f"The analyst asks: {question}\n\n"
            "Answer in plain language. Call a tool only if you need a lookup that is not already in the conversation. "
            "Every number must come from a tool result or the flagged transaction. "
            "Do not reply with the decision JSON unless they ask you to decide again."
        ))
        usage = {"model": backend.model, "turns": 0, "input_tokens": 0, "output_tokens": 0}
        self.last_usage = usage
        self.last_transcript = []
        self.last_messages = messages

        for _ in range(self.max_turns):
            response = backend.chat(messages, self.tool_definitions)
            usage["turns"] += 1
            usage["input_tokens"] += response.input_tokens
            usage["output_tokens"] += response.output_tokens
            messages.append(response.assistant_message)
            if response.tool_calls:
                self._dispatch_tool_calls(response, tools, messages)
                continue
            text = (response.text or "").strip() or "I don't have anything further from the tools."
            event = {"role": "assistant", "text": text}
            try:
                event["decision"] = parse_decision(text).to_dict()
            except (ValueError, KeyError, TypeError):
                pass
            self.last_transcript.append(event)
            self.last_messages = messages
            return text

        self.last_messages = messages
        raise RuntimeError(f"no answer after {self.max_turns} model turns")

    def _dispatch_tool_calls(self, response, tools, messages):
        """Run one round of tool calls. Returns whether any call returned data."""
        saw_data = False
        self.last_transcript.append({
            "role": "assistant",
            "text": response.text,
            "tool_calls": [{"name": call["name"], "arguments": call["arguments"]} for call in response.tool_calls],
        })
        for call in response.tool_calls:
            dropped = []
            if call["name"] not in self.allowed_tools:
                result = {"error": f"tool {call['name']!r} is not available"}
            elif call["arguments"] is None:
                result = {"error": "the tool arguments were not valid JSON"}
            else:
                try:
                    used, dropped = prepare_tool_arguments(call["name"], call["arguments"])
                except ValueError as error:
                    result = {"error": str(error)}
                else:
                    result = tools.call(call["name"], used)
            if isinstance(result, dict) and "error" not in result:
                saw_data = True
            messages.append(self.backend.tool_result(call["id"], result))
            self.last_transcript.append({
                "role": "tool",
                "name": call["name"],
                "arguments": call["arguments"],
                "dropped": dropped,
                "result": result,
            })
        return saw_data
