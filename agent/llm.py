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
        response = self.client.chat.completions.create(model=self.model, messages=messages, tools=tools, **options)
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


def parse_decision(text):
    """Parse the model's final answer: a JSON object, possibly wrapped in a code fence or prose."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("no JSON object in the reply")
    return Decision.from_dict(json.loads(text[start:end + 1]))


@dataclass
class LLMReviewer:
    backend: object
    include_score: bool = True  # False = system C: the XGBoost score is withheld
    allowed_tools: list = field(default_factory=lambda: list(ALL_TOOLS))  # HISTORY_TOOLS = system D
    max_turns: int = 8  # model calls per case before giving up
    name: str = ""

    def __post_init__(self):
        unknown = set(self.allowed_tools) - set(ALL_TOOLS)
        if unknown:
            raise ValueError(f"unknown tools: {sorted(unknown)}")
        self.tool_definitions = [d for d in TOOL_DEFINITIONS if d["name"] in self.allowed_tools]
        if not self.name:
            variant = ("" if self.include_score else "-noscore") + ("" if set(self.allowed_tools) == set(ALL_TOOLS) else "-nokyc")
            self.name = f"llm-{self.backend.provider}-{self.backend.model}{variant}".replace("/", "_")
        self.last_usage = None

    def review(self, case, tools):
        backend = self.backend
        messages = [backend.system(SYSTEM_PROMPT), backend.user(render_case(case, self.include_score))]
        usage = {"model": backend.model, "turns": 0, "input_tokens": 0, "output_tokens": 0}
        self.last_usage = usage  # filled in as the loop runs, so a failed case still reports its spend
        retried_format = False

        for _ in range(self.max_turns):
            response = backend.chat(messages, self.tool_definitions)
            usage["turns"] += 1
            usage["input_tokens"] += response.input_tokens
            usage["output_tokens"] += response.output_tokens
            messages.append(response.assistant_message)

            if response.tool_calls:
                for call in response.tool_calls:
                    if call["name"] not in self.allowed_tools:
                        result = {"error": f"tool {call['name']!r} is not available"}
                    elif call["arguments"] is None:
                        result = {"error": "the tool arguments were not valid JSON"}
                    else:
                        result = tools.call(call["name"], call["arguments"])
                    messages.append(backend.tool_result(call["id"], result))
                continue

            try:
                return parse_decision(response.text)
            except (ValueError, KeyError, TypeError) as error:
                if retried_format:
                    raise
                retried_format = True
                messages.append(backend.user(
                    f"That reply could not be read as a decision ({error}). "
                    "Reply with only the JSON object described in your instructions."
                ))

        raise RuntimeError(f"no decision after {self.max_turns} model turns")
