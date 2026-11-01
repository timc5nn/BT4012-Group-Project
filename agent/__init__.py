"""Stage 2 of the fraud pipeline: review XGBoost's flagged transactions.

Modules:
    data       label-free, point-in-time access to transactions and KYC
    tools      the agent's tools, bound to one case
    schemas    the Case shown to a reviewer and the Decision it returns
    prompt     provider-neutral system prompt and case rendering
    reviewers  the Reviewer interface and a no-LLM baseline
    llm        the LLM reviewer (agent loop over any OpenAI-compatible API)
    pipeline   run a reviewer over flagged cases, then evaluate
"""
