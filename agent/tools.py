"""The agent's tools, bound to one case.

A CaseTools object is created per flagged transaction with that case's
customer and time fixed. The tools take no customer or date arguments, so a
reviewer cannot ask for another customer, or for anything at or after the
transaction under review. Definitions are plain JSON Schema, so any LLM API
with tool calling can use them.
"""

import json

import numpy as np
import pandas as pd

TOOL_DEFINITIONS = [
    {
        "name": "get_kyc_profile",
        "description": (
            "The customer's KYC (Know Your Customer) file: declared income and monthly spend, "
            "employment, residency, credit score and limit, how identity was verified, screening "
            "flags, and the bank's risk rating."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_transaction_history",
        "description": (
            "The customer's most recent transactions before the one under review, oldest first: "
            "time, amount, merchant and category."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "description": "How many recent transactions to return (default 20).",
                }
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_spending_summary",
        "description": (
            "Summary of the customer's behaviour before the transaction under review: usual amounts "
            "overall and per category, time-of-day pattern, and recent activity (counts over the last "
            "1 hour, 24 hours and 7 days, spend in the last 24 hours, time since the previous transaction)."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
]

TIME_OF_DAY = {"night (00–05)": range(0, 6), "morning (06–11)": range(6, 12),
               "afternoon (12–17)": range(12, 18), "evening (18–23)": range(18, 24)}


def _jsonable(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 2)
    return value


class CaseTools:
    definitions = TOOL_DEFINITIONS

    def __init__(self, store, cc_num, before):
        self._store = store
        self._cc_num = cc_num
        self._before = pd.Timestamp(before)
        self.calls = []  # (name, arguments) in call order, for the run log

    def call(self, name, arguments=None):
        """Run a tool by name. Returns a JSON-serialisable dict; unknown tools or bad arguments return an error dict."""
        arguments = arguments or {}
        self.calls.append({"name": name, "arguments": arguments})
        try:
            if name == "get_kyc_profile" and not arguments:
                return self.get_kyc_profile()
            if name == "get_transaction_history" and set(arguments) <= {"limit"}:
                return self.get_transaction_history(**arguments)
            if name == "get_spending_summary" and not arguments:
                return self.get_spending_summary()
        except (TypeError, ValueError) as error:
            return {"error": str(error)}
        return {"error": f"unknown tool or unexpected arguments: {name} {json.dumps(arguments)}"}

    def get_kyc_profile(self):
        return {k: _jsonable(v) for k, v in self._store.kyc_profile(self._cc_num).items()}

    def get_transaction_history(self, limit=20):
        limit = int(limit)
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        history = self._store.history(self._cc_num, self._before, limit=limit)
        return {
            "transactions": [
                {"time": pd.Timestamp(t).isoformat(), "amount": round(float(a), 2), "merchant": m, "category": c}
                for t, a, m, c in history.itertuples(index=False)
            ],
        }

    def get_spending_summary(self):
        history = self._store.history(self._cc_num, self._before)
        if history.empty:
            return {"earlier_transactions": 0, "note": "This is the customer's first transaction."}

        amount = history["amount"]
        age = self._before - history["time"]
        hours = history["time"].dt.hour
        by_category = history.groupby("category")["amount"].agg(
            count="size", median="median", p95=lambda s: s.quantile(0.95)
        )
        return {
            "earlier_transactions": len(history),
            "first_transaction": history["time"].iloc[0].date().isoformat(),
            "amount_overall": {"median": _jsonable(amount.median()), "p95": _jsonable(amount.quantile(0.95)),
                               "max": _jsonable(amount.max())},
            "amount_by_category": {
                cat: {"count": int(r["count"]), "median": _jsonable(r["median"]), "p95": _jsonable(r["p95"])}
                for cat, r in by_category.iterrows()
            },
            "share_by_time_of_day": {
                name: round(float(hours.isin(hrs).mean()), 3) for name, hrs in TIME_OF_DAY.items()
            },
            "recent_activity": {
                "transactions_last_1h": int((age <= pd.Timedelta(hours=1)).sum()),
                "transactions_last_24h": int((age <= pd.Timedelta(hours=24)).sum()),
                "transactions_last_7d": int((age <= pd.Timedelta(days=7)).sum()),
                "spend_last_24h": _jsonable(amount[age <= pd.Timedelta(hours=24)].sum()),
                "hours_since_previous_transaction": _jsonable(age.iloc[-1] / pd.Timedelta(hours=1)),
            },
        }
