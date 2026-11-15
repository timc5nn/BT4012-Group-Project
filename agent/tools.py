"""The agent's tools, bound to one case.

A CaseTools object is created per flagged transaction with that case's
customer and time fixed. The tools take no customer or date arguments, so a
reviewer cannot ask for another customer, or for anything at or after the
transaction under review. Definitions are plain JSON Schema, so any LLM API
with tool calling can use them.
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import THRESHOLDS_FILE

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
    {
        "name": "get_merchant_history",
        "description": (
            "The customer's history with the flagged transaction's merchant: whether it is new, "
            "the number of earlier visits, first and last visit, and typical amounts."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_amount_deviation",
        "description": (
            "Compares the flagged amount with the customer's overall, category, and merchant history, "
            "including median, 95th percentile, and percentile rank."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_velocity_profile",
        "description": (
            "A point-in-time velocity profile: transaction counts, spend, and time gaps over the "
            "previous 5 minutes, 1 hour, and 24 hours."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_location_history",
        "description": (
            "Compares the flagged transaction's merchant location with earlier customer locations, "
            "including distance from the previous transaction and whether the merchant location is new."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_model_explanation",
        "description": (
            "Explains the XGBoost model context: the case score, flag threshold, and standalone threshold. "
            "The score prioritizes cases and is not proof of fraud."
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

    def __init__(self, store, cc_num, before, amount=None, merchant=None, category=None,
                 latitude=None, longitude=None, merchant_latitude=None, merchant_longitude=None,
                 xgb_score=None):
        self._store = store
        self._cc_num = cc_num
        self._before = pd.Timestamp(before)
        context = store.transaction_context(cc_num, before)
        context = context or {}
        self._amount = amount if amount is not None else context.get("amount")
        self._merchant = merchant if merchant is not None else context.get("merchant")
        self._category = category if category is not None else context.get("category")
        self._latitude = latitude if latitude is not None else context.get("latitude")
        self._longitude = longitude if longitude is not None else context.get("longitude")
        self._merchant_latitude = (
            merchant_latitude if merchant_latitude is not None else context.get("merchant_latitude")
        )
        self._merchant_longitude = (
            merchant_longitude if merchant_longitude is not None else context.get("merchant_longitude")
        )
        self._xgb_score = xgb_score
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
            if name in {
                "get_merchant_history", "get_amount_deviation", "get_velocity_profile",
                "get_location_history", "get_model_explanation",
            } and not arguments:
                return getattr(self, name)()
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
                for t, a, m, c in history[["time", "amount", "merchant", "category"]].itertuples(index=False)
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

    def get_merchant_history(self):
        if self._merchant is None:
            return {"error": "current merchant is not available for this case"}
        history = self._store.history(self._cc_num, self._before)
        visits = history[history["merchant"] == self._merchant]
        if visits.empty:
            return {"merchant": self._merchant, "first_time": True, "earlier_transactions": 0}
        return {
            "merchant": self._merchant,
            "first_time": False,
            "earlier_transactions": len(visits),
            "first_seen": pd.Timestamp(visits["time"].iloc[0]).isoformat(),
            "last_seen": pd.Timestamp(visits["time"].iloc[-1]).isoformat(),
            "amounts": {
                "median": _jsonable(visits["amount"].median()),
                "p95": _jsonable(visits["amount"].quantile(0.95)),
                "max": _jsonable(visits["amount"].max()),
            },
        }

    def get_amount_deviation(self):
        if self._amount is None or self._category is None:
            return {"error": "current amount and category are not available for this case"}
        history = self._store.history(self._cc_num, self._before)
        if history.empty:
            return {"earlier_transactions": 0, "note": "No earlier amount history is available."}

        def stats(values):
            values = values.astype(float)
            return {
                "count": len(values),
                "median": _jsonable(values.median()),
                "p95": _jsonable(values.quantile(0.95)),
                "percentile_rank": round(float((values <= self._amount).mean()), 3),
            }

        category_values = history.loc[history["category"] == self._category, "amount"]
        merchant_values = history.loc[history["merchant"] == self._merchant, "amount"]
        result = {
            "current_amount": _jsonable(self._amount),
            "overall": stats(history["amount"]),
            "category": {"name": self._category, **stats(category_values)} if not category_values.empty else None,
            "merchant": {"name": self._merchant, **stats(merchant_values)} if not merchant_values.empty else None,
        }
        return result

    def get_velocity_profile(self):
        history = self._store.history(self._cc_num, self._before)
        if history.empty:
            return {"earlier_transactions": 0, "windows": {}}
        age = self._before - history["time"]
        windows = {}
        for name, duration in (("last_5m", 5), ("last_1h", 60), ("last_24h", 1440)):
            recent = history[age <= pd.Timedelta(minutes=duration)]
            windows[name] = {
                "transactions": len(recent),
                "spend": _jsonable(recent["amount"].sum()),
            }
        gaps = age.dt.total_seconds() / 3600
        return {
            "earlier_transactions": len(history),
            "hours_since_previous_transaction": _jsonable(gaps.iloc[-1]),
            "windows": windows,
        }

    def get_location_history(self):
        required = (
            self._latitude, self._longitude, self._merchant_latitude, self._merchant_longitude
        )
        if any(value is None for value in required):
            return {"error": "transaction location is not available for this case"}
        history = self._store.history(self._cc_num, self._before)

        def distance(lat1, lon1, lat2, lon2):
            radius = 6371
            lat1, lat2 = math.radians(float(lat1)), math.radians(float(lat2))
            dlat = lat2 - lat1
            dlon = math.radians(float(lon2) - float(lon1))
            value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
            return 2 * radius * math.asin(math.sqrt(value))

        result = {
            "customer_latitude": _jsonable(self._latitude),
            "customer_longitude": _jsonable(self._longitude),
            "merchant_latitude": _jsonable(self._merchant_latitude),
            "merchant_longitude": _jsonable(self._merchant_longitude),
            "earlier_locations": len(history),
        }
        if history.empty:
            result["first_location"] = True
            return result
        previous = history.iloc[-1]
        result.update({
            "first_location": False,
            "distance_from_previous_merchant_km": round(distance(
                previous["merch_lat"], previous["merch_long"],
                self._merchant_latitude, self._merchant_longitude,
            ), 2),
            "distance_from_previous_customer_km": round(distance(
                previous["lat"], previous["long"], self._latitude, self._longitude,
            ), 2),
        })
        return result

    def get_model_explanation(self):
        if self._xgb_score is None:
            return {"error": "XGBoost score is not available for this case"}
        thresholds = json.loads(Path(THRESHOLDS_FILE).read_text())
        score = float(self._xgb_score)
        return {
            "xgb_score": round(score, 4),
            "flag_threshold": thresholds["flag"],
            "standalone_threshold": thresholds["standalone"],
            "above_flag_threshold": score >= thresholds["flag"],
            "above_standalone_threshold": score >= thresholds["standalone"],
            "interpretation": "The score prioritizes cases for review; it is not a final decision.",
        }
