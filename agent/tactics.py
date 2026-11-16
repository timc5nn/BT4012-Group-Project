"""Fraud-tactic documents and the checklist matcher behind the tactic RAG reviewer.

Each tactic document has two parts:
  - text: how the tactic works, with one illustrative example. This is what the model reads.
  - a checklist of signals the code tests against the case's tool results:
      core        every core signal must be present for the tactic to match at all
      supporting  each present supporting signal strengthens the match
      against     each present innocent explanation weakens it

Retrieval is by checklist, not by wording: the code runs the case tools, ticks every
checklist, keeps the tactics whose core signals are all present, ranks them by
(supporting - against), and sends the top three to the model with the ticks shown.
The LOOKALIKES document is always sent.

The tactics were chosen from a check of the fit rows (the first 80% of fraudTrain.csv,
before every validation case): late-night spending and amounts far above a customer's
usual were the strongest signals; online categories were moderate; fraud arrives as a
burst of about 10 transactions over two days; small-then-large charges appear but are
weak on their own. Merchant distance from home was identical for fraud and legitimate
transactions, so there is no location tactic. The documents describe mechanisms, not
statistics, and nothing here reads a fraud label.

History-based signals use the last 100 earlier transactions (the transaction-history
tool's maximum), so "new merchant in the last 24 hours" means "not seen in the rest of
those 100 rows"; for the flagged merchant itself the merchant-history tool is exact.
"""

from dataclasses import dataclass, field

import pandas as pd

LATE_HOURS = {22, 23, 0, 1, 2, 3}  # 22:00-03:59
SMALL_AMOUNT, LARGE_AMOUNT, PAIR_WINDOW_HOURS = 20.0, 200.0, 6
HISTORY_LIMIT = 100
TACTIC_TOOLS = [  # what the checklists read; run by code before the model's first turn
    ("get_spending_summary", {}),
    ("get_amount_deviation", {}),
    ("get_merchant_history", {}),
    ("get_transaction_history", {"limit": HISTORY_LIMIT}),
]


@dataclass(frozen=True)
class Tactic:
    document_id: str
    title: str
    text: str
    core: tuple
    supporting: tuple
    against: tuple
    strong_only_with: str = None  # cap at "moderate" unless this other tactic also matches


TACTICS = [
    Tactic(
        "tactic_late_night_out_of_character",
        "Out-of-character late-night spending",
        "How it works: someone using stolen card details makes large purchases late at night, "
        "when the real cardholder is asleep and will not see an alert until morning. They often "
        "use merchants the cardholder has never visited and spend far more than the cardholder "
        "normally does. A late purchase or a large purchase alone is common and is not fraud; "
        "the tactic is the combination.\n"
        "Example (illustrative, not a real case): $940 at a merchant the customer has never used, "
        "at 01:15. The customer's usual purchase is $45, their largest ever was $310, and they "
        "almost never spend after 22:00. All signs fit and no innocent explanation applies.",
        core=("amount_5x_usual", "late_night"),
        supporting=("new_merchant", "rarely_late", "largest_ever"),
        against=("large_purchases_routine", "regular_merchant_similar_amounts", "often_late"),
    ),
    Tactic(
        "tactic_stolen_card_spree",
        "Stolen-card spree",
        "How it works: once a card is stolen it is used as much as possible before it is "
        "blocked, typically a burst of purchases over one or two days, roughly an hour apart, "
        "at several merchants the cardholder has never used and across several categories. "
        "The burst as a whole is out of character even when single purchases look ordinary.\n"
        "Example (illustrative, not a real case): in the 24 hours before the flagged purchase "
        "the card was used 7 times at 5 merchants it had never used, in 4 categories, spending "
        "$2,600 against a usual $90 a day. The customer has never had a day like this.",
        core=("new_merchants_24h", "spend_24h_3x_daily"),
        supporting=("new_merchant", "many_categories_24h", "burst_3h", "late_night"),
        against=("had_days_this_big", "recent_merchants_familiar"),
    ),
    Tactic(
        "tactic_online_purchase",
        "Online (card-not-present) purchase fraud",
        "How it works: stolen card numbers are easiest to use online, where no physical card "
        "or PIN is needed. A typical case is a large online purchase at a merchant the customer "
        "has never used, often at night, from a customer who rarely shops online or only spends "
        "small amounts online.\n"
        "Example (illustrative, not a real case): $1,050 at a new online merchant at 23:40. The "
        "customer's usual purchase is $60, and their few online purchases were all under $30.",
        core=("online_category", "amount_5x_usual"),
        supporting=("new_merchant", "rarely_online", "late_night"),
        against=("online_similar_before", "regular_merchant_similar_amounts"),
    ),
    Tactic(
        "tactic_small_charges_in_spree",
        "Small charges inside a spree",
        "How it works: a fraudster may make a small charge to check that a stolen card works, "
        "or mix small purchases into a spree, close in time to a large purchase. In this data "
        "small fraud charges are spread through a spree rather than only at the start, so this "
        "pattern is weak on its own and matters mainly together with new merchants or late hours.\n"
        "Example (illustrative, not a real case): $4.20 at a new merchant at 00:50, then $880 at "
        "another new merchant at 02:10. The customer rarely makes purchases under $20 at "
        "unfamiliar merchants.",
        core=("small_large_pair",),
        supporting=("pair_small_new_merchant", "pair_large_new_merchant", "late_night"),
        against=("pair_small_merchant_regular",),
        # Fit rows: "strong" was 86% fraud alone vs 97% alongside a spree match (sample 50% fraud).
        strong_only_with="tactic_stolen_card_spree",
    ),
]
TACTICS_BY_ID = {t.document_id: t for t in TACTICS}

LOOKALIKES = (
    "Common innocent explanations (always check these before raising an alert):\n"
    "- A planned large purchase at a merchant the customer already uses, or a size of "
    "purchase the customer has made before.\n"
    "- A customer who often shops late at night or online; for them late or online "
    "spending is normal.\n"
    "- A busy day of errands at familiar merchants: many transactions but no new merchants.\n"
    "- A single unusual sign (late, large, new merchant) with everything else ordinary.\n"
    "The XGBoost flag means the case deserves review, not that it is fraud."
)


@dataclass
class TacticMatch:
    document_id: str
    title: str
    matched: bool  # every core signal present
    strength: str  # strong / moderate / weak / no match
    score: int  # supporting present - against present
    core: list = field(default_factory=list)  # [{"signal", "present", "detail"}]
    supporting: list = field(default_factory=list)
    against: list = field(default_factory=list)
    capped: str = ""  # why the strength was lowered, if it was


# ---------------------------------------------------------------------------
# Facts: the numbers every signal reads, computed once per case from tool results
# ---------------------------------------------------------------------------

def _money(value):
    return f"{value:,.2f}"


def case_facts(case, evidence, history_rows=None):
    """Derive the checklist inputs from the flagged transaction and the tool results."""
    time = pd.Timestamp(case.time)
    amount = float(case.amount)
    summary = evidence.get("get_spending_summary") or {}
    deviation = evidence.get("get_amount_deviation") or {}
    merchant = evidence.get("get_merchant_history") or {}
    if history_rows is None:
        history_rows = (evidence.get("get_transaction_history") or {}).get("transactions") or []

    overall = summary.get("amount_overall") or {}
    median = (deviation.get("overall") or {}).get("median") or overall.get("median")
    facts = {
        "time": time,
        "hour": time.hour,
        "amount": amount,
        "category": case.category,
        "merchant": case.merchant,
        "median": median,
        "ratio": amount / median if median else None,
        "max_before": overall.get("max"),
        "p95": (deviation.get("overall") or {}).get("p95") or overall.get("p95"),
        "first_time": merchant.get("first_time") if "error" not in merchant else None,
        "merchant_visits": merchant.get("earlier_transactions"),
        "merchant_max": (merchant.get("amounts") or {}).get("max"),
        "history_n": len(history_rows),
    }

    hist = pd.DataFrame(history_rows, columns=["time", "amount", "merchant", "category"])
    hist["time"] = pd.to_datetime(hist["time"], format="ISO8601")
    hist["amount"] = hist["amount"].astype(float)
    if hist.empty:
        return facts

    hist["late"] = hist["time"].dt.hour.isin(LATE_HOURS)
    hist["online"] = hist["category"].str.endswith("_net")
    facts["late_share"] = float(hist["late"].mean())
    facts["online_share"] = float(hist["online"].mean())
    facts["online_max"] = float(hist.loc[hist["online"], "amount"].max()) if hist["online"].any() else None

    window_start = time - pd.Timedelta(hours=24)
    recent = hist[hist["time"] >= window_start]
    older = hist[hist["time"] < window_start]
    known = set(older["merchant"])
    facts["older_n"] = len(older)
    facts["n_24h"] = len(recent)
    facts["spend_24h"] = float(recent["amount"].sum())
    facts["categories_24h"] = int(recent["category"].nunique())
    facts["merchants_24h"] = int(recent["merchant"].nunique())
    facts["new_merchants_24h"] = int(len(set(recent["merchant"]) - known)) if len(older) else None
    facts["n_3h"] = int((hist["time"] >= time - pd.Timedelta(hours=3)).sum())
    if len(older):
        days = max(1.0, (window_start - older["time"].min()) / pd.Timedelta(days=1))
        facts["daily_spend"] = float(older["amount"].sum()) / days
        facts["max_day_spend"] = float(older.groupby(older["time"].dt.date)["amount"].sum().max())
        facts["older_days"] = round(days)

    # Small/large pair within PAIR_WINDOW_HOURS, among the last 24h and the flagged transaction
    rows = recent[["time", "amount", "merchant"]].to_dict("records")
    rows.append({"time": time, "amount": amount, "merchant": case.merchant, "current": True})
    window = pd.Timedelta(hours=PAIR_WINDOW_HOURS)
    pairs = [(s, b) for s in rows if s["amount"] < SMALL_AMOUNT for b in rows
             if b["amount"] > LARGE_AMOUNT and abs(b["time"] - s["time"]) <= window
             and (s.get("current") or b.get("current"))]  # the pair must involve the flagged transaction
    if pairs:
        small, large = min(pairs, key=lambda p: abs(p[1]["time"] - p[0]["time"]))

        def is_new(row):
            if row.get("current"):
                return facts["first_time"]
            return row["merchant"] not in known
        facts["pair"] = {
            "small": small, "large": large,
            "gap_hours": abs(large["time"] - small["time"]) / pd.Timedelta(hours=1),
            "small_new": is_new(small), "large_new": is_new(large),
            # earlier visits to the small charge's merchant, before that charge
            "small_merchant_visits": (int(facts["merchant_visits"] or 0) if small.get("current")
                                      else int((older["merchant"] == small["merchant"]).sum())),
        }
    return facts


# ---------------------------------------------------------------------------
# Signals: each returns (present: True/False/None, detail text with the numbers)
# ---------------------------------------------------------------------------

def _amount_5x_usual(f):
    if f["ratio"] is None:
        return None, "amount at least 5x the customer's usual (no earlier spending to compare)"
    return f["ratio"] >= 5, (f"amount at least 5x the customer's usual: {_money(f['amount'])} is "
                             f"{f['ratio']:.2f}x their median {_money(f['median'])}")


def _late_night(f):
    return f["hour"] in LATE_HOURS, f"late night (22:00-03:59): the purchase was at {f['time']:%H:%M}"


def _new_merchant(f):
    if f["first_time"] is None:
        return None, "merchant is new to the customer (merchant history unavailable)"
    if f["first_time"]:
        return True, f"merchant is new to the customer: no earlier visits to {f['merchant']}"
    return False, f"merchant is new to the customer: {f['merchant_visits']} earlier visits to {f['merchant']}"


def _rarely_late(f):
    if "late_share" not in f:
        return None, "customer rarely spends late at night (no transaction history)"
    return f["late_share"] < 0.10, (f"customer rarely spends late at night (under 10%): "
                                    f"{f['late_share']:.0%} of their last {f['history_n']} transactions were late")


def _often_late(f):
    if "late_share" not in f:
        return None, "customer often spends late at night (no transaction history)"
    return f["late_share"] >= 0.30, (f"customer often spends late at night (30% or more): "
                                     f"{f['late_share']:.0%} of their last {f['history_n']} transactions were late")


def _largest_ever(f):
    if f["max_before"] is None:
        return None, "largest purchase this customer has made (no earlier spending)"
    return f["amount"] > f["max_before"], (f"largest purchase this customer has made: earlier maximum "
                                           f"{_money(f['max_before'])}")


def _large_purchases_routine(f):
    if f["p95"] is None:
        return None, "large purchases are routine for this customer (no earlier spending)"
    return f["p95"] >= f["amount"] / 2, (f"large purchases are routine for this customer (95th percentile at "
                                         f"least half this amount): their 95th percentile is {_money(f['p95'])}")


def _regular_merchant_similar(f):
    visits, top = f["merchant_visits"], f["merchant_max"]
    if f["first_time"] is None:
        return None, "regular at this merchant with similar amounts (merchant history unavailable)"
    if not visits:
        return False, "regular at this merchant with similar amounts: no earlier visits"
    present = visits >= 3 and top is not None and top >= f["amount"] / 2
    return present, (f"regular at this merchant with similar amounts (3+ visits, earlier maximum at least half "
                     f"this amount): {visits} visits, earlier maximum {_money(top)}")


def _new_merchants_24h(f):
    if f.get("new_merchants_24h") is None:
        return None, "2 or more new merchants in the previous 24 hours (not enough history)"
    return f["new_merchants_24h"] >= 2, (f"2 or more new merchants in the previous 24 hours: "
                                         f"{f['new_merchants_24h']} of {f['merchants_24h']} merchants were new "
                                         f"(not seen in the {f['older_days']} days of earlier history available)")


def _spend_24h_3x_daily(f):
    if f.get("daily_spend") is None:
        return None, "spending in the previous 24 hours at least 3x the usual daily spend (not enough history)"
    present = f["daily_spend"] > 0 and f["spend_24h"] >= 3 * f["daily_spend"]
    return present, (f"spending in the previous 24 hours at least 3x the usual daily spend: "
                     f"{_money(f['spend_24h'])} over {f['n_24h']} transactions vs a usual "
                     f"{_money(f['daily_spend'])} a day")


def _many_categories_24h(f):
    if "categories_24h" not in f:
        return None, "3 or more categories in the previous 24 hours (no transaction history)"
    return f["categories_24h"] >= 3, f"3 or more categories in the previous 24 hours: {f['categories_24h']}"


def _burst_3h(f):
    if "n_3h" not in f:
        return None, "2 or more transactions in the previous 3 hours (no transaction history)"
    return f["n_3h"] >= 2, f"2 or more transactions in the previous 3 hours: {f['n_3h']}"


def _had_days_this_big(f):
    if f.get("max_day_spend") is None:
        return None, "customer has had a day of spending this large before (not enough history)"
    total = f["spend_24h"] + f["amount"]
    return f["max_day_spend"] >= total, (f"customer has had a day of spending this large before: biggest earlier "
                                         f"day {_money(f['max_day_spend'])} vs {_money(total)} in the last 24 "
                                         f"hours including this purchase")


def _recent_merchants_familiar(f):
    if f.get("new_merchants_24h") is None or not f.get("merchants_24h"):
        return None, "most merchants in the previous 24 hours are familiar (not enough history)"
    familiar = f["merchants_24h"] - f["new_merchants_24h"]
    return familiar > f["merchants_24h"] / 2, (f"most merchants in the previous 24 hours are familiar: "
                                               f"{familiar} of {f['merchants_24h']} seen in the "
                                               f"{f['older_days']} days of earlier history available")


def _online_category(f):
    return f["category"].endswith("_net"), f"online category: {f['category']}"


def _rarely_online(f):
    if "online_share" not in f:
        return None, "customer rarely shops online (no transaction history)"
    return f["online_share"] < 0.10, (f"customer rarely shops online (under 10%): {f['online_share']:.0%} of "
                                      f"their last {f['history_n']} transactions were online")


def _online_similar_before(f):
    if "online_share" not in f:
        return None, "customer has made online purchases of a similar size (no transaction history)"
    top = f.get("online_max")
    if top is None:
        return False, "customer has made online purchases of a similar size: no earlier online purchases"
    return top >= f["amount"] / 2, (f"customer has made online purchases of a similar size (at least half this "
                                    f"amount): largest earlier online purchase {_money(top)}")


def _pair_detail(f):
    p = f["pair"]
    return (f"{_money(p['small']['amount'])} at {p['small']['merchant']} and {_money(p['large']['amount'])} at "
            f"{p['large']['merchant']}, {p['gap_hours']:.1f} hours apart")


def _small_large_pair(f):
    if "pair" not in f:
        if not f["history_n"]:
            return None, "a charge under $20 within 6 hours of a charge over $200 (no transaction history)"
        return False, "a charge under $20 within 6 hours of a charge over $200, involving this purchase: none"
    return True, "a charge under $20 within 6 hours of a charge over $200, involving this purchase: " + _pair_detail(f)


def _pair_flag(key, text):
    def signal(f):
        if "pair" not in f:
            return None, f"{text} (no small/large pair)"
        value = f["pair"][key]
        if value is None:
            return None, f"{text} (merchant history unavailable)"
        return bool(value), f"{text}: {'yes' if value else 'no'}"
    return signal


def _pair_small_merchant_regular(f):
    if "pair" not in f:
        return None, "the small charge's merchant is a regular one (no small/large pair)"
    visits = f["pair"]["small_merchant_visits"]
    return visits >= 3, f"the small charge's merchant is a regular one (3+ earlier visits): {visits} earlier visits"


SIGNALS = {
    "amount_5x_usual": _amount_5x_usual,
    "late_night": _late_night,
    "new_merchant": _new_merchant,
    "rarely_late": _rarely_late,
    "often_late": _often_late,
    "largest_ever": _largest_ever,
    "large_purchases_routine": _large_purchases_routine,
    "regular_merchant_similar_amounts": _regular_merchant_similar,
    "new_merchants_24h": _new_merchants_24h,
    "spend_24h_3x_daily": _spend_24h_3x_daily,
    "many_categories_24h": _many_categories_24h,
    "burst_3h": _burst_3h,
    "had_days_this_big": _had_days_this_big,
    "recent_merchants_familiar": _recent_merchants_familiar,
    "online_category": _online_category,
    "rarely_online": _rarely_online,
    "online_similar_before": _online_similar_before,
    "small_large_pair": _small_large_pair,
    "pair_small_new_merchant": _pair_flag("small_new", "the small charge was at a new merchant"),
    "pair_large_new_merchant": _pair_flag("large_new", "the large charge was at a new merchant"),
    "pair_small_merchant_regular": _pair_small_merchant_regular,
}


# ---------------------------------------------------------------------------
# Matching and rendering
# ---------------------------------------------------------------------------

def _check(names, facts):
    results = []
    for name in names:
        present, detail = SIGNALS[name](facts)
        results.append({"signal": name, "present": present, "detail": detail})
    return results


def _strength(matched, score):
    if not matched:
        return "no match"
    return "strong" if score >= 2 else "moderate" if score == 1 else "weak"


def match_tactics(case, evidence, history_rows=None):
    """Tick every tactic's checklist. Returns all tactics, matched ones first, best first."""
    facts = case_facts(case, evidence, history_rows)
    matches = []
    for tactic in TACTICS:
        core = _check(tactic.core, facts)
        supporting = _check(tactic.supporting, facts)
        against = _check(tactic.against, facts)
        matched = all(item["present"] is True for item in core)
        score = sum(i["present"] is True for i in supporting) - sum(i["present"] is True for i in against)
        matches.append(TacticMatch(tactic.document_id, tactic.title, matched, _strength(matched, score),
                                   score, core, supporting, against))
    matched_ids = {m.document_id for m in matches if m.matched}
    for match in matches:
        partner = TACTICS_BY_ID[match.document_id].strong_only_with
        if match.strength == "strong" and partner and partner not in matched_ids:
            match.score, match.strength = 1, "moderate"
            match.capped = f"capped at moderate: only strong together with {TACTICS_BY_ID[partner].title}"
    order = {t.document_id: i for i, t in enumerate(TACTICS)}
    return sorted(matches, key=lambda m: (not m.matched, -m.score, order[m.document_id]))


def _mark(present):
    return "[yes]" if present is True else "[no]" if present is False else "[unknown]"


def shown_tactics(matches, top_k=3):
    """What the model is shown: the best match in full, plus up to top_k - 1 other matches
    that are moderate or strong, as one line each. Weak secondary matches are dropped: in a
    100-case validation run they were the most common second match and the model rarely used
    them, while two strong matches agreeing (e.g. spree + small charges) is real evidence."""
    matched = [m for m in matches if m.matched]
    if not matched:
        return None, []
    others = [m for m in matched[1:] if m.strength in ("strong", "moderate")][: max(0, top_k - 1)]
    return matched[0], others


def render_tactic_check(matches, top_k=3):
    """The guidance message the model reads."""
    best, others = shown_tactics(matches, top_k)
    lines = [
        "Tactic check. Code compared this case with known fraud tactics, using only the tool "
        "results above. [yes] = the sign is present, [no] = absent, [unknown] = data not "
        "available. These marks are computed facts; do not contradict them.",
    ]
    if best is None:
        lines += ["", "No known tactic matched: no tactic had all of its core signs present."]
    else:
        tactic = TACTICS_BY_ID[best.document_id]
        n_support = sum(i["present"] is True for i in best.supporting)
        n_against = sum(i["present"] is True for i in best.against)
        lines += [
            "",
            f"Best-matching tactic: {tactic.title} - {best.strength} match "
            f"({n_support} supporting sign(s), {n_against} innocent explanation(s))"
            + (f"; {best.capped}" if best.capped else ""),
            tactic.text,
            "Core signs (all present, which is why this tactic matched):",
            *[f"  {_mark(i['present'])} {i['detail']}" for i in best.core],
            "Supporting signs:",
            *[f"  {_mark(i['present'])} {i['detail']}" for i in best.supporting],
            "Innocent explanations:",
            *[f"  {_mark(i['present'])} {i['detail']}" for i in best.against],
        ]
        for other in others:
            core = "; ".join(i["detail"] for i in other.core)
            lines += ["", f"Also matched: {other.title} - {other.strength} match. Core signs: {core}."]
    lines += ["", LOOKALIKES]
    return "\n".join(lines)
