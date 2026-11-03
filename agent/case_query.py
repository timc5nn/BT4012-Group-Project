"""Decide whether a chat message names one flagged transaction.

The model never chooses the case. A message opens an investigation only when
the details in it match exactly one row the scoring model already flagged.
Anything looser stays a question, or comes back as a short list to choose from.
"""

import re

import pandas as pd

MAX_CHOICES = 5

_TRANS_NUM = re.compile(r"\b[0-9a-f]{32}\b", re.I)
_CUSTOMER = re.compile(r"\bKYC-\d{5}\b", re.I)
_EXACT_AMOUNT = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*|\d+)\.(\d{2})\b|(?<![\d.])(\d{1,5})\.(\d{2})(?!\d)")
_ROUNDED_AMOUNT = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*|\d+)(?![\d.])|(\d{1,5})\s+dollars?\b", re.I)
_AMOUNT_KEY = re.compile(r'"amount"\s*:\s*(\d+(?:\.\d+)?)', re.I)
_ISO_WHEN = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2}))?")
_MONTH_WHEN = re.compile(
    r"\b(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(20\d{2})(?:[, ]+(\d{1,2}):(\d{2}))?",
    re.I,
)
_SCORE = re.compile(r"(?:xgb_)?score\s*[\"']?\s*[:=]?\s*(0\.\d{2,4})\b|·\s*(0\.\d{2,4})\b", re.I)
_OPEN = re.compile(
    r"\b(investigate|review|reopen|examine)\b|\bpull up\b|\bopen (?:the |this )?(?:case|transaction|flag)\b",
    re.I,
)
_RERUN = re.compile(r"\b(again|rerun|re-run|reinvestigate|re-investigate)\b", re.I)

_MONTHS = {name: number for number, name in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1
)}


def route_message(text, cases, active_id=None, pending=None):
    """Return investigate, clarify, follow_up, or general.

    `cases` is the flagged table, one row per transaction, with customer_id filled in.
    `active_id` is the open case's trans_num, if a review is already on screen.
    `pending` is the row positions from the last numbered list, when the user still
    has to choose one.
    """
    cases = cases.reset_index(drop=True)
    if pending:
        choice = _resolve_choice(text, pending, cases)
        if choice == "out_of_range":
            return _clarify(f"Reply with a number from 1 to {len(pending)}.", pending)
        if choice is not None:
            return _investigate(choice)
    found = _extract(text, cases["merchant"].unique())
    wants_open = _OPEN.search(text) is not None
    rerun = _RERUN.search(text) is not None
    mask = _constrain(cases, found)
    hits = cases[mask]
    specific = found["specific"]
    hints = found["hints"]
    case_attempt = bool(specific) or len(hints) >= 2 or (wants_open and bool(hints))

    if found["missing_customer"] and not any(cid in set(cases["customer_id"].str.upper()) for cid in found["customer_ids"]):
        return _clarify(f"{found['missing_customer']} has no flagged transaction to review.")
    if found["missing_trans_num"]:
        return _clarify("That transaction id is not among the flagged cases.")

    if not case_attempt:
        if rerun and active_id is not None:
            return _investigate(cases.index[cases["trans_num"] == active_id][0])
        if wants_open and active_id is None:
            return _clarify(
                "Name one flagged transaction: a customer id such as KYC-00042, "
                "an amount with cents such as $19.78, or a merchant together with an amount."
            )
        if active_id is not None:
            return {"action": "follow_up"}
        return {"action": "general"}

    if hits.empty:
        return _clarify("No flagged transaction matches those details. Add the customer id, or the amount in cents and the merchant.")

    if len(hits) == 1:
        row = hits.iloc[0]
        if active_id is not None and row.trans_num == active_id and not rerun:
            return {"action": "follow_up"}
        return _investigate(int(hits.index[0]))

    if active_id is not None and active_id in set(hits["trans_num"]) and not wants_open and not rerun:
        return {"action": "follow_up"}
    return _choices(hits, found)


def _investigate(index):
    return {"action": "investigate", "row": int(index)}


def _clarify(text, choices=None):
    result = {"action": "clarify", "text": text}
    if choices:
        result["choices"] = [int(index) for index in choices]
    return result


def _extract(text, merchants):
    customer_ids = [match.group(0).upper() for match in _CUSTOMER.finditer(text)]
    trans_nums = [match.group(0).lower() for match in _TRANS_NUM.finditer(text)]
    exact, rounded = _amounts(text)
    categories = _categories(text)
    scores = [match.group(1) or match.group(2) for match in _SCORE.finditer(text)]
    specific = set()
    hints = set()
    if customer_ids:
        specific.add("customer_id")
    if trans_nums:
        specific.add("trans_num")
    if exact:
        specific.add("amount")
    if _merchant_hits(text, merchants):
        specific.add("merchant")
    if _whens(text):
        specific.add("when")
    if categories:
        hints.add("category")
    if rounded:
        hints.add("dollars")
    if scores:
        hints.add("score")
    return {
        "customer_ids": customer_ids,
        "trans_nums": trans_nums,
        "exact": exact,
        "rounded": rounded,
        "merchants": _merchant_hits(text, merchants),
        "categories": categories,
        "whens": _whens(text),
        "scores": scores,
        "specific": specific,
        "hints": hints,
        "missing_customer": None,
        "missing_trans_num": None,
    }


def _constrain(cases, found):
    """Keep rows consistent with every detail the message actually gave."""
    mask = pd.Series(True, index=cases.index)
    if found["trans_nums"]:
        known = set(cases["trans_num"].str.lower())
        present = [item for item in found["trans_nums"] if item in known]
        if not present:
            found["missing_trans_num"] = found["trans_nums"][0]
            return pd.Series(False, index=cases.index)
        mask &= cases["trans_num"].str.lower().isin(present)
    if found["customer_ids"]:
        known = set(cases["customer_id"].str.upper())
        present = [item for item in found["customer_ids"] if item in known]
        if not present:
            found["missing_customer"] = found["customer_ids"][0]
            return pd.Series(False, index=cases.index)
        mask &= cases["customer_id"].str.upper().isin(present)
    if found["exact"]:
        amounts = cases["amt"].round(2)
        mask &= amounts.isin(found["exact"])
    elif found["rounded"]:
        shown = cases["amt"].map(lambda amount: int(f"{amount:.0f}"))
        mask &= shown.isin(found["rounded"])
    if found["merchants"]:
        mask &= cases["merchant"].isin(found["merchants"])
    if found["categories"]:
        mask &= cases["category"].isin(found["categories"])
    if found["whens"]:
        when = pd.to_datetime(cases["trans_date_trans_time"])
        keep = pd.Series(False, index=cases.index)
        for stamp, precise in found["whens"]:
            if precise:
                keep |= when.dt.floor("min") == stamp
            else:
                keep |= when.dt.normalize() == stamp
        mask &= keep
    if found["scores"] and (found["specific"] or len(found["hints"]) >= 2):
        keep = pd.Series(False, index=cases.index)
        for raw in found["scores"]:
            places = min(len(raw.split(".")[1]), 4)
            keep |= cases["xgb_score"].round(places) == float(raw)
        mask &= keep
    return mask


def _amounts(text):
    exact = []
    rounded = []
    consumed = []
    for match in _AMOUNT_KEY.finditer(text):
        consumed.append(match.span())
        value = float(match.group(1))
        if "." in match.group(1):
            exact.append(round(value, 2))
        else:
            rounded.append(int(value))
    for match in _EXACT_AMOUNT.finditer(text):
        if _overlaps(match.span(), consumed) or _is_score(text, match.start()):
            continue
        whole, cents = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
        exact.append(round(float(whole.replace(",", "")) + int(cents) / 100, 2))
    if exact:
        return exact, []
    for match in _ROUNDED_AMOUNT.finditer(text):
        if _overlaps(match.span(), consumed):
            continue
        raw = (match.group(1) or match.group(2)).replace(",", "")
        rounded.append(int(raw))
    return exact, rounded


def _is_score(text, start):
    prefix = text[max(0, start - 16):start].lower()
    return "score" in prefix or prefix.rstrip().endswith("·")


def _overlaps(span, spans):
    return any(span[0] < end and span[1] > start for start, end in spans)


def _merchant_hits(text, merchants):
    folded = _fold(text)
    found = [name for name in merchants if _fold(name) and _fold(name) in folded]
    found.sort(key=lambda name: len(_fold(name)), reverse=True)
    kept = []
    for name in found:
        folded_name = _fold(name)
        if any(folded_name in _fold(longer) and folded_name != _fold(longer) for longer in kept):
            continue
        kept.append(name)
    return kept


def _fold(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _categories(text):
    folded = _fold(text)
    found = []
    names = [
        "entertainment", "food_dining", "gas_transport", "grocery_net", "grocery_pos",
        "health_fitness", "home", "kids_pets", "misc_net", "misc_pos", "personal_care",
        "shopping_net", "shopping_pos", "travel",
    ]
    for name in sorted(names, key=len, reverse=True):
        phrase = name.replace("_", " ")
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", folded):
            found.append(name)
    return found


def _whens(text):
    found = []
    for year, month, day, hour, minute in _ISO_WHEN.findall(text):
        found.append(_stamp(year, month, day, hour, minute))
    for day, month, year, hour, minute in _MONTH_WHEN.findall(text):
        found.append(_stamp(year, _MONTHS[month[:3].lower()], day, hour, minute))
    return found


def _stamp(year, month, day, hour, minute):
    stamp = pd.Timestamp(int(year), int(month), int(day))
    if hour == "" and minute == "":
        return stamp, False
    return stamp + pd.Timedelta(hours=int(hour), minutes=int(minute or 0)), True


def _resolve_choice(text, pending, cases):
    """Map a short reply onto one row from the list just shown."""
    number = _selection_number(text, len(pending))
    if number is not None:
        if 1 <= number <= len(pending):
            return int(pending[number - 1])
        return "out_of_range"
    if _CUSTOMER.search(text) or _TRANS_NUM.search(text) or _OPEN.search(text):
        return None
    subset = cases.loc[list(pending)]
    found = _extract(text, subset["merchant"].unique())
    if not found["specific"] and not found["hints"]:
        return None
    hits = subset[_constrain(subset, found)]
    if len(hits) == 1:
        return int(hits.index[0])
    return None


def _selection_number(text, count):
    stripped = text.strip()
    matched = re.fullmatch(r"(?:option\s*|number\s*|#)?(\d{1,2})\.?", stripped, re.I)
    if matched:
        return int(matched.group(1))
    if re.search(r"\b(why|how|what|did|does|do|is|are|was|were|can|could|should)\b", stripped, re.I):
        return None
    words = re.findall(r"\b(first|second|third|fourth|fifth|last|1st|2nd|3rd|4th|5th)\b", stripped, re.I)
    if len(words) == 1 and len(stripped) <= 48:
        number = {
            "first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
            "fourth": 4, "4th": 4, "fifth": 5, "5th": 5, "last": count,
        }[words[0].lower()]
        return number
    if len(stripped) <= 40 and "$" not in stripped and _CUSTOMER.search(stripped) is None:
        numbers = re.findall(r"\b(\d{1,2})\b", stripped)
        if len(numbers) == 1:
            return int(numbers[0])
    return None


def _choices(hits, found):
    ranked = hits
    if found["scores"]:
        target = float(found["scores"][0])
        ranked = ranked.assign(_distance=(ranked["xgb_score"] - target).abs())
        ranked = ranked.sort_values(["_distance", "trans_date_trans_time"])
    elif found["exact"] or found["rounded"]:
        target = found["exact"][0] if found["exact"] else found["rounded"][0]
        ranked = ranked.assign(_distance=(ranked["amt"] - target).abs())
        ranked = ranked.sort_values(["_distance", "trans_date_trans_time"])
    else:
        ranked = ranked.sort_values("trans_date_trans_time")
    one_customer = ranked["customer_id"].str.upper().nunique() == 1 and found["customer_ids"]
    cap = 12 if one_customer else MAX_CHOICES
    shown = ranked.head(cap)
    if one_customer and len(ranked) > len(shown):
        lead = (
            f"{found['customer_ids'][0]} has {len(ranked)} flagged transactions. "
            f"Reply with a number for one of these {len(shown)}, or name the amount and merchant."
        )
    elif one_customer:
        lead = f"{found['customer_ids'][0]} has {len(ranked)} flagged transactions. Which one should I investigate?"
    elif len(ranked) > len(shown):
        lead = f"That matches {len(ranked)} flagged transactions. Reply with a number for one of these, or name the merchant and amount."
    else:
        lead = "Which transaction should I investigate?"
    lines = [
        f"{number}. {_format_choice(row)}"
        for number, (_, row) in enumerate(shown.iterrows(), start=1)
    ]
    return _clarify(lead + "\n" + "\n".join(lines) + "\nReply with a number.", list(shown.index))


def _format_choice(row):
    when = pd.Timestamp(row.trans_date_trans_time).strftime("%d %b %Y, %H:%M")
    category = str(row.category).replace("_", " ")
    return f"${row.amt:,.2f} · {category} · {row.merchant} · {when} · {row.customer_id}"
