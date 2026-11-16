"""Run the tactic checklists (agent/tactics.py) on FIT-row transactions; no LLM, no val/test data.

Samples 2,000 fraud and 2,000 legitimate transactions from the first 80% of fraudTrain.csv,
runs the case tools point-in-time, ticks every checklist, and reports how match strength
relates to fraud. Used to justify checklist changes without looking at validation labels.
Run from the repo root (about 30 seconds):

    python results/agent_eval/scripts/fit_tactic_matcher.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent.data import TransactionStore  # noqa: E402
from agent.schemas import Case  # noqa: E402
from agent.tactics import TACTIC_TOOLS, match_tactics  # noqa: E402
from agent.tools import CaseTools  # noqa: E402

cols = ["trans_num", "trans_date_trans_time", "cc_num", "merchant", "category", "amt", "is_fraud"]
df = pd.read_csv("data/fraudTrain.csv", usecols=cols)
fit = df.iloc[: int(len(df) * 0.8)]
sample = pd.concat([fit[fit.is_fraud == 1].sample(2000, random_state=4102),
                    fit[fit.is_fraud == 0].sample(2000, random_state=4102)])
store = TransactionStore.load()
rows = []
for r in sample.itertuples(index=False):
    t = pd.Timestamp(r.trans_date_trans_time)
    case = Case(r.trans_num, store.customer_id(r.cc_num), t, r.amt, r.merchant.removeprefix("fraud_"), r.category, 0.0)
    tools = CaseTools(store, r.cc_num, t)
    evidence = {n: tools.call(n, a) for n, a in TACTIC_TOOLS}
    hist = evidence["get_transaction_history"].get("transactions", [])
    span = (t - pd.Timestamp(hist[0]["time"])) / pd.Timedelta(days=1) if hist else np.nan
    matches = match_tactics(case, evidence)
    top = [m for m in matches if m.matched]
    rows.append({"fraud": r.is_fraud, "history_rows": len(hist), "span_days": span,
                 "top_strength": top[0].strength if top else "no match",
                 **{m.document_id: m.strength for m in matches}})
res = pd.DataFrame(rows)

full = res[res.history_rows == 100]
print(f"{len(res)} fit-row transactions (50% fraud)")
print(f"days covered by 100 earlier rows: median {full.span_days.median():.0f}, under 30 days for "
      f"{(full.span_days < 30).mean():.0%}")
order = ["strong", "moderate", "weak", "no match"]
print("\nEach tactic: count (share fraud) by strength")
for tid in [c for c in res.columns if c.startswith("tactic_")]:
    g = res.groupby(tid).fraud.agg(["size", "mean"]).reindex(order)
    print(f"  {tid:38} " + "  ".join(f"{s}: {int(n)} ({m:.0%})" for s, (n, m) in g.dropna().iterrows()))
print("\nBest match strength:")
print(res.groupby("top_strength").fraud.agg(["size", "mean"]).reindex(order).round(3).to_string())

S, P = "tactic_stolen_card_spree", "tactic_small_charges_in_spree"
res["short_history"] = res.span_days < 30
print("\nSpree matches, share fraud by history coverage under 30 days:")
print(res[res[S] != "no match"].groupby("short_history").fraud.agg(["size", "mean"]).round(3).to_string())
small = res[res[P] != "no match"].assign(spree_too=lambda d: d[S] != "no match")
print("\nSmall-charges matches, split by whether a spree also matched:")
print(small.groupby(["spree_too", P]).fraud.agg(["size", "mean"]).round(3).to_string())
