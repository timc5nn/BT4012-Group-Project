"""Score the committed agent runs against validation labels and XGBoost alone on the same cases.

Labels are read from data/processed/scores_val.csv, only here, after the runs. Run from the
repo root after the Stage 1 notebook has produced data/processed/:

    python results/agent_eval/scripts/summarise_runs.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd  # noqa: E402

from agent.pipeline import load_thresholds  # noqa: E402

RUNS = Path(__file__).resolve().parents[1] / "runs"
NAMES = ["pilot-qwen-rag", "pilot-qwen8k-alltools", "pilot-qwen8k-tactics",
         "eval100-qwen8k-tactics-v2", "eval100-qwen8k-tactics-v3"]

scores = pd.read_csv("data/processed/scores_val.csv").set_index("trans_num")
standalone = load_thresholds()["standalone"]


def score(name):
    records = [json.loads(line) for line in (RUNS / f"{name}_val.jsonl").open()]
    df = pd.DataFrame({
        "fraud": [int(scores.is_fraud[r["case_id"]]) for r in records],
        "decision": [(r["decision"] or {}).get("decision", "ERROR") for r in records],
        "xgb": [scores.xgb_score[r["case_id"]] for r in records],
        "seconds": [r["seconds"] for r in records],
        "tokens": [(r["usage"] or {}).get("input_tokens", 0) for r in records],
    })
    F, L = df[df.fraud == 1], df[df.fraud == 0]
    agent = {
        "cases (fraud/legit)": f"{len(F)}/{len(L)}",
        "fraud caught": (F.decision == "fraud").sum(),
        "fraud -> review": (F.decision == "needs_human_review").sum(),
        "fraud cleared": (F.decision == "legitimate").sum(),
        "legit cleared": (L.decision == "legitimate").sum(),
        "legit -> review": (L.decision == "needs_human_review").sum(),
        "false alarms": (L.decision == "fraud").sum(),
        "errors": (df.decision == "ERROR").sum(),
        "correct": (F.decision == "fraud").sum() + (L.decision == "legitimate").sum(),
        "s / case": round(df.seconds.mean(), 1),
        "input tokens / case": round(df.tokens.mean()),
    }
    xgb_fraud = df.xgb >= standalone
    xgb = {"xgb correct": int((xgb_fraud == df.fraud.astype(bool)).sum()),
           "xgb fraud caught": int((xgb_fraud & (df.fraud == 1)).sum()),
           "xgb false alarms": int((xgb_fraud & (df.fraud == 0)).sum())}
    return {**agent, **xgb}


table = pd.DataFrame({name: score(name) for name in NAMES})
print(table.to_string())
