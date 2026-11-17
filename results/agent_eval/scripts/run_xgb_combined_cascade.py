"""Evaluate XGBoost alone against the full XGBoost + combined-agent cascade.

The agent reviews only transactions in flagged_{split}.csv. The final cascade
metrics are nevertheless calculated over every transaction in scores_{split}.csv:
transactions not flagged by XGBoost are automatically treated as legitimate.

Example:
    python results/agent_eval/scripts/run_xgb_combined_cascade.py \
        --split val --provider ollama --model qwen2.5:3b-8k
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from agent.config import PROCESSED_DIR, RUNS_DIR
from agent.data import TransactionStore
from agent.llm import ALL_TOOLS, OpenAICompatibleBackend
from agent.pipeline import load_thresholds, run
from agent.rag_corpus import load_corpus
from agent.reviewers import CombinedReviewer
from agent.retriever import EmbeddingRetriever


ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"


def _git_commit():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _counts(name, truth, prediction, human_review=0, errors=0):
    fraud = truth == 1
    legitimate = truth == 0
    human = pd.isna(prediction)
    caught = int((fraud & (prediction == True)).sum())
    false_alarms = int((legitimate & (prediction == True)).sum())
    cleared = int((legitimate & (prediction == False)).sum())
    missed = int((fraud & (prediction == False)).sum())
    total_fraud = int(fraud.sum())
    return {
        "system": name,
        "cases": int(len(truth)),
        "fraud": total_fraud,
        "fraud_caught": caught,
        "fraud_sent_to_human": int((fraud & human).sum()),
        "fraud_missed": missed,
        "legitimate_cleared": cleared,
        "false_alarms": false_alarms,
        "human_review": int(human.sum()),
        "errors": int(errors),
        "correct_automatic": caught + cleared,
        "precision": caught / (caught + false_alarms) if caught + false_alarms else None,
        "fraud_recall": caught / total_fraud if total_fraud else None,
        "review_rate": int(human.sum()) / len(truth) if len(truth) else None,
    }


def evaluate_cascade(split, run_path, sample=None):
    thresholds = load_thresholds()
    scores = pd.read_csv(PROCESSED_DIR / f"scores_{split}.csv")
    records = [json.loads(line) for line in Path(run_path).open(encoding="utf-8")]
    by_case = {record["case_id"]: record for record in records}
    flagged = scores["xgb_score"] >= thresholds["flag"]
    flagged_ids = set(scores.loc[flagged, "trans_num"])
    reviewed_ids = set(by_case)

    if sample is None:
        missing = flagged_ids - reviewed_ids
        if missing:
            raise RuntimeError(
                f"Run is incomplete: {len(missing):,} flagged cases were not reviewed. "
                "Resume the run or provide --sample for a partial evaluation."
            )

    truth = scores["is_fraud"].astype(int)
    xgb_prediction = scores["xgb_score"] >= thresholds["standalone"]
    xgb_row = _counts("XGBoost alone", truth, xgb_prediction)

    cascade_prediction = pd.Series(False, index=scores.index, dtype=object)
    cascade_errors = pd.Series(False, index=scores.index)
    human_review = 0
    errors = 0
    for index, row in scores.loc[flagged].iterrows():
        record = by_case.get(row["trans_num"])
        if record is None:
            cascade_prediction.at[index] = None
            human_review += 1
            continue
        decision = record.get("decision")
        if record.get("error") or not decision:
            cascade_prediction.at[index] = None
            cascade_errors.at[index] = True
            errors += 1
        elif decision["decision"] == "fraud":
            cascade_prediction.at[index] = True
        elif decision["decision"] == "legitimate":
            cascade_prediction.at[index] = False
        else:
            cascade_prediction.at[index] = None
        if cascade_prediction.at[index] is None:
            human_review += 1

    if sample is not None:
        reviewed = scores["trans_num"].isin(reviewed_ids)
        truth = truth[reviewed]
        cascade_prediction = cascade_prediction[reviewed]
        cascade_errors = cascade_errors[reviewed]
        xgb_row = _counts("XGBoost alone (same reviewed sample)", truth, xgb_prediction[reviewed])

    human_review = int(cascade_prediction.isna().sum())
    errors = int(cascade_errors.sum())
    cascade_row = _counts(
        "XGBoost + combined agent",
        truth,
        cascade_prediction,
        human_review=human_review,
        errors=errors,
    )
    return {
        "split": split,
        "thresholds": thresholds,
        "population": "reviewed sample" if sample is not None else "full validation/test split",
        "reviewed_records": len(records),
        "metrics": [xgb_row, cascade_row],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--provider", required=True, help="LLM provider, for example ollama")
    parser.add_argument("--model", required=True, help="LLM model name")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--corpus", help="RAG corpus directory or documents.jsonl")
    parser.add_argument("--sample", type=int, help="development-only sample of flagged cases")
    parser.add_argument("--seed", type=int, default=4102)
    parser.add_argument("--name", default="combined-cascade")
    parser.add_argument("--max-turns", type=int, default=8)
    args = parser.parse_args()

    retriever = EmbeddingRetriever(load_corpus(args.corpus), model=args.embedding_model)
    reviewer = CombinedReviewer(
        OpenAICompatibleBackend(args.provider, args.model),
        retriever=retriever,
        allowed_tools=ALL_TOOLS,
        max_turns=args.max_turns,
    )
    run_path = run(
        args.split,
        reviewer,
        sample=args.sample,
        seed=args.seed,
        run_name=args.name,
        store=TransactionStore.load(),
    )
    report = evaluate_cascade(args.split, run_path, sample=args.sample)
    report["run_file"] = str(run_path)
    report["configuration"] = {
        "git_commit": _git_commit(),
        "provider": args.provider,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "sample": args.sample,
        "seed": args.seed,
        "max_turns": args.max_turns,
        "reviewer": "combined tactic checklist + embedding RAG",
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"{args.name}_{args.split}_cascade.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    main()
