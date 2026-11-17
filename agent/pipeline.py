"""Run a reviewer over XGBoost's flagged cases, then evaluate the combined system.

Inputs (written by xgboost.ipynb):
    data/processed/flagged_{split}.csv   cases at or above the flagging threshold, no labels
    data/processed/scores_{split}.csv    every transaction's score and label, for evaluation only
    data/processed/xgb_thresholds.json   the standalone and flagging thresholds

Usage:
    python -m agent.pipeline run --split val --sample 300                     # no-LLM baseline
    python -m agent.pipeline run --split val --sample 300 --reviewer llm --provider openrouter --model <model>
    python -m agent.pipeline evaluate --split val --run data/agent_runs/<run>_val.jsonl

Runs append one JSON line per case and skip cases already in the file, so an
interrupted run can be resumed by running the same command again.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from .config import PROCESSED_DIR, RUNS_DIR, THRESHOLDS_FILE
from .data import TransactionStore
from .reviewers import CombinedReviewer, RAGReviewer, ScoreThresholdReviewer, TacticReviewer
from .schemas import Case
from .tools import CaseTools


def load_thresholds():
    return json.loads(THRESHOLDS_FILE.read_text())


def load_flagged(split):
    flagged = pd.read_csv(PROCESSED_DIR / f"flagged_{split}.csv", parse_dates=["trans_date_trans_time"])
    assert "is_fraud" not in flagged.columns, "flagged cases must not carry labels"
    return flagged


def run(split, reviewer, sample=None, seed=4102, run_name=None, store=None, progress_every=100):
    """Review the flagged cases of `split` (a random sample of `sample` cases if given). Returns the run file."""
    store = store or TransactionStore.load()
    flagged = load_flagged(split)
    if sample is not None and sample < len(flagged):
        flagged = flagged.sample(n=sample, random_state=seed)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    path = RUNS_DIR / f"{run_name or reviewer.name}_{split}.jsonl"
    done = {json.loads(line)["case_id"] for line in path.open()} if path.exists() else set()
    todo = flagged[~flagged["trans_num"].isin(done)]
    print(f"{split}: {len(flagged):,} cases in this run, {len(done):,} already done, {len(todo):,} to review -> {path}")

    with path.open("a") as out:
        for i, row in enumerate(todo.itertuples(index=False), start=1):
            case = Case(
                case_id=row.trans_num,
                customer_id=store.customer_id(row.cc_num),
                time=row.trans_date_trans_time,
                amount=row.amt,
                merchant=row.merchant,
                category=row.category,
                xgb_score=row.xgb_score,
            )
            tools = CaseTools(store, row.cc_num, row.trans_date_trans_time, xgb_score=row.xgb_score)
            started = time.perf_counter()
            try:
                decision, error = reviewer.review(case, tools).to_dict(), None
            except Exception as exc:  # a failed case is logged and counted, not fatal
                decision, error = None, repr(exc)
            out.write(json.dumps({
                "case_id": case.case_id,
                "split": split,
                "reviewer": reviewer.name,
                "decision": decision,
                "error": error,
                "tool_calls": tools.calls,
                "retrieved_documents": [d.__dict__ for d in getattr(reviewer, "last_retrieval", [])],
                "retrieved_guidance_documents": [
                    d.__dict__ for d in getattr(reviewer, "last_rag_retrieval", [])
                ],
                "retrieval_error": getattr(getattr(reviewer, "retriever", None), "last_error", None),
                "usage": getattr(reviewer, "last_usage", None),  # an LLM reviewer can report tokens/cost here
                "seconds": round(time.perf_counter() - started, 3),
            }) + "\n")
            if i % progress_every == 0:
                out.flush()
                print(f"  {i:,} / {len(todo):,}")
    return path


def _row(name, alerts, frauds_caught, total_frauds):
    false_alarms = alerts - frauds_caught
    return {
        "system": name,
        "alerts": round(alerts),
        "frauds caught": round(frauds_caught),
        "false alarms": round(false_alarms),
        "precision": frauds_caught / alerts if alerts else float("nan"),
        "recall": frauds_caught / total_frauds,
    }


def evaluate(split, run_path):
    """Compare XGBoost alone with XGBoost + the run's reviewer.

    When the run reviewed a sample of the flagged cases, counts for the sample
    are scaled up to the full flagged set, and "XGBoost alone" is also scored on
    the same sample so the two systems are compared like for like.
    """
    thresholds = load_thresholds()
    scores = pd.read_csv(PROCESSED_DIR / f"scores_{split}.csv")
    records = [json.loads(line) for line in Path(run_path).open()]
    reviewed = pd.DataFrame({
        "trans_num": [r["case_id"] for r in records],
        "agent_fraud": [r["decision"] is not None and r["decision"]["decision"] == "fraud" for r in records],
        "error": [r["error"] is not None for r in records],
    }).merge(scores, on="trans_num", how="left", validate="one_to_one")

    total_frauds = int(scores["is_fraud"].sum())
    flagged = scores["xgb_score"] >= thresholds["flag"]
    scale = flagged.sum() / len(reviewed)
    standalone_all = scores["xgb_score"] >= thresholds["standalone"]
    standalone_sample = reviewed["xgb_score"] >= thresholds["standalone"]
    fraud = reviewed["is_fraud"] == 1

    rows = [
        _row("Stage 1 flags (ceiling for any reviewer)", flagged.sum(),
             (flagged & (scores["is_fraud"] == 1)).sum(), total_frauds),
        _row("A. XGBoost alone (all transactions, exact)", standalone_all.sum(),
             (standalone_all & (scores["is_fraud"] == 1)).sum(), total_frauds),
        _row("A. XGBoost alone (reviewed sample, scaled)", standalone_sample.sum() * scale,
             (standalone_sample & fraud).sum() * scale, total_frauds),
        _row(f"B. XGBoost + {records[0]['reviewer']} (scaled)", reviewed["agent_fraud"].sum() * scale,
             (reviewed["agent_fraud"] & fraud).sum() * scale, total_frauds),
    ]
    result = pd.DataFrame(rows).set_index("system")
    print(f"{split}: {len(reviewed):,} cases reviewed of {flagged.sum():,} flagged (scale x{scale:.2f}); "
          f"{reviewed['error'].sum()} errors; {total_frauds:,} frauds in the split")
    usage = [r["usage"] for r in records if r.get("usage")]
    if usage:
        print(f"Per case: {sum(u['input_tokens'] for u in usage) / len(usage):,.0f} input tokens, "
              f"{sum(u['output_tokens'] for u in usage) / len(usage):,.0f} output tokens, "
              f"{sum(u['turns'] for u in usage) / len(usage):.1f} model turns, "
              f"{sum(r['seconds'] for r in records) / len(records):.1f} s")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser("run", help="review flagged cases")
    run_cmd.add_argument("--split", choices=["val", "test"], required=True)
    run_cmd.add_argument("--sample", type=int, help="review a random sample of this many flagged cases")
    run_cmd.add_argument("--seed", type=int, default=4102)
    run_cmd.add_argument("--name", help="run name (default: the reviewer's name)")
    run_cmd.add_argument("--reviewer", choices=["baseline", "llm"], default="baseline")
    run_cmd.add_argument("--rag", action="store_true", help="llm: retrieve synthetic investigation guidance")
    run_cmd.add_argument("--tactics", action="store_true",
                         help="llm: match the case to fraud-tactic documents by checklist (agent/tactics.py)")
    run_cmd.add_argument("--corpus", help="rag: corpus directory or documents.jsonl")
    run_cmd.add_argument("--embedding-model", default="nomic-embed-text",
                         help="rag: Ollama embedding model (default: nomic-embed-text)")
    run_cmd.add_argument("--top-k", type=int, default=3, help="rag: number of documents to retrieve")
    run_cmd.add_argument("--threshold", type=float, help="baseline: score threshold (default: standalone)")
    run_cmd.add_argument("--provider", help="llm: openai, gemini, openrouter or ollama")
    run_cmd.add_argument("--model", help="llm: the provider's model name")
    run_cmd.add_argument("--no-score", action="store_true", help="llm: withhold the XGBoost score (system C)")
    run_cmd.add_argument("--no-kyc", action="store_true", help="llm: remove the KYC tool (system D)")
    run_cmd.add_argument("--max-turns", type=int, default=8, help="llm: model calls per case before giving up")

    eval_cmd = sub.add_parser("evaluate", help="evaluate a finished run")
    eval_cmd.add_argument("--split", choices=["val", "test"], required=True)
    eval_cmd.add_argument("--run", required=True, help="path to the run's .jsonl file")

    args = parser.parse_args()
    if args.command == "run":
        if args.reviewer == "llm":
            if not (args.provider and args.model):
                parser.error("--reviewer llm needs --provider and --model")
            from .llm import ALL_TOOLS, HISTORY_TOOLS, LLMReviewer, OpenAICompatibleBackend
            ReviewerClass = LLMReviewer
            reviewer_kwargs = {}
            if args.rag and args.tactics:
                from .rag_corpus import load_corpus
                from .retriever import EmbeddingRetriever
                ReviewerClass = CombinedReviewer
                reviewer_kwargs = {
                    "retriever": EmbeddingRetriever(
                        load_corpus(args.corpus),
                        model=args.embedding_model,
                    ),
                    "top_k": args.top_k,
                }
            elif args.tactics:
                ReviewerClass = TacticReviewer
                reviewer_kwargs = {"top_k": args.top_k}
            if args.rag:
                from .rag_corpus import load_corpus
                from .retriever import EmbeddingRetriever
                ReviewerClass = RAGReviewer
                reviewer_kwargs = {
                    "retriever": EmbeddingRetriever(
                        load_corpus(args.corpus),
                        model=args.embedding_model,
                    ),
                    "top_k": args.top_k,
                }
            reviewer = ReviewerClass(
                OpenAICompatibleBackend(args.provider, args.model),
                include_score=not args.no_score,
                allowed_tools=HISTORY_TOOLS if args.no_kyc else ALL_TOOLS,
                max_turns=args.max_turns,
                **reviewer_kwargs,
            )
        else:
            threshold = args.threshold if args.threshold is not None else load_thresholds()["standalone"]
            reviewer = ScoreThresholdReviewer(threshold)
        path = run(args.split, reviewer, args.sample, args.seed, args.name)
        print(evaluate(args.split, path).round(3).to_string())
    else:
        print(evaluate(args.split, args.run).round(3).to_string())


if __name__ == "__main__":
    main()
