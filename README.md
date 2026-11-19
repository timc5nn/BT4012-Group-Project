# BT4102-Group-Project

## Data

This project uses the [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (`fraudTrain.csv`, `fraudTest.csv`).

The CSVs are too large for GitHub (143MB / 335MB) so they are **not** committed — `data/` is gitignored. Download the dataset from the link above and place both files in a local `data/` folder before running the notebooks.

The one exception is `data/kyc_customers.csv` (the synthetic KYC table, 171KB), which is committed so that everyone uses identical profiles. You don't need to run `kyc.ipynb`. If you do re-run it, `git status` will show whether your output differs from the shared copy; don't commit a changed version without agreeing it with the group.

The Stage 1 output is committed for the same reason: `data/processed/flagged_val.csv` and `flagged_test.csv` (the flagged cases, without labels), `scores_val.csv` and `scores_test.csv` (every transaction's score and label, read only when a run is evaluated), `xgb_thresholds.json` and `xgb_model.json`, all from the reference Stage 1 run, `xgboost_reference_run.ipynb` (XGBoost, all 110 features, 98% recall target, 513 trees, standalone threshold 0.603, flagging threshold 0.0075, test PR-AUC 0.969), which keeps that run's original outputs. The outputs saved in `xgboost.ipynb` and `xgboost_final.ipynb` come from a re-run of the same setup on another machine (453 trees, thresholds 0.822 and 0.011, test PR-AUC 0.969): XGBoost's multithreaded training is not bit-for-bit reproducible across machines, so the reference run and the committed files are what every agent evaluation and the report use. With these files and the raw Kaggle CSVs you can run and evaluate the agent pipeline without re-running the notebooks. The test scores are for the single final run of the frozen design. If you re-run `xgboost.ipynb`, check with `git status` that the shared files are unchanged, and don't commit a changed version without agreeing it with the group.

## Run order

Each step writes files into `data/` that the next one reads.

| Step | Notebook | Writes |
|---|---|---|
| 1 | `eda.ipynb` | nothing (exploration only) |
| 2 | `kyc.ipynb` | `data/kyc_customers.csv`, a synthetic KYC profile per customer (already committed; optional) |
| 3 | `preprocessing.ipynb` | `data/processed/fraud{Train,Test}_features.csv` |
| 4 | `logistic_regression.ipynb` | nothing (baseline model for the comparison) |
| 5 | `random_forest.ipynb` | nothing (tree model for the comparison) |
| 6 | `xgboost.ipynb` | stage 1: model, thresholds, flagged cases and scores in `data/processed/` |
| 7 | `agent_pipeline.ipynb` | stage 2 demo: runs in `data/agent_runs/` |

Feature definitions are in `engineered_features.md`.

## Agent pipeline (`agent/`)

XGBoost flags transactions, and a reviewer (an LLM agent, or the no-LLM baseline) confirms or clears each flag using label-free, point-in-time tools.

```bash
python -m pytest agent -q                                  # leakage tests for the tools
python -m agent.pipeline run --split val --sample 300      # no-LLM baseline reviewer on 300 flagged cases
python -m agent.pipeline evaluate --split val --run data/agent_runs/<run>.jsonl

# LLM reviewer (pip install openai; set the provider's API key, e.g. OPENROUTER_API_KEY)
python -m agent.pipeline run --split val --sample 20 --reviewer llm --provider openrouter --model <model>
#   --no-score  withhold the XGBoost score    --no-kyc  remove the KYC tool
```

The LLM reviewer (`agent/llm.py`) works with any OpenAI-compatible API: `openai`, `gemini`, `openrouter` or a local `ollama`. Start with a 20-case pilot to measure tokens and cost per case before larger runs.

For a local model, create the 8k-context Qwen variant once. Ollama's default 4,096-token
window is too small for a prompt carrying all eight tool results, and it truncates silently:

```bash
ollama pull qwen2.5:3b
ollama create qwen2.5:3b-8k -f ollama/qwen2.5-3b-8k.Modelfile
python -m agent.pipeline run --split val --sample 20 --reviewer llm --rag --provider ollama --model qwen2.5:3b-8k
```

Before the LLM analyses a case, the reviewer pre-collects transaction history and
spending summary, then exposes the remaining enabled case-bound tools for the model to
request when their evidence is needed. It also runs deterministic amount and
transaction-velocity checks. Numeric evidence is validated against the collected case
data. Reviews below the configured confidence threshold are returned as
`needs_human_review` rather than being auto-resolved.

The case-bound tools also provide merchant familiarity, amount deviation, velocity
windows, transaction-location comparisons, and XGBoost threshold context. These
lookups use only data strictly before the flagged transaction and never expose the
fraud label.

The Streamlit chat stores each session under `data/agent_runs/chat_sessions/` and
keeps its session ID in the dashboard URL. Refreshing the same URL restores the
conversation, open case, and follow-up context. Use **Clear chat** to remove the
saved session. The collapsible sidebar lists the 20 most recent saved chats and
its **New chat** button starts a separate session. Dashboard case reviews use the
same synthetic RAG guidance as the command-line reviewer.

## Tactic RAG (`--tactics`)

`agent/tactics.py` holds four fraud-tactic documents (out-of-character late-night
spending, stolen-card spree, online purchase fraud, small charges inside a spree) and a
note on common innocent explanations. Each tactic has a checklist of **core**,
**supporting** and **against** signals that code tests against the case's tool results.
A tactic matches only when all its core signals are present; matches are ranked by
supporting minus against, and each sign is shown to the model as
`[yes]`/`[no]`/`[unknown]`. Retrieval is by checklist, not by wording.

```bash
python -m agent.pipeline run --split val --sample 20 --reviewer llm --tactics --provider ollama --model qwen2.5:3b-8k
```

The tactics were chosen from a check of the fit rows (the first 80% of
`fraudTrain.csv`): late-night spending and amounts far above a customer's usual were
the strongest signals, online categories moderate, and fraud arrives as a burst of
about ten transactions over two days. Merchant distance from home is the same for fraud
and legitimate transactions in this simulated data, so there is no location tactic. The
documents describe mechanisms rather than statistics, and nothing in them reads a label.
Each run's JSONL records every tactic's ticks under `retrieved_documents`.

The model sees the best-matching tactic in full, plus one line for any other strong or
moderate match. A fraud call with no matched tactic is returned as `needs_human_review`.
Validation results for every design tried so far, with the run logs and the scripts behind
them, are in [`results/agent_eval/`](results/agent_eval/README.md).

## Synthetic RAG experiment

Generate the inspectable label-free corpus:

```bash
python -m agent.rag_corpus
```

Run the LLM reviewer with retrieval enabled:

```bash
ollama pull nomic-embed-text
python -m agent.pipeline run --split val --sample 20 --reviewer llm --rag --tactics --provider ollama --model qwen2.5:3b-8k
```

With both `--rag` and `--tactics`, the combined reviewer first runs the deterministic
tactic checklist, then uses the matched tactic and case signals to query the local
Ollama `nomic-embed-text` model. The resulting generic investigation guidance is
provided to the same LLM decision alongside the tactic check. If Ollama or the
embedding model is unavailable, it records the error on the retriever and falls back
to the original lexical ranking. Customer-specific evidence still comes only from the
existing point-in-time tools. Use `--embedding-model <name>` to select another Ollama
embedding model. Document embeddings are persisted under
`data/agent_runs/rag_embeddings/` and are reused when the corpus and embedding model
have not changed; query embeddings are still generated for each case.

Use only `--rag` for generic embedding RAG or only `--tactics` for checklist RAG.
