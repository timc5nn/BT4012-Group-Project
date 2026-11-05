# BT4102-Group-Project

## Data

This project uses the [Credit Card Transactions Fraud Detection Dataset](https://www.kaggle.com/datasets/kartik2112/fraud-detection) (`fraudTrain.csv`, `fraudTest.csv`).

The CSVs are too large for GitHub (143MB / 335MB) so they are **not** committed — `data/` is gitignored. Download the dataset from the link above and place both files in a local `data/` folder before running the notebooks.

## Run order

Each step writes files into `data/` that the next one reads.

| Step | Notebook | Writes |
|---|---|---|
| 1 | `eda.ipynb` | nothing (exploration only) |
| 2 | `kyc.ipynb` | `data/kyc_customers.csv`, a synthetic KYC profile per customer |
| 3 | `preprocessing.ipynb` | `data/processed/fraud{Train,Test}_features.csv` |
| 4 | `supervised.ipynb` | nothing (logistic regression comparison) |
| 5 | `xgboost.ipynb` | stage 1: model, thresholds, flagged cases and scores in `data/processed/` |
| 6 | `agent_pipeline.ipynb` | stage 2 demo: runs in `data/agent_runs/` |

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

Before the LLM analyses a case, the reviewer collects the enabled case-bound tools
and runs deterministic amount and transaction-velocity checks. Numeric evidence is
validated against the collected case data. Reviews below the configured confidence
threshold are returned as `needs_human_review` rather than being auto-resolved.

The Streamlit chat stores each session under `data/agent_runs/chat_sessions/` and
keeps its session ID in the dashboard URL. Refreshing the same URL restores the
conversation, open case, and follow-up context. Use **Clear chat** to remove the
saved session. The collapsible sidebar lists the 20 most recent saved chats and
its **New chat** button starts a separate session.
