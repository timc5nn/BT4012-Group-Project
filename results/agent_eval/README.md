# Stage 2 agent evaluation results

Saved run logs and analysis scripts for the LLM reviewer experiments of 15–20 Nov 2026, so
the numbers in the report can be re-checked without re-running the model (about 20–30 s per
case on a laptop). Every run is on the validation split except `test100_test`, the single
frozen-design run on the test split.

- `runs/`: the pipeline's JSONL output for each run, copied from `data/agent_runs/`. One line
  per case: decision, tool calls, retrieved documents or tactic checks, token use, time.
  **The logs contain no fraud labels.**
  - `current-combined-qwen8k-s2033_val` (+ `_transcripts`): 20-case diagnostic run of the
    combined reviewer, used for the process evaluation.
  - `test100_test`: the frozen design (confidence threshold 0.89) on a random sample of 100
    flagged test cases (seed 4102).
- `reports/`: `agent_eval_summary.csv` and `agent_eval_cases.csv` (process metrics from
  `evaluate_agent.py`), `judge_cache.jsonl` (cached LLM-judge scores), `qwen3b-val-100_sample.jsonl`
  (the 100-case validation run of the base LLM reviewer, without tactics or RAG, used to choose
  the confidence threshold) and
  `test100_test_cascade.json` (XGBoost alone against XGBoost + agent on the 100 test cases).
- `scripts/summarise_runs.py`: scores every run against the validation labels and against
  XGBoost alone on the same cases. It produces the tables below.
- `scripts/evaluate_agent.py`: process metrics (tool recall, argument accuracy, order, gates)
  and the LLM-judge grounding scores.
- `scripts/tune_confidence.py`: picks the human-review confidence threshold from a saved run.
- `scripts/run_xgb_combined_cascade.py`: XGBoost alone against the full cascade (below).
- `scripts/fit_tactic_signals.py`: which fraud tactics occur in the fit rows (no val/test data).
- `scripts/fit_tactic_matcher.py`: the tactic checklists run on fit-row transactions (no val/test data).

Run the scripts from the repo root. `summarise_runs.py` needs `data/processed/` from the Stage 1
notebook. These runs used Stage 1 files generated with recall target 0.98 (flag threshold
0.0075, standalone 0.603, 2,232 flagged validation cases). With different Stage 1 files the
case ids may not line up.

## End-to-end XGBoost versus combined-agent runner

To evaluate only XGBoost alone against the complete XGBoost + tactic checklist + embedding-RAG
cascade, run:

```bash
ollama pull nomic-embed-text
python results/agent_eval/scripts/run_xgb_combined_cascade.py \
  --split val --provider ollama --model qwen2.5:3b-8k
```

By default this reviews every XGBoost-flagged validation case and scores both systems over the
full validation split. Transactions below the XGBoost flag threshold are automatically cleared
by the cascade, while `needs_human_review` and failed cases are reported separately. Use
`--sample N --seed S` only for development; that produces a reviewed-sample report rather than
an exact full-split cascade result. The report is saved under `reports/` and the agent records
under `data/agent_runs/`.

## Designs tested

All runs used `qwen2.5:3b` on Ollama. The design runs below are on validation only.

| Run | Design | Code |
|---|---|---|
| `pilot-qwen-rag` | Only the spending summary pre-collected; model chooses other tools (it never did); generic RAG; 4k context | before `4b3b66f` |
| `pilot-qwen8k-alltools` | All 8 tools pre-collected; generic RAG; 8k context | `4b3b66f` |
| `pilot-qwen8k-tactics` | Tactic RAG v1: top 3 matched tactics in full | not committed separately |
| `eval100-qwen8k-tactics-v2` | v2: + small-charges tactic capped at moderate without a spree; + a fraud call with no matched tactic goes to human review | not committed separately |
| `eval100-qwen8k-tactics-v3` | v3: best tactic in full, other strong/moderate matches as one line, weak ones dropped | **this commit** (`--tactics`) |

The current combined reviewer on main (tactic checklist + embedding RAG) was then checked on a
20-case validation diagnostic (`current-combined-qwen8k-s2033`). The confidence threshold of
0.89 was chosen on a 100-case validation run of the base LLM reviewer, without tactics or RAG
(`reports/qwen3b-val-100_sample.jsonl`), and applied unchanged to the combined reviewer. The
frozen design was then run once on 100 flagged test cases
(`test100_test`; XGBoost alone 92 correct, agent 43 correct automatically with 29 sent to human
review, see `reports/test100_test_cascade.json`).

## Results

`correct` = frauds called fraud + legitimate cases called legitimate. Human review counts as
neither. XGBoost alone = the standalone threshold (0.603) on the same cases.

| | pilot (old) | pilot (all tools) | pilot (tactics v1) | 100 cases (v2) | 100 cases (v3) |
|---|---|---|---|---|---|
| Sample seed | 4102 | 4102 | 4102 | 2026 | 2027 |
| Cases (fraud/legit) | 11/9 | 11/9 | 11/9 | 72/28 | 65/35 |
| Fraud caught | 3 | 10 | 8 | 51 | 46 |
| Fraud → human review | 0 | 0 | 0 | 8 | 5 |
| Fraud cleared | 6 | 0 | 2 | 13 | 14 |
| Legit cleared | 7 | 0 | 3 | 9 | 9 |
| Legit → human review | 0 | 0 | 2 | 4 | 6 |
| False alarms | 1 | 7 | 4 | 13 | 20 |
| Errors | 3 | 3 | 1 | 2 | 0 |
| **Agent correct** | 10 | 10 | 11 | 60 | 55 |
| **XGBoost alone correct** | 19 | 19 | 19 | 91 | 93 |
| Seconds / case | 12.7 | 30.5 | 20.8 | 21.0 | 19.9 |
| Input tokens / case | 3,118 | 6,130 | 4,664 | 4,424 | 4,111 |

The three pilots use the same 20 cases. The two 100-case runs use different cases, so compare
rates: both caught 71% of frauds; v2 called 46% of legitimate cases fraud and v3 57%, a
difference within noise at this sample size. Seed 2027 shares 2 cases with the earlier samples;
excluding them gives 53 correct for the agent against 91 for XGBoost.

## Findings

1. **No reviewer design beats XGBoost alone at deciding fraud** with this 3B model. Changing
   how evidence or RAG is presented (1 tool, 8 tools, 3 tactics, 1 tactic) did not close the gap.
2. **Strong tactic matches are reliable:** 31 of 35 (v2) and 31 of 37 (v3) fraud calls on strong
   matches were correct. Most false alarms come from moderate or weak matches, and most missed
   frauds match no tactic.
3. **The model leans towards "fraud"** (66 of 100 calls in v3) and sometimes contradicts the
   `[yes]`/`[no]` marks it is given, e.g. calling a merchant with earlier visits "new".
4. **Fit-row facts behind the tactics** (`fit_tactic_signals.py`): 22:00–03:59 covers 85% of fraud
   vs 23% of legit; amount ≥ 5× the card's earlier mean covers 50% vs 2%; together 43% vs 0.4%.
   Fraud arrives as about 10 transactions over about 45 hours. Merchant distance from home is
   identical (78 km median) for fraud and legit, so the location tool carries no signal here.
5. **Checklist changes were justified on fit rows, not validation labels** (`fit_tactic_matcher.py`).
   The small-charges cap was kept because that tactic alone was weaker than with a spree (86% vs
   97% fraud in a 50/50 fit sample, before the cap). A proposed rule to mark "new merchant"
   unknown when history covers under 30 days was dropped: it would hide 46% of cases whose
   spree matches were still 89% fraud.

## Method notes

- Validation has now been used for three tactic design rounds (v1 → v2 → v3). Report this.
- Each new round was scored on a fresh random sample, not on the cases that prompted it.
- The test split was used once, after the design and the 0.89 threshold were frozen (`test100_test`).
