"""Architecture page: how a flagged transaction reaches a decision."""

import streamlit as st
from urllib.parse import quote

DIAGRAM = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body { margin: 0; background: #101014; color: #ececf1;
    font-family: "Segoe UI", sans-serif; }
  .wrap { padding: 8px 4px 24px; }
  h2 { font-size: 13px; font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase; color: #9b9ba7; margin: 22px 0 10px; }
  .col { display: flex; flex-direction: column; align-items: center; gap: 8px; }
  .row { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; width: 100%; }
  .node { background: #18181d; border: 1px solid #2a2a32; border-radius: 12px;
    padding: 12px 14px; width: 100%; max-width: 420px; flex: 1 1 240px; box-sizing: border-box; }
  .node strong { display: block; font-size: 14px; font-weight: 600; margin-bottom: 4px; }
  .node span { display: block; color: #9b9ba7; font-size: 12px; line-height: 1.4; }
  .node.accent { border-color: #3d5a94; }
  .node.quiet { border-style: dashed; opacity: 0.85; }
  .arrow { color: #6d7380; font-size: 14px; line-height: 1; }
  .tools { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; }
  .tool { background: #14141a; border: 1px solid #2a2a32; border-radius: 999px;
    padding: 7px 12px; font-size: 12px; color: #d5d5de; }
  .note { margin-top: 18px; color: #9b9ba7; font-size: 13px; line-height: 1.5; }
</style>
</head>
<body>
<div class="wrap">
  <h2>Stage 1 · score every transaction</h2>
  <div class="col">
    <div class="node"><strong>Sparkov transactions</strong><span>fraudTrain.csv and fraudTest.csv. Card, time, amount, merchant.</span></div>
    <div class="arrow">↓</div>
    <div class="row">
      <div class="node"><strong>KYC profiles</strong><span>One synthetic profile per card. Built without the fraud label.</span></div>
      <div class="node"><strong>Behaviour features</strong><span>Spending history using only earlier transactions.</span></div>
    </div>
    <div class="arrow">↓</div>
    <div class="node accent"><strong>XGBoost</strong><span>Depth 8, 94 trees. Standalone threshold 0.958. Flagging threshold 0.853.</span></div>
    <div class="arrow">↓</div>
    <div class="row">
      <div class="node accent"><strong>Flagged cases</strong><span>Score at or above 0.853. No fraud label. This is what the reviewer sees.</span></div>
      <div class="node quiet"><strong>Scores file</strong><span>Every score and the label. Used only when a run is evaluated.</span></div>
    </div>
  </div>

  <h2>Stage 2 · investigate one flag</h2>
  <div class="col">
    <div class="node accent"><strong>Case</strong><span>Customer, time, amount, merchant, category, and the XGBoost score.</span></div>
    <div class="arrow">↓</div>
    <div class="node"><strong>Baseline evidence</strong><span>The application pre-collects transaction history and spending summary before the model’s first turn.</span></div>
    <div class="arrow">↓</div>
    <div class="node accent"><strong>Qwen 2.5 3B · 8K</strong><span>Local Ollama model. It receives the baseline evidence and RAG guidance, then requests at least one relevant secondary tool before deciding.</span></div>
    <div class="arrow">↓</div>
    <div class="tools">
      <div class="tool">KYC profile</div>
      <div class="tool">Merchant history</div>
      <div class="tool">Amount deviation</div>
      <div class="tool">Velocity profile</div>
      <div class="tool">Location history</div>
      <div class="tool">Model explanation</div>
    </div>
    <div class="arrow">↓</div>
    <div class="node accent"><strong>Decision</strong><span>Fraud or legitimate, a confidence, the evidence, and a short case note.</span></div>
  </div>
  <p class="note">Each tool is locked to that customer and to times before the flagged transaction. The model receives synthetic RAG guidance for general investigation patterns, but customer-specific facts come only from the flagged transaction and case-bound tool results. A customer id or a date in the model’s arguments is ignored. The tools never load the fraud label.</p>
</div>
</body>
</html>
"""


def architecture_page():
    st.title("Architecture")
    st.caption("Two stages. The reviewer only sees flagged transactions, and only the past of that one customer.")
    st.iframe(f"data:text/html;charset=utf-8,{quote(DIAGRAM)}", height=980)
