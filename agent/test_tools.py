"""Leakage tests for the agent's tools. Run with: python -m pytest agent -q

Needs data/fraudTrain.csv, data/fraudTest.csv and data/kyc_customers.csv.
"""

import json

import numpy as np
import pandas as pd
import pytest

from agent.data import RAW_FILES, TransactionStore
from agent.config import DATA_DIR
from agent.tools import CaseTools, TOOL_DEFINITIONS


@pytest.fixture(scope="module")
def store():
    return TransactionStore.load()


@pytest.fixture(scope="module")
def cases():
    """200 random transactions (with labels, for the tests only), half of them fraud."""
    raw = pd.read_csv(DATA_DIR / RAW_FILES[0], usecols=["cc_num", "trans_date_trans_time", "is_fraud"])
    raw["trans_date_trans_time"] = pd.to_datetime(raw["trans_date_trans_time"])
    fraud = raw[raw["is_fraud"] == 1].sample(100, random_state=0)
    legit = raw[raw["is_fraud"] == 0].sample(100, random_state=0)
    return pd.concat([fraud, legit])


def all_outputs(store, row):
    tools = CaseTools(store, row.cc_num, row.trans_date_trans_time)
    return tools, {d["name"]: tools.call(d["name"], {"limit": 100} if d["name"] == "get_transaction_history" else {})
                   for d in TOOL_DEFINITIONS}


def test_store_never_loads_labels(store):
    assert "is_fraud" not in store.columns


def test_no_label_or_fraud_wording_in_any_output(store, cases):
    for row in cases.itertuples(index=False):
        _, outputs = all_outputs(store, row)
        text = json.dumps(outputs).lower()
        assert "is_fraud" not in text
        assert "fraud" not in text, "merchant names must have their 'fraud_' prefix stripped"


def test_history_is_strictly_before_the_case(store, cases):
    for row in cases.itertuples(index=False):
        _, outputs = all_outputs(store, row)
        times = [pd.Timestamp(t["time"]) for t in outputs["get_transaction_history"]["transactions"]]
        assert all(t < row.trans_date_trans_time for t in times)


def test_summary_counts_only_earlier_transactions(store, cases):
    raw = pd.concat([pd.read_csv(DATA_DIR / f, usecols=["cc_num", "trans_date_trans_time"]) for f in RAW_FILES])
    raw["trans_date_trans_time"] = pd.to_datetime(raw["trans_date_trans_time"])
    for row in cases.head(40).itertuples(index=False):
        _, outputs = all_outputs(store, row)
        expected = ((raw["cc_num"] == row.cc_num) & (raw["trans_date_trans_time"] < row.trans_date_trans_time)).sum()
        assert outputs["get_spending_summary"].get("earlier_transactions", 0) == expected


def test_history_belongs_to_the_case_customer(store, cases):
    row = cases.iloc[0]
    history = store.history(row.cc_num, row.trans_date_trans_time)
    raw = pd.concat([pd.read_csv(DATA_DIR / f, usecols=["cc_num", "trans_date_trans_time", "amt"]) for f in RAW_FILES])
    raw["trans_date_trans_time"] = pd.to_datetime(raw["trans_date_trans_time"])
    own = raw[(raw["cc_num"] == row.cc_num) & (raw["trans_date_trans_time"] < row.trans_date_trans_time)]
    assert np.allclose(np.sort(history["amount"].to_numpy()), np.sort(own["amt"].to_numpy()))


def test_tools_reject_unexpected_arguments(store, cases):
    row = cases.iloc[0]
    tools = CaseTools(store, row.cc_num, row.trans_date_trans_time)
    assert "error" in tools.call("get_transaction_history", {"before": "2099-01-01"})
    assert "error" in tools.call("get_kyc_profile", {"customer_id": "KYC-00001"})
    assert "error" in tools.call("get_transaction_history", {"limit": 1000})
    assert "error" in tools.call("delete_everything", {})
