"""Label-free, point-in-time access to transactions and KYC profiles.

The fraud label is never loaded: only the columns in TRANSACTION_COLUMNS are
read from the raw files. Lookups take a `before` timestamp and return only
strictly earlier transactions, so nothing at or after the transaction under
review can reach the agent.
"""

import numpy as np
import pandas as pd

from .config import DATA_DIR

TRANSACTION_COLUMNS = [
    "cc_num", "trans_num", "trans_date_trans_time", "amt", "merchant", "category",
    "lat", "long", "merch_lat", "merch_long",
]
RAW_FILES = ["fraudTrain.csv", "fraudTest.csv"]


class TransactionStore:
    """All transactions (train and test, one timeline) plus the KYC table."""

    def __init__(self, transactions, kyc):
        transactions = transactions.sort_values(["cc_num", "time"], kind="stable", ignore_index=True)
        self._cc = transactions["cc_num"].to_numpy()
        self._time = transactions["time"].to_numpy()
        self._amount = transactions["amt"].to_numpy()
        self._merchant = transactions["merchant"].to_numpy()
        self._category = transactions["category"].to_numpy()
        self._lat = transactions["lat"].to_numpy()
        self._long = transactions["long"].to_numpy()
        self._merch_lat = transactions["merch_lat"].to_numpy()
        self._merch_long = transactions["merch_long"].to_numpy()
        self.columns = list(transactions.columns)

        self._kyc = kyc.set_index("cc_num")
        self._cc_by_customer_id = dict(zip(kyc["customer_id"], kyc["cc_num"]))

    @classmethod
    def load(cls, data_dir=DATA_DIR):
        transactions = pd.concat(
            [pd.read_csv(data_dir / name, usecols=TRANSACTION_COLUMNS) for name in RAW_FILES],
            ignore_index=True,
        )
        transactions["time"] = pd.to_datetime(transactions.pop("trans_date_trans_time"))
        # Every Sparkov merchant name starts with "fraud_", legitimate ones included.
        # It is a simulator artefact, not a label, but it would mislead an LLM.
        assert transactions["merchant"].str.startswith("fraud_").all()
        transactions["merchant"] = transactions["merchant"].str.removeprefix("fraud_")
        kyc = pd.read_csv(data_dir / "kyc_customers.csv")
        return cls(transactions, kyc)

    def customer_id(self, cc_num):
        return self._kyc.at[cc_num, "customer_id"]

    def transaction_context(self, cc_num, time):
        """Return the flagged transaction's non-label context for a bound case."""
        start = np.searchsorted(self._cc, cc_num, side="left")
        end = np.searchsorted(self._cc, cc_num, side="right")
        matches = np.flatnonzero(self._time[start:end] == np.datetime64(pd.Timestamp(time)))
        if len(matches) == 0:
            return None
        index = start + matches[0]
        return {
            "amount": _number(self._amount[index]),
            "merchant": self._merchant[index],
            "category": self._category[index],
            "latitude": _number(self._lat[index]),
            "longitude": _number(self._long[index]),
            "merchant_latitude": _number(self._merch_lat[index]),
            "merchant_longitude": _number(self._merch_long[index]),
        }

    def cc_num(self, customer_id):
        return self._cc_by_customer_id[customer_id]

    def kyc_profile(self, cc_num):
        return self._kyc.loc[cc_num].to_dict()

    def history(self, cc_num, before, limit=None):
        """The customer's transactions strictly before `before`, oldest first (the last `limit` if given)."""
        start = np.searchsorted(self._cc, cc_num, side="left")
        end = np.searchsorted(self._cc, cc_num, side="right")
        times = self._time[start:end]
        stop = start + np.searchsorted(times, np.datetime64(pd.Timestamp(before)), side="left")
        first = start if limit is None else max(start, stop - limit)
        return pd.DataFrame({
            "time": self._time[first:stop],
            "amount": self._amount[first:stop],
            "merchant": self._merchant[first:stop],
            "category": self._category[first:stop],
            "lat": self._lat[first:stop],
            "long": self._long[first:stop],
            "merch_lat": self._merch_lat[first:stop],
            "merch_long": self._merch_long[first:stop],
        })


def _number(value):
    return float(value)
