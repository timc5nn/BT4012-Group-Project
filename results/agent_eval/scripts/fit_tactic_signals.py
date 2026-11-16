"""Which fraud tactics occur in the FIT rows (first 80% of fraudTrain.csv)?

Compares fraud with legitimate transactions on each candidate tactic's signals, then on
signal combinations. Every per-transaction signal uses only the same card's earlier rows.
No validation or test data is read. Run from the repo root (about 5 seconds):

    python results/agent_eval/scripts/fit_tactic_signals.py
"""
import numpy as np
import pandas as pd

cols = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "lat", "long",
        "merch_lat", "merch_long", "is_fraud"]
df = pd.read_csv("data/fraudTrain.csv", usecols=cols)
df = df.iloc[: int(len(df) * 0.8)].copy()
df["t"] = pd.to_datetime(df.trans_date_trans_time)
df = df.sort_values(["cc_num", "t"], kind="stable").reset_index(drop=True)
df["night"] = df.t.dt.hour.isin([22, 23, 0, 1, 2, 3])
df["online"] = df.category.str.endswith("_net")
g = df.groupby("cc_num", sort=False)

# Earlier transactions on the same card in the previous 1h / 24h
ts = df.t.values.astype("int64") // 10**9
cards = df.cc_num.values
n1, n24 = np.zeros(len(df), int), np.zeros(len(df), int)
starts = np.r_[0, np.flatnonzero(cards[1:] != cards[:-1]) + 1, len(df)]
for a, b in zip(starts[:-1], starts[1:]):
    t = ts[a:b]
    n1[a:b] = np.arange(b - a) - np.searchsorted(t, t - 3600, side="left")
    n24[a:b] = np.arange(b - a) - np.searchsorted(t, t - 86400, side="left")
df["n1h"], df["n24h"] = n1, n24

df["new_merchant"] = df.groupby(["cc_num", "merchant"], sort=False).cumcount() == 0
prior_n = g.cumcount()
prior_mean = (g.amt.cumsum() - df.amt) / prior_n.replace(0, np.nan)
df["big"] = (prior_n >= 10) & (df.amt / prior_mean >= 5)


def km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(h))


df["home_km"] = km(df.lat, df.long, df.merch_lat, df.merch_long)
df["prev_km"] = km(g.merch_lat.shift(), g.merch_long.shift(), df.merch_lat, df.merch_long)
next_amt, next_gap = g.amt.shift(-1), (g.t.shift(-1) - df.t).dt.total_seconds() / 3600
df["small_then_big"] = (df.amt < 20) & (next_amt > 200) & (next_gap <= 24)

F, L = df[df.is_fraud == 1], df[df.is_fraud == 0]
print(f"fit rows {len(df):,}  fraud {len(F):,}  cards with fraud {F.cc_num.nunique()}\n")
print(f"{'signal':48} {'fraud':>7} {'legit':>7}")
for name, col in [("online (_net) category", "online"), ("22:00-03:59", "night"),
                  ("first visit to this merchant", "new_merchant"), ("amount >= 5x card's earlier mean", "big")]:
    print(f"{name:48} {F[col].mean():7.0%} {L[col].mean():7.0%}")
print(f"{'>=3 earlier transactions in previous hour':48} {(F.n1h >= 3).mean():7.0%} {(L.n1h >= 3).mean():7.0%}")
print(f"{'>=8 earlier transactions in previous 24h':48} {(F.n24h >= 8).mean():7.0%} {(L.n24h >= 8).mean():7.0%}")
print(f"{'amount < $20':48} {(F.amt < 20).mean():7.0%} {(L.amt < 20).mean():7.0%}")
print(f"{'median km, merchant from home':48} {F.home_km.median():7.0f} {L.home_km.median():7.0f}")
print(f"{'median km, from previous merchant':48} {F.prev_km.median():7.0f} {L.prev_km.median():7.0f}")

ep = F.groupby("cc_num").agg(n=("amt", "size"), start=("t", "min"), end=("t", "max"), cats=("category", "nunique"))
hours = (ep.end - ep.start).dt.total_seconds() / 3600
print(f"\nfraud episodes {len(ep)}: median {ep.n.median():.0f} transactions over {hours.median():.0f} h, "
      f"{ep.cats.median():.0f} categories")

print(f"\n{'combination':40} {'% fraud':>8} {'% legit':>8} {'fraud rate when true':>21}")
combos = {
    "night + amount >=5x": df.night & df.big,
    "night + new merchant + amount >=5x": df.night & df.new_merchant & df.big,
    "online + amount >=5x": df.online & df.big,
    "night + online": df.night & df.online,
    "night + new merchant": df.night & df.new_merchant,
    "small charge then >$200 within 24h": df.small_then_big,
}
for name, mask in combos.items():
    print(f"{name:40} {mask[df.is_fraud == 1].mean():8.0%} {mask[df.is_fraud == 0].mean():8.1%} "
          f"{df.is_fraud[mask].mean():21.1%}")
print(f"(base fraud rate {df.is_fraud.mean():.2%})")
