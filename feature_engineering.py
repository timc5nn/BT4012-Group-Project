"""Create modelling-ready feature files for the fraud-detection dataset.

Input:
    data/fraudTrain.csv
    data/fraudTest.csv

Output:
    data/processed/fraudTrain_features.csv
    data/processed/fraudTest_features.csv

The transformations are target-independent. Training-set quantiles are used
for threshold features so that test data does not influence preprocessing.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed"


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two latitude/longitude pairs."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def add_features(df, amount_95, distance_95):
    """Add shared, target-independent features to one dataframe."""
    result = df.copy()

    timestamp = pd.to_datetime(result["trans_date_trans_time"], errors="coerce")
    dob = pd.to_datetime(result["dob"], errors="coerce")

    result["year"] = timestamp.dt.year
    result["month"] = timestamp.dt.month
    result["day_of_month"] = timestamp.dt.day
    result["hour"] = timestamp.dt.hour
    result["day_of_week"] = timestamp.dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")
    result["is_night"] = result["hour"].isin([0, 1, 2, 3, 4, 5]).astype("int8")

    # Cyclical encodings preserve the fact that adjacent periods are similar.
    result["hour_sin"] = np.sin(2 * np.pi * result["hour"] / 24)
    result["hour_cos"] = np.cos(2 * np.pi * result["hour"] / 24)
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    result["month_sin"] = np.sin(2 * np.pi * result["month"] / 12)
    result["month_cos"] = np.cos(2 * np.pi * result["month"] / 12)

    result["age"] = ((timestamp - dob).dt.days / 365.25).round().astype("float")
    result["log_amt"] = np.log1p(result["amt"].clip(lower=0))
    result["amt_is_large"] = (result["amt"] >= amount_95).astype("int8")

    result["distance_km"] = haversine_km(
        result["lat"], result["long"], result["merch_lat"], result["merch_long"]
    )
    result["log_distance_km"] = np.log1p(result["distance_km"].clip(lower=0))
    result["distance_is_large"] = (result["distance_km"] >= distance_95).astype("int8")

    category = result["category"].fillna("missing").astype(str)
    result["is_online"] = category.str.endswith("_net").astype("int8")
    result["is_pos"] = category.str.endswith("_pos").astype("int8")
    result["category_base"] = (
        category.str.replace("_net", "", regex=False)
        .str.replace("_pos", "", regex=False)
    )

    # Interactions useful for a linear model; tree models can learn these too.
    result["online_far_transaction"] = result["is_online"] * result["log_distance_km"]
    result["amount_distance_interaction"] = result["log_amt"] * result["log_distance_km"]

    # Drop raw identifiers/free text/date columns. The target is retained only
    # in the training output and is never used to construct a feature.
    columns_to_drop = [
        "trans_date_trans_time", "dob", "trans_num", "first", "last",
        "street", "city", "zip", "merchant", "job", "Unnamed: 0",
    ]
    result = result.drop(columns=[c for c in columns_to_drop if c in result.columns])
    return result


def one_hot_encode(train_features, test_features):
    """One-hot encode categorical columns with identical columns in both files."""
    target = train_features["is_fraud"] if "is_fraud" in train_features else None
    train_without_target = train_features.drop(columns=["is_fraud"], errors="ignore")
    test_without_target = test_features.drop(columns=["is_fraud"], errors="ignore")

    categorical_columns = train_without_target.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Encode each dataset separately using the union of train/test categories.
    # sparse=True prevents a large dense dummy matrix from being allocated.
    train_parts = [train_without_target.drop(columns=categorical_columns)]
    test_parts = [test_without_target.drop(columns=categorical_columns)]
    for column in categorical_columns:
        categories = pd.Index(
            pd.concat(
                [train_without_target[column], test_without_target[column]],
                ignore_index=True,
            ).dropna().unique()
        )
        train_category = train_without_target[column].astype(
            pd.CategoricalDtype(categories=categories)
        )
        test_category = test_without_target[column].astype(
            pd.CategoricalDtype(categories=categories)
        )
        train_parts.append(pd.get_dummies(train_category, prefix=column, dummy_na=True, sparse=True))
        test_parts.append(pd.get_dummies(test_category, prefix=column, dummy_na=True, sparse=True))

    train_encoded = pd.concat(train_parts, axis=1).reset_index(drop=True)
    test_encoded = pd.concat(test_parts, axis=1).reset_index(drop=True)
    if target is not None:
        train_encoded.insert(0, "is_fraud", target.reset_index(drop=True).astype("int8"))
    return train_encoded, test_encoded


def main():
    train_path = DATA_DIR / "fraudTrain.csv"
    test_path = DATA_DIR / "fraudTest.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            "Expected data/fraudTrain.csv and data/fraudTest.csv. "
            "Download the dataset and place the files in data/."
        )

    train = pd.read_csv(train_path, index_col=0)
    test = pd.read_csv(test_path, index_col=0)

    # Fit thresholds on train only; do not let test distribution influence them.
    amount_95 = train["amt"].quantile(0.95)
    train_distance = haversine_km(
        train["lat"], train["long"], train["merch_lat"], train["merch_long"]
    )
    distance_95 = train_distance.quantile(0.95)

    train_features = add_features(train, amount_95, distance_95)
    test_features = add_features(test, amount_95, distance_95)
    train_features, test_features = one_hot_encode(train_features, test_features)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    train_output = OUTPUT_DIR / "fraudTrain_features.csv"
    test_output = OUTPUT_DIR / "fraudTest_features.csv"
    train_features.to_csv(train_output, index=False)
    test_features.to_csv(test_output, index=False)

    print(f"Wrote {train_features.shape[0]:,} rows x {train_features.shape[1]:,} columns to {train_output}")
    print(f"Wrote {test_features.shape[0]:,} rows x {test_features.shape[1]:,} columns to {test_output}")
    print(f"Training 95th-percentile amount threshold: {amount_95:.2f}")
    print(f"Training 95th-percentile distance threshold: {distance_95:.2f} km")


if __name__ == "__main__":
    main()
