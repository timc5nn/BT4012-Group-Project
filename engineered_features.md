# Engineered Features

This document describes the features created by `feature_engineering.py` for
the credit-card fraud-detection models.

## Transaction amount

| Feature | Description |
|---|---|
| `log_amt` | `log1p(amt)`, reducing the effect of the highly right-skewed transaction amount. |
| `amt_is_large` | 1 when `amt` is at or above the 95th percentile of the training-set transaction amounts; otherwise 0. |

## Time features

The original transaction timestamp is converted into calendar and time-of-day
features. The raw timestamp is then removed.

| Feature | Description |
|---|---|
| `year` | Calendar year of the transaction. |
| `month` | Month number from 1 to 12. |
| `day_of_month` | Day of the month. |
| `hour` | Hour of the day from 0 to 23. |
| `day_of_week` | Day of the week, where Monday is 0 and Sunday is 6. |
| `is_weekend` | 1 for Saturday or Sunday; otherwise 0. |
| `is_night` | 1 for transactions between midnight and 5:59 a.m.; otherwise 0. |
| `hour_sin`, `hour_cos` | Cyclical representation of the hour, preserving the relationship between 23:00 and 00:00. |
| `dow_sin`, `dow_cos` | Cyclical representation of the day of the week. |
| `month_sin`, `month_cos` | Cyclical representation of the month. |

## Demographic features

| Feature | Description |
|---|---|
| `age` | Cardholder age at the time of the transaction, calculated from `dob` and the transaction timestamp. |

The raw `dob` column is removed after calculating age.

## Geographic features

| Feature | Description |
|---|---|
| `distance_km` | Great-circle distance in kilometres between the cardholder location and merchant location, calculated using the Haversine formula. |
| `log_distance_km` | `log1p(distance_km)`, reducing distance skew. |
| `distance_is_large` | 1 when distance is at or above the training-set 95th percentile; otherwise 0. |

## Merchant and transaction-channel features

| Feature | Description |
|---|---|
| `is_online` | 1 when the category ends in `_net`, indicating an online transaction. |
| `is_pos` | 1 when the category ends in `_pos`, indicating a point-of-sale transaction. |
| `category_base` | Merchant category with `_net` and `_pos` suffixes removed. |

The original `category` column is retained so that models can distinguish
categories such as `grocery_net` and `grocery_pos`.

## Interaction features

| Feature | Description |
|---|---|
| `online_far_transaction` | `is_online * log_distance_km`, combining online-channel and distance information. |
| `amount_distance_interaction` | `log_amt * log_distance_km`, combining transaction value and distance. |

These interactions are particularly useful for logistic regression. Tree-based
models can learn similar interactions automatically, but retaining them keeps
the feature set consistent across models.

## Removed columns

The following raw columns are removed by the script:

- `trans_date_trans_time`: replaced by time-derived features
- `dob`: replaced by `age`
- `trans_num`: transaction identifier
- `first`, `last`: personal names
- `street`, `city`, `zip`: high-cardinality location or personal-information fields
- `merchant`, `job`: high-cardinality categorical fields excluded from the
  initial baseline to reduce runtime, memory usage, and overfitting risk
- `Unnamed: 0`: CSV index column, when present

## Leakage prevention

The target column, `is_fraud`, is never used to construct these features.
The thresholds for `amt_is_large` and `distance_is_large` are calculated using
the training data only, then applied unchanged to the test data.

The generated files are intended for use with model-specific preprocessing:

- Logistic regression: imputation and standardization of numeric columns; the
  generated files already contain one-hot encoded categorical columns
- Random forest: imputation; standardization is normally unnecessary because
  the generated files already contain one-hot encoded categorical columns
- XGBoost: imputation; standardization is normally unnecessary because the
  generated files already contain one-hot encoded categorical columns

## Categorical encoding

All remaining categorical columns are converted to one-hot indicator columns
using `pandas.get_dummies`. Missing categorical values receive their own
indicator through `dummy_na=True`. The indicators are stored sparsely to avoid
creating a very large dense matrix in memory.

The train and test data are encoded together only to establish the same set of
output columns. The target column `is_fraud` is removed before encoding and is
then restored only in the training output. No fraud labels are used to create
the encoded features.

The encoded columns include lower-cardinality fields such as `category`,
`category_base`, `gender`, and `state`. The raw categorical columns are
replaced by their indicator columns, for example:

```text
gender_F, gender_M, gender_missing
category_grocery_net, category_grocery_pos, ...
```

`merchant` and `job` are intentionally excluded from this initial baseline
because they have many unique values. They can be added later using frequency
encoding or carefully regularized target encoding as a separate experiment.
