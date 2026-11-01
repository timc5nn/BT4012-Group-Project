# Engineered Features

This document describes the features created by `preprocessing.ipynb` for
the credit-card fraud-detection models. The notebook writes
`data/processed/fraudTrain_features.csv` and
`data/processed/fraudTest_features.csv`, each with 111 columns: the `is_fraud`
label plus 110 features.

Run order: `kyc.ipynb` (generates `data/kyc_customers.csv`), then
`preprocessing.ipynb`, then `supervised.ipynb`.

The features fall into three groups, which `supervised.ipynb` compares:

| Group | Columns | Count |
|---|---|---|
| Transaction (baseline) | everything without a prefix below | 83 |
| KYC | `kyc_*` | 22 |
| Behavioural | `beh_*` | 5 |

## Raw columns kept as-is

| Feature | Description |
|---|---|
| `amt` | Transaction amount. |
| `city_pop` | Population of the cardholder's city. |

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
| `month` | Month number from 1 to 12. |
| `day_of_month` | Day of the month. |
| `hour` | Hour of the day from 0 to 23. |
| `day_of_week` | Day of the week, where Monday is 0 and Sunday is 6. |
| `is_weekend` | 1 for Saturday or Sunday; otherwise 0. |
| `is_night` | 1 for transactions between midnight and 5:59 a.m.; otherwise 0. |
| `hour_sin`, `hour_cos` | Cyclical representation of the hour, preserving the relationship between 23:00 and 00:00. |
| `dow_sin`, `dow_cos` | Cyclical representation of the day of the week. |
| `month_sin`, `month_cos` | Cyclical representation of the month. |

There is no `year` feature: it has only two values and the test set is
entirely 2020, so it cannot generalise to the test period.

## Demographic features

| Feature | Description |
|---|---|
| `age` | Cardholder age at the time of the transaction, calculated from `dob` and the transaction timestamp. |
| `gender_M` | One-hot indicator (baseline: `F`). |

## Geographic features

| Feature | Description |
|---|---|
| `log_distance_km` | `log1p` of the Haversine distance in kilometres between the cardholder and merchant locations. |
| `distance_is_large` | 1 when distance is at or above the training-set 95th percentile; otherwise 0. |
| `state_*` | One-hot indicators for the cardholder's state (baseline: `TX`). |

Distance shows no relationship with fraud in this dataset (fraud rate is flat
across distance quintiles for online, POS, and other categories), so only these
two distance features are kept. Raw `distance_km` and the earlier
distance interactions (`online_far_transaction`,
`amount_distance_interaction`) were removed as redundant.

## Merchant category

| Feature | Description |
|---|---|
| `category_*` | One-hot indicators for the merchant category (baseline: `gas_transport`), distinguishing e.g. `grocery_net` and `grocery_pos`. |

`is_online`, `is_pos`, and a suffix-free `category_base` are not created.
Each is an exact sum of `category` dummies (for example, `is_online` = sum of
the `_net` dummies), so including them would create perfect multicollinearity.

## KYC features (synthetic)

`kyc.ipynb` generates an artificial KYC (Know Your Customer) file for all 999
customers, keyed by `cc_num`. Every field is derived from attributes already
in the data (age, job title, city population, first transaction date) plus
seeded noise. **No KYC field is generated from `is_fraud`.**

This is deliberate. Each customer in this dataset is defrauded at most once,
in a single 1–2 day burst, and never again. A KYC field built from past
fraud labels would therefore point the wrong way, and one built from all
labels would leak the test answers.

The KYC table is joined to every transaction on `cc_num`.

| Feature | Description |
|---|---|
| `kyc_log_annual_income` | Log of declared annual income. |
| `kyc_credit_score` | Credit score (300–850). |
| `kyc_declared_txn_per_month` | Expected transactions per month, declared at onboarding. |
| `kyc_num_products` | Number of products held with the bank. |
| `kyc_account_tenure_years` | Years since the account was opened, at the time of each transaction. |
| `kyc_address_verified`, `kyc_phone_verified`, `kyc_email_verified` | 0/1 verification flags. |
| `kyc_pep_flag` | 1 for a politically exposed person. |
| `kyc_sanctions_potential_match` | 1 when sanctions screening found a potential match that was cleared. |
| `kyc_review_overdue` | 1 when the last KYC review is older than the risk rating's review cycle. |
| `kyc_risk_rating` | 0 = Low, 1 = Medium, 2 = High. |
| `kyc_employment_status_*` | One-hot (baseline: `Employed`). |
| `kyc_residency_status_*` | One-hot (baseline: `Citizen`). |
| `kyc_id_verification_method_*` | One-hot (baseline: `Document (in person)`). |
| `kyc_amt_to_declared_spend` | `amt` ÷ declared monthly spend: how this purchase compares with what the customer said they would spend. |
| `kyc_amt_to_credit_limit` | `amt` ÷ credit limit. |

Not used as features: `onboarding_channel` (exactly collinear with the ID
verification method, because branch always means in-person verification),
`source_of_funds` (almost fully determined by employment), `income_band`
(binned income), the declared spend and credit limit levels (both derived
from income; they enter through the two ratios), and `kyc_risk_points`
(a near-copy of the rating).

## Behavioural features

These compare each transaction with the customer's own recent and past
behaviour. They use only the current and earlier transactions of the same
customer, never later ones and never labels. Train and test are processed as
one timeline, so early test transactions can see the end of the training
period, as a live system would.

| Feature | Description |
|---|---|
| `beh_txn_count_1h` | The customer's transactions in the past hour, including this one. |
| `beh_txn_count_24h` | The customer's transactions in the past 24 hours, including this one. |
| `beh_log_amt_sum_24h` | Log of the customer's total spend in the past 24 hours, including this one. |
| `beh_log_secs_since_prev` | Log of seconds since the customer's previous transaction (capped at 30 days). |
| `beh_amt_to_cust_avg` | `amt` ÷ the average of the customer's earlier transactions (capped at 20; 1 for the first transaction). |

## Removed columns

| Column(s) | Reason |
|---|---|
| `trans_date_trans_time`, `dob` | Replaced by time features and `age` |
| `trans_num`, `cc_num` | Identifiers; a card number has no numeric meaning (`cc_num` is used first, to join the KYC table and group the behavioural features) |
| `first`, `last`, `street` | Personal information |
| `city`, `zip`, `merchant`, `job` | High-cardinality fields excluded from this baseline |
| `unix_time` | Same information as `month`/`day_of_month`/`hour` |
| `merch_lat`, `merch_long` | Near-copies of the cardholder coordinates; their only distinct information is the distance |
| `lat`, `long` | Largely determined by `state`; `state` is kept as the more interpretable location feature |

`merchant` and `job` can be added later using frequency encoding or carefully
regularised target encoding as a separate experiment.

## Categorical encoding

`category`, `gender`, `state`, and the KYC fields `kyc_employment_status`,
`kyc_residency_status` and `kyc_id_verification_method` are converted to one-hot indicator columns
using `pandas.get_dummies` with `drop_first=True`. A full set of dummies
always sums to 1 and is perfectly collinear with the intercept (the "dummy
variable trap"), so one category per column is dropped as the baseline. Each
kept coefficient is read relative to that baseline.

The baseline is the most frequent category in the training set. A rare
baseline (e.g. `AK`, 2,120 rows) leaves the other dummies summing to 1 for
almost every row, which causes near-collinearity.

There are no missing values in `category`, `gender`, or `state`. Rows with a
missing `gender` are dropped before encoding as a safeguard; otherwise they
would be silently encoded as the baseline.

The train and test data share one category list so both files have identical
columns. Indicators are stored sparsely to save memory.

## Multicollinearity

On a 200,000-row training sample, the feature matrix plus an intercept has
full rank (no exact linear dependency), and every variance inflation factor
is below 10. The highest VIFs are the time features (`hour`, `is_weekend`,
`day_of_week`), which describe overlapping information by design.

Two first designs of the new features were replaced to keep this true:
`log(amt) − log(customer average)` was a near-copy of `log_amt` (VIF ~35), so
`beh_amt_to_cust_avg` is a plain ratio instead; and the credit limit level
(VIF ~12, derived from income) was dropped in favour of
`kyc_amt_to_credit_limit`.

## Leakage prevention

The target column, `is_fraud`, is never used to construct these features,
including the synthetic KYC fields and the behavioural features.
The thresholds for `amt_is_large` and `distance_is_large` are calculated using
the training data only, then applied unchanged to the test data. `is_fraud`
is set aside before encoding and carried through unchanged as the label in
both output files.

Model-specific notes:

- Logistic regression: standardise numeric columns; categorical columns are
  already one-hot encoded
- Random forest / XGBoost: standardisation is normally unnecessary
