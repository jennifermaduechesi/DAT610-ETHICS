"""
DAT610: Ethics & Privacy - Assignment 1
Synthetic Data Validation for Fraud Detection

SCRIPT 01 - BUILD THE 'REAL' DATASET
------------------------------------
This script builds the *real* reference dataset that everything else is
validated against. In a live setting this would be a genuine extract of
FintechPay transactions, but raw transaction records contain customer PII and,
under Nigeria's NDPA (2023), cannot be used freely in development/staging
environments. We therefore construct a realistic stand-in that follows the
exact FintechPay schema from the lecture (transaction_amount in Naira, customer
age, account balance, etc.) with deliberately built-in fraud signal so that the
downstream CTGAN model and the five-level validation framework have something
meaningful to learn and to test.

Schema (from the lecture slides):
    transaction_amount     - continuous, transaction value in NGN
    customer_age           - discrete, 18-75
    account_balance        - continuous, balance before the transaction
    num_transactions_30d   - discrete, transaction count in prior 30 days
    transaction_hour       - discrete, hour of day 0-23
    distance_from_home_km  - continuous, distance from registered home address
    merchant_category      - categorical (grocery/electronics/entertainment/travel/utilities)
    is_fraud               - binary target (1 = fraud, 0 = legitimate)

Output: data/real_data.csv
"""

import numpy as np
import pandas as pd

# A fixed seed keeps the whole assignment reproducible: rerunning any script
# reproduces the same figures and validation numbers reported in Part 1.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# We generate more than the required 1,500 records. Real-world fraud is ~1% of
# volume, but at 1% a 2,000-row sample yields only ~20 fraud cases, which is too
# few for a stable KS / chi-squared / TSTR result. We use 6,000 records at a ~4%
# fraud rate: still a strongly imbalanced problem (the point of the case study),
# but with enough positives (~240) for the statistical tests to be reliable.
N_RECORDS = 6000
FRAUD_RATE = 0.04

MERCHANT_CATEGORIES = ["grocery", "electronics", "entertainment", "travel", "utilities"]


def generate_real_dataset(n_records: int, fraud_rate: float) -> pd.DataFrame:
    """Generate the real reference dataset with realistic fraud correlations.

    We draw the fraud label first, then generate each feature *conditionally* on
    that label. This guarantees a learnable relationship between the features and
    is_fraud (fraud tends to be larger, later at night, and further from home),
    which is exactly what the ML-utility test in Level 4 depends on.
    """
    # --- Fraud label -----------------------------------------------------------
    is_fraud = np.random.binomial(1, fraud_rate, n_records)
    fraud_mask = is_fraud == 1
    legit_mask = ~fraud_mask
    n_fraud = int(fraud_mask.sum())
    n_legit = n_records - n_fraud

    # --- customer_age ----------------------------------------------------------
    # Roughly normal around 35, clipped to the documented 18-75 range. Age is
    # only weakly related to fraud, so both classes share the same distribution.
    customer_age = np.clip(np.random.normal(35, 12, n_records), 18, 75).round().astype(int)

    # --- account_balance -------------------------------------------------------
    # Heavy right-skew (lognormal). Fraudulent accounts tend to be drained or
    # newly opened, so they carry slightly lower balances on average.
    account_balance = np.empty(n_records)
    account_balance[legit_mask] = np.random.lognormal(mean=11.5, sigma=1.0, size=n_legit)
    account_balance[fraud_mask] = np.random.lognormal(mean=11.0, sigma=1.1, size=n_fraud)
    account_balance = account_balance.round(2)

    # --- transaction_amount ----------------------------------------------------
    # Legitimate spend is a moderate lognormal. Fraud is multi-modal: a body of
    # mid-size probing transactions plus a heavier high-value tail (large cash-out
    # attempts). This multi-modality is exactly what CTGAN's mode-specific
    # normalisation is designed to reproduce.
    transaction_amount = np.empty(n_records)
    transaction_amount[legit_mask] = np.random.lognormal(mean=8.5, sigma=0.8, size=n_legit)
    # For fraud, mix two modes: 60% probing spend, 40% high-value cash-out.
    fraud_amounts = np.where(
        np.random.random(n_fraud) < 0.6,
        np.random.lognormal(mean=9.0, sigma=0.7, size=n_fraud),
        np.random.lognormal(mean=10.5, sigma=0.6, size=n_fraud),
    )
    transaction_amount[fraud_mask] = fraud_amounts
    transaction_amount = transaction_amount.round(2)

    # --- num_transactions_30d --------------------------------------------------
    # Velocity: fraud rings push many transactions in a short window, so the
    # fraud class has a higher Poisson rate.
    num_transactions_30d = np.empty(n_records, dtype=int)
    num_transactions_30d[legit_mask] = np.random.poisson(lam=15, size=n_legit)
    num_transactions_30d[fraud_mask] = np.random.poisson(lam=28, size=n_fraud)

    # --- transaction_hour ------------------------------------------------------
    # Legitimate activity peaks during the day; fraud skews toward the small
    # hours (0-5) when monitoring is thinnest.
    transaction_hour = np.empty(n_records, dtype=int)
    transaction_hour[legit_mask] = np.clip(
        np.random.normal(14, 4, n_legit).round(), 0, 23
    ).astype(int)
    # Fraud: mixture of late-night (0-5) and otherwise-uniform hours.
    night = np.random.random(n_fraud) < 0.55
    fraud_hours = np.where(
        night,
        np.random.randint(0, 6, n_fraud),
        np.random.randint(0, 24, n_fraud),
    )
    transaction_hour[fraud_mask] = fraud_hours

    # --- distance_from_home_km -------------------------------------------------
    # Legitimate spend is close to home (short exponential); fraud happens far
    # away far more often.
    distance_from_home_km = np.empty(n_records)
    distance_from_home_km[legit_mask] = np.random.exponential(scale=8, size=n_legit)
    distance_from_home_km[fraud_mask] = np.random.exponential(scale=45, size=n_fraud)
    distance_from_home_km = distance_from_home_km.round(2)

    # --- merchant_category -----------------------------------------------------
    # Legitimate spend leans toward everyday categories (grocery/utilities).
    # Fraud concentrates in high-resale categories (electronics/travel).
    legit_probs = [0.35, 0.15, 0.20, 0.10, 0.20]  # grocery-heavy
    fraud_probs = [0.10, 0.40, 0.15, 0.30, 0.05]  # electronics/travel-heavy
    merchant_category = np.empty(n_records, dtype=object)
    merchant_category[legit_mask] = np.random.choice(
        MERCHANT_CATEGORIES, size=n_legit, p=legit_probs
    )
    merchant_category[fraud_mask] = np.random.choice(
        MERCHANT_CATEGORIES, size=n_fraud, p=fraud_probs
    )

    df = pd.DataFrame(
        {
            "transaction_amount": transaction_amount,
            "customer_age": customer_age,
            "account_balance": account_balance,
            "num_transactions_30d": num_transactions_30d,
            "transaction_hour": transaction_hour,
            "distance_from_home_km": distance_from_home_km,
            "merchant_category": merchant_category,
            "is_fraud": is_fraud,
        }
    )
    # Shuffle so fraud rows are not clustered at the end of the file.
    return df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)


def main() -> None:
    df = generate_real_dataset(N_RECORDS, FRAUD_RATE)
    out_path = "data/real_data.csv"
    df.to_csv(out_path, index=False)

    print("=" * 60)
    print("REAL DATASET GENERATED")
    print("=" * 60)
    print(f"Saved to        : {out_path}")
    print(f"Records         : {len(df)}")
    print(f"Fraud cases     : {int(df['is_fraud'].sum())} "
          f"({df['is_fraud'].mean() * 100:.2f}%)")
    print(f"Columns         : {list(df.columns)}")
    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nSummary statistics:")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
