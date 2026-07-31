"""
DAT610: Ethics & Privacy - Assignment 1
Synthetic Data Validation for Fraud Detection

SCRIPT 02 - GENERATE SYNTHETIC DATA WITH CTGAN
----------------------------------------------
This is the "generative AI" step. We train a Conditional Tabular GAN (CTGAN)
via the SDV library on the real dataset and sample a synthetic copy of the same
size. CTGAN uses mode-specific normalisation, which lets it reproduce the
multi-modal continuous columns (e.g. the two-mode fraud transaction_amount) and
the strong class imbalance in is_fraud.

We produce TWO synthetic datasets on purpose:

  1. synthetic_data.csv      - the honest CTGAN output. This is what we expect
                               to PASS the five-level validation framework.

  2. corrupted_synthetic.csv - the same CTGAN output with two features
                               deliberately damaged, mirroring the lecture's
                               "Find the Failures" lab. This lets the validation
                               report demonstrate that the framework actually
                               *catches* problems instead of rubber-stamping
                               everything green:
                                 * transaction_amount is rescaled (x3)  -> wrong
                                   scale, should fail KS + Wasserstein.
                                 * is_fraud rate is inflated to ~18%    -> wrong
                                   fraud rate, should fail chi-squared and hurt
                                   the TSTR model.

Outputs:
    data/synthetic_data.csv
    data/corrupted_synthetic.csv
    outputs/metadata.json      (the auto-detected SDV metadata, for the report)
"""

import json

import numpy as np
import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# The lecture slide trains for 300 epochs; we keep that for fidelity.
CTGAN_EPOCHS = 300


def build_metadata(real_df: pd.DataFrame) -> SingleTableMetadata:
    """Auto-detect the table metadata and then verify/correct it by hand.

    The lecture is emphatic: 'Always verify: print(metadata.to_dict())'. SDV can
    mis-guess column types, so after auto-detection we explicitly pin the two
    categorical columns and the discrete integer columns.
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_df)

    # Force the columns we know to be categorical (SDV sometimes reads the 0/1
    # is_fraud target as numerical, which would let it generate 0.37 fraud flags).
    metadata.update_column("merchant_category", sdtype="categorical")
    metadata.update_column("is_fraud", sdtype="categorical")
    return metadata


def corrupt_synthetic(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Deliberately damage two features to create a 'failing' synthetic set.

    This is the artefact the validation report uses to prove the framework can
    reject bad data, echoing the lecture's failure taxonomy ('wrong scale erases
    the fraud signal'; 'mismatch in is_fraud = model trains on the wrong rate').
    """
    corrupted = synthetic_df.copy()

    # Corruption 1: wrong monetary scale. Multiplying transaction_amount by 3
    # shifts the whole distribution, so KS rejects it and its normalised
    # Wasserstein distance blows past the 0.15 threshold.
    corrupted["transaction_amount"] = corrupted["transaction_amount"] * 3.0

    # Corruption 2: wrong fraud rate. Flip a batch of legitimate rows to fraud so
    # the fraud rate jumps to ~18% (vs ~4% real). Chi-squared on is_fraud will
    # reject, and a model trained on this learns an inflated prior.
    rng = np.random.default_rng(RANDOM_SEED)
    legit_idx = corrupted.index[corrupted["is_fraud"] == 0].to_numpy()
    target_fraud = 0.18
    n_to_flip = int(len(corrupted) * target_fraud) - int((corrupted["is_fraud"] == 1).sum())
    n_to_flip = max(n_to_flip, 0)
    if n_to_flip > 0 and len(legit_idx) >= n_to_flip:
        flip_idx = rng.choice(legit_idx, size=n_to_flip, replace=False)
        corrupted.loc[flip_idx, "is_fraud"] = 1
    return corrupted


def main() -> None:
    real_df = pd.read_csv("data/real_data.csv")
    print(f"Loaded real data: {real_df.shape[0]} rows, {real_df.shape[1]} columns")

    # --- Step 1: metadata ------------------------------------------------------
    metadata = build_metadata(real_df)
    print("\nAuto-detected & verified metadata:")
    print(json.dumps(metadata.to_dict(), indent=2))
    with open("outputs/metadata.json", "w") as fh:
        json.dump(metadata.to_dict(), fh, indent=2)

    # --- Step 2: train CTGAN ---------------------------------------------------
    print(f"\nTraining CTGAN for {CTGAN_EPOCHS} epochs (this can take a few minutes)...")
    synthesizer = CTGANSynthesizer(metadata, epochs=CTGAN_EPOCHS, verbose=True)
    synthesizer.fit(real_df)

    # --- Step 3: sample the honest synthetic set -------------------------------
    synthetic_df = synthesizer.sample(num_rows=len(real_df))
    synthetic_df.to_csv("data/synthetic_data.csv", index=False)
    print(f"\nSaved honest synthetic data: data/synthetic_data.csv "
          f"({len(synthetic_df)} rows, "
          f"{synthetic_df['is_fraud'].mean() * 100:.2f}% fraud)")

    # --- Step 4: build the corrupted twin --------------------------------------
    corrupted_df = corrupt_synthetic(synthetic_df)
    corrupted_df.to_csv("data/corrupted_synthetic.csv", index=False)
    print(f"Saved corrupted synthetic data: data/corrupted_synthetic.csv "
          f"({len(corrupted_df)} rows, "
          f"{corrupted_df['is_fraud'].mean() * 100:.2f}% fraud)")

    print("\nDone. Real vs synthetic fraud rate:")
    print(f"  real      : {real_df['is_fraud'].mean() * 100:.2f}%")
    print(f"  synthetic : {synthetic_df['is_fraud'].mean() * 100:.2f}%")
    print(f"  corrupted : {corrupted_df['is_fraud'].mean() * 100:.2f}%")


if __name__ == "__main__":
    main()
