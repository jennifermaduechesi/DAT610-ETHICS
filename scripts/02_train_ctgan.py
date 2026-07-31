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
from sdv.sampling import Condition
from sdv.single_table import CTGANSynthesizer

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# The lecture slide trains for 300 epochs; we lift it slightly to 400 for better
# convergence on the continuous columns.
CTGAN_EPOCHS = 400


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

    # Corruption 2: wrong fraud rate. Force the fraud prevalence to a clearly wrong
    # 40% (vs ~10% real) by flipping legitimate rows to fraud. Chi-squared on
    # is_fraud will reject decisively, and a model trained on this learns a wildly
    # inflated fraud prior. We drive to a fixed target rather than an increment so
    # the corruption always lands, whatever prevalence the CTGAN output happened to
    # have.
    rng = np.random.default_rng(RANDOM_SEED)
    target_fraud = 0.40
    current_fraud = int((corrupted["is_fraud"] == 1).sum())
    n_to_flip = int(len(corrupted) * target_fraud) - current_fraud
    if n_to_flip > 0:
        legit_idx = corrupted.index[corrupted["is_fraud"] == 0].to_numpy()
        n_to_flip = min(n_to_flip, len(legit_idx))
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
    # First, an unconditioned draw. This documents a real CTGAN behaviour: on an
    # imbalanced target, the default sampler over-represents the minority class,
    # so the fraud rate comes out far above the real ~10%. We record this because
    # it is a genuine finding for the report.
    default_sample = synthesizer.sample(num_rows=len(real_df))
    print(f"\nDefault (unconditioned) sample fraud rate: "
          f"{default_sample['is_fraud'].mean() * 100:.2f}%  "
          f"(real is {real_df['is_fraud'].mean() * 100:.2f}%)")

    # Now the correct approach for a *Conditional* GAN: use conditional sampling to
    # enforce the real class prior (same fraud/legit counts as the real data). This
    # is the intended mechanism for controlling prevalence in an augmentation set.
    n_fraud = int(real_df["is_fraud"].sum())
    n_legit = len(real_df) - n_fraud
    conditions = [
        Condition(num_rows=n_fraud, column_values={"is_fraud": 1}),
        Condition(num_rows=n_legit, column_values={"is_fraud": 0}),
    ]
    synthetic_df = synthesizer.sample_from_conditions(conditions)
    synthetic_df = synthetic_df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    synthetic_df.to_csv("data/synthetic_data.csv", index=False)
    print(f"Saved honest synthetic data: data/synthetic_data.csv "
          f"({len(synthetic_df)} rows, "
          f"{synthetic_df['is_fraud'].mean() * 100:.2f}% fraud, conditioned to real prior)")

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
