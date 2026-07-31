"""
DAT610: Ethics & Privacy - Assignment 1
Synthetic Data Validation for Fraud Detection

SCRIPT 03 - FIVE-LEVEL VALIDATION FRAMEWORK
-------------------------------------------
This script runs the full validation framework taught in class against the
CTGAN synthetic dataset (and, for contrast, against the deliberately corrupted
one). Each level maps directly to a slide in the lecture:

    Level 1  Summary statistics       - means within 10% (>20% => stop)
    Level 2  Visual distributions     - KDE plots, bar charts, correlation heatmaps
    Level 3a Kolmogorov-Smirnov test  - continuous columns, p > 0.05
    Level 3b Chi-squared test         - categorical columns, p > 0.05
    Level 3c Wasserstein distance     - normalised drift < 0.15 per feature
    Level 4  ML utility - TSTR vs TRTR- AUC gap < 0.05
    Level 5  Privacy - DNNR           - median(d_SR)/median(d_RR) > 1.5

All numeric results are written to outputs/validation_results.json so the Part 1
report is built from the exact numbers produced here. All figures are written to
figures/.
"""

import json
import warnings

import matplotlib
matplotlib.use("Agg")  # headless backend - we only save figures, never display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import wasserstein_distance
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CONTINUOUS_COLS = [
    "transaction_amount",
    "customer_age",
    "account_balance",
    "num_transactions_30d",
    "transaction_hour",
    "distance_from_home_km",
]
CATEGORICAL_COLS = ["merchant_category", "is_fraud"]
TARGET = "is_fraud"

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# LEVEL 1 - SUMMARY STATISTICS
# ---------------------------------------------------------------------------
def level1_summary_statistics(real, synth):
    """Compare column means; flag any >10% difference, hard-stop above 20%."""
    print("\n" + "=" * 70)
    print("LEVEL 1 - SUMMARY STATISTICS")
    print("=" * 70)
    rows = []
    for col in CONTINUOUS_COLS:
        real_mean, synth_mean = real[col].mean(), synth[col].mean()
        pct = abs(real_mean - synth_mean) / abs(real_mean) * 100 if real_mean else 0.0
        flag = "OK" if pct <= 10 else ("WARN" if pct <= 20 else "STOP")
        rows.append(
            {
                "feature": col,
                "real_mean": round(real_mean, 2),
                "synthetic_mean": round(synth_mean, 2),
                "pct_diff": round(pct, 2),
                "flag": flag,
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    worst = df["pct_diff"].max()
    passed = bool((df["flag"] != "STOP").all())
    print(f"\nWorst mean difference: {worst:.2f}%  ->  "
          f"{'PASS' if passed else 'FAIL (>20% on a feature)'}")
    return {"table": rows, "worst_pct_diff": round(worst, 2), "passed": passed}


# ---------------------------------------------------------------------------
# LEVEL 2 - VISUAL DISTRIBUTIONS
# ---------------------------------------------------------------------------
def level2_visuals(real, synth):
    """KDE plots per continuous column, bar charts per categorical column, and
    side-by-side correlation heatmaps. Figures are saved to figures/."""
    print("\n" + "=" * 70)
    print("LEVEL 2 - VISUAL DISTRIBUTIONS (saving figures)")
    print("=" * 70)

    # KDE grid for continuous features
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, col in zip(axes.ravel(), CONTINUOUS_COLS):
        sns.kdeplot(real[col], ax=ax, label="Real", fill=True, alpha=0.35)
        sns.kdeplot(synth[col], ax=ax, label="Synthetic", fill=True, alpha=0.35)
        ax.set_title(col)
        ax.legend()
    fig.suptitle("Level 2 - KDE: Real vs Synthetic (continuous features)", fontsize=14)
    fig.tight_layout()
    fig.savefig("figures/level2_kde_continuous.png", dpi=120)
    plt.close(fig)

    # Bar charts for categorical proportions
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, col in zip(axes, CATEGORICAL_COLS):
        real_prop = real[col].value_counts(normalize=True).sort_index()
        synth_prop = synth[col].value_counts(normalize=True).sort_index()
        comp = pd.DataFrame({"Real": real_prop, "Synthetic": synth_prop}).fillna(0)
        comp.plot(kind="bar", ax=ax)
        ax.set_title(f"{col} - category proportions")
        ax.set_ylabel("Proportion")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Level 2 - Categorical proportions: Real vs Synthetic", fontsize=14)
    fig.tight_layout()
    fig.savefig("figures/level2_bar_categorical.png", dpi=120)
    plt.close(fig)

    # Correlation heatmaps (numeric columns) - must preserve inter-column structure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    num_cols = CONTINUOUS_COLS + [TARGET]
    for ax, (title, data) in zip(
        axes, [("Real", real[num_cols]), ("Synthetic", synth[num_cols])]
    ):
        sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, ax=ax, cbar=False)
        ax.set_title(f"{title} correlation")
    fig.suptitle("Level 2 - Correlation structure: Real vs Synthetic", fontsize=14)
    fig.tight_layout()
    fig.savefig("figures/level2_correlation_heatmaps.png", dpi=120)
    plt.close(fig)

    # Quantify how well correlations are preserved (mean abs error of corr matrix)
    corr_diff = float(np.abs(real[num_cols].corr() - synth[num_cols].corr()).mean().mean())
    print("Saved: level2_kde_continuous.png, level2_bar_categorical.png, "
          "level2_correlation_heatmaps.png")
    print(f"Mean absolute correlation-matrix error: {corr_diff:.4f}")
    return {
        "figures": [
            "figures/level2_kde_continuous.png",
            "figures/level2_bar_categorical.png",
            "figures/level2_correlation_heatmaps.png",
        ],
        "mean_abs_corr_error": round(corr_diff, 4),
    }


# ---------------------------------------------------------------------------
# LEVEL 3a - KOLMOGOROV-SMIRNOV TEST
# ---------------------------------------------------------------------------
def level3a_ks_test(real, synth):
    """Two-sample KS test per continuous column. H0: same distribution.
    p > 0.05 => fail to reject H0 => aligned (pass)."""
    print("\n" + "=" * 70)
    print("LEVEL 3a - KOLMOGOROV-SMIRNOV TEST (continuous)")
    print("=" * 70)
    rows = []
    for col in CONTINUOUS_COLS:
        ks_stat, p = stats.ks_2samp(real[col].dropna(), synth[col].dropna())
        aligned = bool(p > 0.05)
        rows.append(
            {
                "feature": col,
                "ks_stat": round(float(ks_stat), 4),
                "p_value": round(float(p), 4),
                "aligned": aligned,
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    n_pass = int(df["aligned"].sum())
    passed = n_pass == len(df)
    print(f"\nAligned: {n_pass}/{len(df)}  ->  {'PASS' if passed else 'FAIL'}")
    return {"table": rows, "n_pass": n_pass, "n_total": len(df), "passed": passed}


# ---------------------------------------------------------------------------
# LEVEL 3b - CHI-SQUARED TEST
# ---------------------------------------------------------------------------
def level3b_chi_squared(real, synth):
    """Chi-squared goodness-of-fit per categorical column. The expected counts
    come from the real proportions scaled to the synthetic sample size, so we are
    testing whether the synthetic category mix matches the real one."""
    print("\n" + "=" * 70)
    print("LEVEL 3b - CHI-SQUARED TEST (categorical)")
    print("=" * 70)
    rows = []
    for col in CATEGORICAL_COLS:
        categories = sorted(set(real[col].unique()) | set(synth[col].unique()))
        real_prop = real[col].value_counts(normalize=True).reindex(categories, fill_value=0)
        synth_counts = synth[col].value_counts().reindex(categories, fill_value=0)
        expected = real_prop * synth_counts.sum()
        # Guard against zero expected counts (chi-squared is undefined there).
        expected = expected.replace(0, 1e-6)
        chi2, p = stats.chisquare(f_obs=synth_counts.values, f_exp=expected.values)
        aligned = bool(p > 0.05)
        rows.append(
            {
                "feature": col,
                "chi2_stat": round(float(chi2), 4),
                "p_value": round(float(p), 4),
                "aligned": aligned,
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    n_pass = int(df["aligned"].sum())
    passed = n_pass == len(df)
    print(f"\nAligned: {n_pass}/{len(df)}  ->  {'PASS' if passed else 'FAIL'}")
    return {"table": rows, "n_pass": n_pass, "n_total": len(df), "passed": passed}


# ---------------------------------------------------------------------------
# LEVEL 3c - WASSERSTEIN DISTANCE
# ---------------------------------------------------------------------------
def level3c_wasserstein(real, synth):
    """Earth-mover distance per continuous column, normalised by the real
    standard deviation so features are comparable. <0.05 excellent, 0.05-0.15
    acceptable, >0.15 poor."""
    print("\n" + "=" * 70)
    print("LEVEL 3c - WASSERSTEIN DISTANCE (normalised)")
    print("=" * 70)
    rows = []
    for col in CONTINUOUS_COLS:
        dist = wasserstein_distance(real[col], synth[col])
        sigma = real[col].std()
        norm = dist / sigma if sigma else 0.0
        rating = "Excellent" if norm < 0.05 else ("Acceptable" if norm <= 0.15 else "Poor")
        rows.append(
            {
                "feature": col,
                "wasserstein_norm": round(float(norm), 4),
                "rating": rating,
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    worst = df["wasserstein_norm"].max()
    passed = bool((df["wasserstein_norm"] < 0.15).all())
    print(f"\nWorst normalised distance: {worst:.4f}  ->  "
          f"{'PASS' if passed else 'FAIL (>0.15 on a feature)'}")
    return {"table": rows, "worst_norm": round(worst, 4), "passed": passed}


# ---------------------------------------------------------------------------
# LEVEL 4 - ML UTILITY (TSTR vs TRTR)
# ---------------------------------------------------------------------------
def _encode(df):
    """One-hot encode merchant_category so the classifier can use it; keep the
    numeric columns and the target as-is."""
    X = df[CONTINUOUS_COLS].copy()
    X = pd.concat([X, pd.get_dummies(df["merchant_category"], prefix="mcat")], axis=1)
    y = df[TARGET].astype(int)
    return X, y


def level4_tstr(real, synth, label="synthetic"):
    """TRTR (train real / test real) is the gold-standard baseline. TSTR
    (train synthetic / test real) simulates production. We hold out a real test
    set once and score both models on it. AUC gap < 0.05 => acceptable."""
    print("\n" + "=" * 70)
    print(f"LEVEL 4 - ML UTILITY: TSTR vs TRTR ({label})")
    print("=" * 70)

    X_real, y_real = _encode(real)
    X_synth, y_synth = _encode(synth)
    # Align synthetic columns to the real feature space (categories may differ).
    X_synth = X_synth.reindex(columns=X_real.columns, fill_value=0)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_real, y_real, test_size=0.3, random_state=RANDOM_SEED, stratify=y_real
    )

    # TRTR baseline
    trtr = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    trtr.fit(X_tr, y_tr)
    trtr_auc = roc_auc_score(y_te, trtr.predict_proba(X_te)[:, 1])

    # TSTR - train on synthetic, test on the same real hold-out
    tstr = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    tstr.fit(X_synth, y_synth)
    tstr_auc = roc_auc_score(y_te, tstr.predict_proba(X_te)[:, 1])

    gap = abs(tstr_auc - trtr_auc)
    verdict = (
        "Fully equivalent" if gap < 0.02
        else "Acceptable with monitoring" if gap <= 0.05
        else "Investigate before deployment"
    )
    passed = bool(gap < 0.05)
    print(f"TRTR AUC : {trtr_auc:.4f}")
    print(f"TSTR AUC : {tstr_auc:.4f}")
    print(f"AUC gap  : {gap:.4f}  ->  {verdict}")
    return {
        "trtr_auc": round(float(trtr_auc), 4),
        "tstr_auc": round(float(tstr_auc), 4),
        "auc_gap": round(float(gap), 4),
        "verdict": verdict,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# LEVEL 5 - PRIVACY (DNNR)
# ---------------------------------------------------------------------------
def level5_dnnr(real, synth):
    """Distance to Nearest Neighbour Ratio. If synthetic rows sit right on top of
    real rows the model has memorised individuals. We compare the median
    synthetic->real distance against the median real->real distance in a scaled
    feature space. DNNR > 1.5 => well separated (privacy preserved)."""
    print("\n" + "=" * 70)
    print("LEVEL 5 - PRIVACY: DNNR")
    print("=" * 70)

    X_real, _ = _encode(real)
    X_synth, _ = _encode(synth)
    X_synth = X_synth.reindex(columns=X_real.columns, fill_value=0)

    scaler = StandardScaler().fit(X_real)
    real_s = scaler.transform(X_real)
    synth_s = scaler.transform(X_synth)

    # d_RR: nearest real-to-real distance, excluding the point itself (k=2 -> [1]).
    nn_real = NearestNeighbors(n_neighbors=2).fit(real_s)
    d_rr, _ = nn_real.kneighbors(real_s)
    d_rr = d_rr[:, 1]

    # d_SR: nearest synthetic-to-real distance.
    nn_for_synth = NearestNeighbors(n_neighbors=1).fit(real_s)
    d_sr, _ = nn_for_synth.kneighbors(synth_s)
    d_sr = d_sr[:, 0]

    dnnr = float(np.median(d_sr) / np.median(d_rr))
    interp = (
        "Privacy preserved" if dnnr > 1.5
        else "Borderline" if dnnr > 1.0
        else "Memorisation risk"
    )
    passed = bool(dnnr > 1.5)
    print(f"median d_SR : {np.median(d_sr):.4f}")
    print(f"median d_RR : {np.median(d_rr):.4f}")
    print(f"DNNR        : {dnnr:.4f}  ->  {interp}")
    return {
        "median_d_sr": round(float(np.median(d_sr)), 4),
        "median_d_rr": round(float(np.median(d_rr)), 4),
        "dnnr": round(dnnr, 4),
        "interpretation": interp,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# ORCHESTRATION
# ---------------------------------------------------------------------------
def validate(real, synth, name, run_visuals=False):
    """Run all five levels against one synthetic dataset and collect results."""
    results = {}
    results["level1"] = level1_summary_statistics(real, synth)
    if run_visuals:
        results["level2"] = level2_visuals(real, synth)
    results["level3a"] = level3a_ks_test(real, synth)
    results["level3b"] = level3b_chi_squared(real, synth)
    results["level3c"] = level3c_wasserstein(real, synth)
    results["level4"] = level4_tstr(real, synth, label=name)
    results["level5"] = level5_dnnr(real, synth)

    checks = [
        results["level1"]["passed"],
        results["level3a"]["passed"],
        results["level3b"]["passed"],
        results["level3c"]["passed"],
        results["level4"]["passed"],
        results["level5"]["passed"],
    ]
    results["overall_pass"] = bool(all(checks))
    results["n_passed"] = int(sum(checks))
    results["n_checks"] = len(checks)
    return results


def main():
    real = pd.read_csv("data/real_data.csv")
    synth = pd.read_csv("data/synthetic_data.csv")
    corrupted = pd.read_csv("data/corrupted_synthetic.csv")

    print("#" * 70)
    print("# VALIDATING THE HONEST CTGAN SYNTHETIC DATASET")
    print("#" * 70)
    honest = validate(real, synth, "CTGAN synthetic", run_visuals=True)

    print("\n\n" + "#" * 70)
    print("# VALIDATING THE CORRUPTED SYNTHETIC DATASET (should fail)")
    print("#" * 70)
    corrupt = validate(real, corrupted, "corrupted synthetic", run_visuals=False)

    out = {
        "dataset_sizes": {
            "real": len(real),
            "synthetic": len(synth),
            "corrupted": len(corrupted),
        },
        "fraud_rates": {
            "real": round(float(real["is_fraud"].mean()), 4),
            "synthetic": round(float(synth["is_fraud"].mean()), 4),
            "corrupted": round(float(corrupted["is_fraud"].mean()), 4),
        },
        "honest_ctgan": honest,
        "corrupted": corrupt,
    }
    with open("outputs/validation_results.json", "w") as fh:
        json.dump(out, fh, indent=2)

    print("\n\n" + "=" * 70)
    print("FINAL RECOMMENDATION")
    print("=" * 70)
    print(f"Honest CTGAN synthetic : {honest['n_passed']}/{honest['n_checks']} "
          f"checks passed  ->  {'APPROVE' if honest['overall_pass'] else 'REJECT'}")
    print(f"Corrupted synthetic    : {corrupt['n_passed']}/{corrupt['n_checks']} "
          f"checks passed  ->  {'APPROVE' if corrupt['overall_pass'] else 'REJECT'}")
    print("\nResults written to outputs/validation_results.json")


if __name__ == "__main__":
    main()
