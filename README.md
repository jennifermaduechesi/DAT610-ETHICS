# DAT610 Assignment 1 — Synthetic Data Validation for Fraud Detection

**Maduechesi Chidiebere Jennifer · Matric No. 25120133019**
MSc Data Science · Pan-Atlantic University · School of Science and Technology

## What this project does

It builds a "real" FintechPay fraud-detection dataset, generates a synthetic
copy with a **Conditional Tabular GAN (CTGAN)** using the SDV library, and then
runs the **five-level validation framework** from the lecture to decide whether
the synthetic data is safe and statistically valid for release into production.

A second, deliberately **corrupted** synthetic dataset is also generated so the
validation framework can be shown catching real failures (mirrors the lecture's
"Find the Failures" lab).

## How to run

```bash
pip install -r requirements.txt

python scripts/01_generate_real_data.py   # -> data/real_data.csv
python scripts/02_train_ctgan.py          # -> data/synthetic_data.csv, data/corrupted_synthetic.csv
python scripts/03_validate.py             # -> outputs/, figures/
```

## Layout

```
scripts/
  01_generate_real_data.py   Build the real reference dataset (FintechPay schema)
  02_train_ctgan.py          Train CTGAN, sample honest + corrupted synthetic sets
  03_validate.py             Run the five-level validation framework
data/                        Generated CSV datasets
figures/                     KDE / bar / correlation-heatmap PNGs (Level 2)
outputs/                     metadata.json + validation_results.json
report/                      Part 1 report (Word) + Synthetic Data Card
```

## The five-level framework

| Level | Test | Pass criterion |
|-------|------|----------------|
| 1 | Summary statistics | means within 10% of real |
| 2 | KDE + correlation heatmaps | visual alignment |
| 3a | Kolmogorov–Smirnov (continuous) | p > 0.05 per column |
| 3b | Chi-squared (categorical) | p > 0.05 per column |
| 3c | Wasserstein distance (normalised) | < 0.15 per feature |
| 4 | TSTR vs TRTR (ML utility) | AUC gap < 0.05 |
| 5 | Privacy DNNR | > 1.5 |

All numbers in the report are produced by `scripts/03_validate.py` and stored in
`outputs/validation_results.json`. A fixed random seed (42) makes the whole
pipeline reproducible.
