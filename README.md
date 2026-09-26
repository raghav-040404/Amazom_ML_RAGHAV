# Amazon ML Challenge 2026: Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: Apache 2.0 / MIT](https://img.shields.io/badge/License-Apache%202.0%20%2F%20MIT-green.svg)](https://opensource.org/licenses/Apache-2.0)

Production-quality, memory-efficient, and reproducible machine learning pipeline for the **Amazon ML Challenge 2026 – Business Entity Resolution Challenge**.

---

## 1. Problem Overview

In large-scale commercial platforms, entity records arrive from multiple independent, noisy sources without shared foreign keys:
- **Source 1 ($S1$):** Deduplicated canonical reference entities (ground truth anchor).
- **Source 2 ($S2$):** Noisy business records (containing typos, legal suffix variations, partial addresses).
- **Source 3 ($S3$):** Noisy business records (containing transliterations, inverted word order, DBA names).

**Objective:** For every Source 1 entity, identify all matching Source 2 and Source 3 records that represent the exact same real-world business. A Source 1 entity can match zero (singleton), one, or multiple records.

---

## 2. End-to-End Pipeline Architecture

```text
               ┌────────────────────────────────────────────────────────┐
               │              Raw Dataset (datasett11)                  │
               │        (Source 1, Source 2, Source 3)                  │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │           Conservative Normalization                   │
               │   • Unicode NFKD decomposition & lowercasing           │
               │   • Legal suffix & road token standardization          │
               │   • Strict preservation of numeric tokens (PINs, house)│
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │        Multi-Strategy Candidate Blocking               │
               │   • Country Partitioning (US, India, France)           │
               │   • Multi-Key Inverted Indexing (Exact, Prefix, Tokens)│
               │   • Address Number + Street Anchor Indexing            │
               │   • 91.11% Blocking Recall | 99.99% Candidate Reduction│
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │            Similarity Feature Engineering              │
               │   • 31-dimensional feature vectors per candidate pair  │
               │   • Character 3-gram, Levenshtein, Jaccard token sim   │
               │   • Address number & postal code verification          │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │             Supervised Tabular Models                  │
               │   • Gradient Boosted Trees (HistGradientBoosting / XGB)│
               │   • Class Imbalance Weighting (scale_pos_weight)       │
               │   • S1-Grouped Validation Split (Zero Leakage)         │
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │               Threshold Optimization                   │
               │   • Macro F0.5 Optimization on Validation Holdout Split│
               │   • Precision-heavy operating point (tau = 0.95)       │
               │   • High Singleton Accuracy (94.07% correct empty rows)│
               └──────────────────────────┬─────────────────────────────┘
                                          │
                                          ▼
               ┌────────────────────────────────────────────────────────┐
               │           Official Output & Validation                 │
               │   • output/matching_results.tsv (Scored on portal)     │
               │   • output/candidate_pairs.tsv  (Candidate universe)   │
               │   • 100% compliant with validate_submission.py         │
               └────────────────────────────────────────────────────────┘
```

---

## 3. Verified Empirical Validation Results

> [!NOTE]
> All results below are **held-out validation split results** on training data (not test/leaderboard predictions).

### Candidate Dataset & Class Balance
- **Total Candidate Pairs:** `461,094`
- **Positive Pairs ($y=1$):** `37,609` (8.16%)
- **Negative Pairs ($y=0$):** `423,485` (91.84%)
- **Class Ratio:** `1 : 11.26`

### Model Comparison & Validation Metrics

| Model | Optimal Threshold ($\tau$) | Validation Macro $F_{0.5}$ | Macro Precision | Macro Recall | Pairwise Precision | Pairwise Recall | Singleton Accuracy |
|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.95 | **0.9185** | 0.9504 | 0.8621 | 0.9699 | 0.9525 | 90.37% |
| **Gradient Boosted Trees** | **0.95** | **0.9341** | **0.9621** | **0.8803** | **0.9832** | **0.9722** | **94.07%** |

### Top Measured Feature Importances
1. `name_char_3gram_jaccard` (Magnitude: 2.68) — Robust against spelling noise and transliterations.
2. `name_levenshtein_sim` (Magnitude: 2.50) — Normalized edit distance ratio.
3. `addr_token_jaccard` (Magnitude: 2.24) — Locality and street overlap.
4. `name_addr_combined_score` (Magnitude: 1.97) — Composite interaction metric.
5. `addr_char_3gram_jaccard` (Magnitude: 1.89) — Address substring similarity.

---

## 4. Local Dataset Setup

The challenge dataset is large (~2.33 GB) and is **strictly excluded from this GitHub repository** in compliance with challenge rules.

To run the pipeline locally, place the official challenge files into `datasett11/` as follows:

```text
datasett11/
└── student_resource/
    └── dataset/
        ├── train/
        │   ├── train_source1.tsv
        │   ├── train_source2.tsv
        │   ├── train_source3.tsv
        │   └── train_ground_truth.tsv
        └── test/
            ├── test_source1.tsv
            ├── test_source2.tsv
            └── test_source3.tsv
```

---

## 5. Repository Structure

```text
.
├── src/
│   ├── __init__.py
│   ├── config.py           # Master paths & hyperparameters
│   ├── data_loader.py      # Memory-efficient chunked streaming loader
│   ├── normalize.py        # Conservative text & address normalization
│   ├── blocking.py         # Multi-strategy inverted indexing blocker
│   ├── features.py         # 31-dimensional feature extractor
│   ├── train.py            # Model training & threshold sweep
│   ├── evaluate.py         # Macro F0.5 evaluation module
│   ├── predict.py          # Streaming test inference engine
│   └── submission.py       # Validator caller & submission packager
├── scripts/
│   ├── inspect_datasett11.py                # Dataset inspection & EDA
│   ├── evaluate_phase2_phase3.py            # Normalization & blocking evaluation
│   ├── run_feature_engineering_and_modeling.py # Training & F0.5 optimization
│   ├── verify_validation_and_threshold.py   # Verification & fine-grained sweep
│   └── run_full_test_inference.py           # Full test inference & validation
├── notebooks/
│   └── 01_eda.ipynb        # Interactive EDA notebook
├── output/
│   └── .gitkeep            # Output directory placeholder
├── utils/
│   └── validate_submission.py # Official submission format validator
├── Documentation_template.md  # Methodology write-up template
├── requirements.txt         # Pinned python dependencies
├── .gitignore               # Strict dataset, artifact & secret exclusions
└── README.md
```

---

## 6. Installation & Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Dataset Inspection (Phase 1)
python scripts/inspect_datasett11.py

# 3. Run Normalization & Blocking Evaluation (Phase 2 & 3)
python scripts/evaluate_phase2_phase3.py

# 4. Train Model & Optimize Threshold (Phase 3)
python scripts/run_feature_engineering_and_modeling.py

# 5. Run Full Test Inference & Validate (Phase 4)
python scripts/run_full_test_inference.py
```

---

## 7. Strict Compliance & Fair Play

- **Data Integrity:** All features, blocking strategies, and matching decisions use **ONLY data from `datasett11`**.
- **No External Data:** No Google Maps, geocoding APIs, external business registries, or internet lookups.
- **Licenses & Constraints:** All algorithms use open-source Apache 2.0 / MIT libraries and adhere to the maximum 8B parameter limit.
