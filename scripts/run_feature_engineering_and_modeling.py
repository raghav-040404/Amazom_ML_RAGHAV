"""
Feature Engineering, Model Training, Threshold Optimization & Evaluation.
Amazon ML Challenge 2026 - Business Entity Resolution.
Strictly processes candidate pairs and records originating from `datasett11`.
"""

import os
import sys
import json
import time
from collections import defaultdict
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    OUTPUT_DIR,
    CONFIG,
)
from src.normalize import normalize_dataframe
from src.blocking import MultiStrategyBlocker
from src.features import EntityFeatureExtractor
from src.train import train_and_evaluate_models


def run_pipeline(
    n_s1_sample=12_000,
    background_targets_sample=250_000,
    random_seed=42,
):
    print("=" * 80)
    print("PHASE 3: FEATURE ENGINEERING, MODELING & F0.5 OPTIMIZATION")
    print("=" * 80)
    start_time = time.time()

    # 1. Load Ground Truth
    print("\n[Step 1] Loading ground truth from datasett11...")
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH_PATH, sep="\t", dtype=str, keep_default_na=False)
    
    rng = np.random.RandomState(random_seed)
    sampled_indices = rng.choice(len(gt_df), size=min(n_s1_sample, len(gt_df)), replace=False)
    sampled_gt_df = gt_df.iloc[sampled_indices].reset_index(drop=True)
    
    s1_eval_ids = set(sampled_gt_df["source1_entity_id"].values)
    ground_truth_map = {}
    target_match_ids = set()
    
    for _, row in sampled_gt_df.iterrows():
        s1 = row["source1_entity_id"]
        raw = row["matched_entity_ids"].strip()
        if raw:
            mids = {m.strip() for m in raw.split(",") if m.strip()}
            ground_truth_map[s1] = mids
            target_match_ids.update(mids)
        else:
            ground_truth_map[s1] = set()

    print(f"Sampled {len(s1_eval_ids):,} S1 entities containing {len(target_match_ids):,} true match links.")

    # 2. Load & Normalize Source 1 Records
    print("\n[Step 2] Loading & normalizing Source 1 records...")
    s1_rows = []
    for chunk in pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False):
        matched = chunk[chunk["entity_id"].isin(s1_eval_ids)]
        if len(matched) > 0:
            s1_rows.append(matched)
    s1_df = pd.concat(s1_rows, ignore_index=True)
    s1_df = normalize_dataframe(s1_df).set_index("entity_id")
    print(f"Loaded {len(s1_df):,} S1 records.")

    # 3. Load & Normalize Target Records (S2 & S3)
    print("\n[Step 3] Loading & indexing target records (S2 and S3)...")
    target_records_dict = {}
    blocker = MultiStrategyBlocker(max_candidates_per_entity=50)

    # S2 Loading
    s2_bg = 0
    for chunk in pd.read_csv(TRAIN_SOURCE2_PATH, sep="\t", chunksize=250_000, dtype=str, keep_default_na=False):
        is_target = chunk["entity_id"].isin(target_match_ids)
        if s2_bg < background_targets_sample:
            take = min(background_targets_sample - s2_bg, len(chunk))
            is_target.iloc[:take] = True
            s2_bg += take
        sel = chunk[is_target]
        if len(sel) > 0:
            sel_norm = normalize_dataframe(sel)
            blocker.index_target_records(sel_norm)
            for _, r in sel_norm.iterrows():
                target_records_dict[r["entity_id"]] = r.to_dict()

    # S3 Loading
    s3_bg = 0
    for chunk in pd.read_csv(TRAIN_SOURCE3_PATH, sep="\t", chunksize=250_000, dtype=str, keep_default_na=False):
        is_target = chunk["entity_id"].isin(target_match_ids)
        if s3_bg < background_targets_sample:
            take = min(background_targets_sample - s3_bg, len(chunk))
            is_target.iloc[:take] = True
            s3_bg += take
        sel = chunk[is_target]
        if len(sel) > 0:
            sel_norm = normalize_dataframe(sel)
            blocker.index_target_records(sel_norm)
            for _, r in sel_norm.iterrows():
                target_records_dict[r["entity_id"]] = r.to_dict()

    print(f"Total target records indexed: {len(target_records_dict):,}")

    # 4. Generate Candidates
    print("\n[Step 4] Generating candidate pairs...")
    candidate_map = blocker.generate_candidates_batch(s1_df.reset_index())
    
    total_candidate_pairs = sum(len(c) for c in candidate_map.values())
    s1_to_s2_cands = 0
    s1_to_s3_cands = 0
    
    candidate_list = []
    for s1_id, cand_set in candidate_map.items():
        for cid in cand_set:
            if cid.startswith("S2-"):
                s1_to_s2_cands += 1
            elif cid.startswith("S3-"):
                s1_to_s3_cands += 1
            candidate_list.append((s1_id, cid))

    print(f"Total candidate pairs: {total_candidate_pairs:,} (S1->S2: {s1_to_s2_cands:,}, S1->S3: {s1_to_s3_cands:,})")

    # 5. Feature Extraction & Label Assignment
    print("\n[Step 5] Extracting matching similarity features & labels...")
    extractor = EntityFeatureExtractor()
    feature_rows = []
    labels = []
    pair_s1_ids = []
    pair_cand_ids = []

    pos_count = 0
    neg_count = 0

    feat_start = time.time()
    for s1_id, cid in candidate_list:
        if s1_id not in s1_df.index or cid not in target_records_dict:
            continue

        s1_rec = s1_df.loc[s1_id]
        cand_rec = target_records_dict[cid]

        feats = extractor.extract_pair_features(
            s1_orig_name=s1_rec["business_name"],
            s1_norm_name=s1_rec["business_name_normalized"],
            s1_orig_addr=s1_rec["business_address"],
            s1_norm_addr=s1_rec["business_address_normalized"],
            s1_country=s1_rec["country_normalized"],
            cand_orig_name=cand_rec["business_name"],
            cand_norm_name=cand_rec["business_name_normalized"],
            cand_orig_addr=cand_rec["business_address"],
            cand_norm_addr=cand_rec["business_address_normalized"],
            cand_country=cand_rec["country_normalized"],
            cand_source_prefix="S3" if cid.startswith("S3-") else "S2",
        )

        # True Match Label
        is_match = 1 if cid in ground_truth_map.get(s1_id, set()) else 0
        if is_match == 1:
            pos_count += 1
        else:
            neg_count += 1

        feature_rows.append(feats)
        labels.append(is_match)
        pair_s1_ids.append(s1_id)
        pair_cand_ids.append(cid)

    feat_time = time.time() - feat_start
    print(f"Extracted features for {len(feature_rows):,} pairs in {feat_time:.2f}s.")
    print(f"Class distribution: Positives={pos_count:,} ({pos_count/len(labels)*100:.2f}%), Negatives={neg_count:,} ({neg_count/len(labels)*100:.2f}%)")

    X = np.array(feature_rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # Verify no NaN / Inf
    assert np.all(np.isfinite(X)), "Features contain non-finite values!"
    print(f"Feature matrix shape: {X.shape} | All values verified finite and non-null.")

    # Save features parquet
    features_df = pd.DataFrame(X, columns=EntityFeatureExtractor.FEATURE_NAMES)
    features_df["source1_entity_id"] = pair_s1_ids
    features_df["candidate_entity_id"] = pair_cand_ids
    features_df["label"] = y
    
    feat_parquet_path = os.path.join(OUTPUT_DIR, "training_features.parquet")
    features_df.to_parquet(feat_parquet_path, index=False)
    print(f"Saved feature dataset to {feat_parquet_path}")

    # 6. S1-Grouped Train / Validation Split (80% Train / 20% Val)
    print("\n[Step 6] Creating S1-level grouped train/validation split...")
    unique_s1 = np.array(list(s1_eval_ids))
    rng.shuffle(unique_s1)
    
    n_val_s1 = int(len(unique_s1) * 0.20)
    val_s1_set = set(unique_s1[:n_val_s1])
    train_s1_set = set(unique_s1[n_val_s1:])
    
    train_mask = np.array([s in train_s1_set for s in pair_s1_ids])
    val_mask = np.array([s in val_s1_set for s in pair_s1_ids])

    X_train, y_train = X[train_mask], y[train_mask]
    train_s1 = [pair_s1_ids[i] for i in range(len(pair_s1_ids)) if train_mask[i]]

    X_val, y_val = X[val_mask], y[val_mask]
    val_s1 = [pair_s1_ids[i] for i in range(len(pair_s1_ids)) if val_mask[i]]
    val_cand = [pair_cand_ids[i] for i in range(len(pair_cand_ids)) if val_mask[i]]

    val_gt_subset = {s1: ground_truth_map[s1] for s1 in val_s1_set}

    print(f"Train split: {len(train_s1_set):,} S1 entities ({len(X_train):,} candidate pairs, {np.sum(y_train):,} positives)")
    print(f"Validation split: {len(val_s1_set):,} S1 entities ({len(X_val):,} candidate pairs, {np.sum(y_val):,} positives)")

    # 7. Model Training & Comparison
    print("\n[Step 7] Training models and sweeping thresholds for Macro F0.5...")
    model_results = train_and_evaluate_models(
        X_train=X_train,
        y_train=y_train,
        train_s1_ids=train_s1,
        X_val=X_val,
        y_val=y_val,
        val_s1_ids=val_s1,
        val_cand_ids=val_cand,
        val_ground_truth=val_gt_subset,
    )

    # 8. Compile Final Report
    pipeline_report = {
        "dataset_source": "datasett11/student_resource/dataset",
        "total_candidate_pairs": total_candidate_pairs,
        "s1_to_s2_candidates": s1_to_s2_cands,
        "s1_to_s3_candidates": s1_to_s3_cands,
        "positive_pairs": pos_count,
        "negative_pairs": neg_count,
        "positive_pct": round(pos_count / len(labels) * 100, 2),
        "negative_pct": round(neg_count / len(labels) * 100, 2),
        "total_features": len(EntityFeatureExtractor.FEATURE_NAMES),
        "feature_names": EntityFeatureExtractor.FEATURE_NAMES,
        "train_candidate_pairs": int(len(X_train)),
        "validation_candidate_pairs": int(len(X_val)),
        "validation_s1_entities": len(val_s1_set),
        "models": model_results,
        "total_pipeline_runtime_seconds": round(time.time() - start_time, 2),
    }

    report_path = os.path.join(OUTPUT_DIR, "phase3_modeling_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_report, f, indent=2)
    print(f"\nSaved Phase 3 Modeling Report to {report_path}")
    return pipeline_report


if __name__ == "__main__":
    run_pipeline()
