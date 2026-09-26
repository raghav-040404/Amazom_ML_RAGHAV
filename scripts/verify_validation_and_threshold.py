"""
Verification of Validation Metric & Fine-Grained Threshold Sweep.
Amazon ML Challenge 2026 - Business Entity Resolution.
Strictly processes validation data originating from `datasett11`.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import OUTPUT_DIR, ARTIFACTS_DIR
from src.evaluate import compute_entity_f05, compute_macro_f05, compute_pairwise_metrics


def run_validation_verification():
    print("=" * 80)
    print("STEP 1 & 2: INDEPENDENT VALIDATION RE-CALCULATION & FINE-GRAINED THRESHOLD SWEEP")
    print("=" * 80)

    # Load feature dataset from output/training_features.parquet
    feat_path = os.path.join(OUTPUT_DIR, "training_features.parquet")
    print(f"Loading cached features from {feat_path}...")
    df_feat = pd.read_parquet(feat_path)
    
    # Load ground truth report to retrieve the validation S1 entities
    report_path = os.path.join(OUTPUT_DIR, "phase3_modeling_report.json")
    with open(report_path, "r", encoding="utf-8") as f:
        rep = json.load(f)

    # Load trained model
    model_path = os.path.join(ARTIFACTS_DIR, "best_tree_model.pkl")
    print(f"Loading trained tree model from {model_path}...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Reconstruct the exact validation split (20% of S1 entities)
    from src.config import TRAIN_GROUND_TRUTH_PATH
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH_PATH, sep="\t", dtype=str, keep_default_na=False)
    
    # Recreate the deterministic sample and split
    rng = np.random.RandomState(42)
    sampled_indices = rng.choice(len(gt_df), size=12_000, replace=False)
    sampled_gt_df = gt_df.iloc[sampled_indices].reset_index(drop=True)
    
    unique_s1 = sampled_gt_df["source1_entity_id"].values.copy()
    rng.shuffle(unique_s1)
    
    n_val_s1 = int(len(unique_s1) * 0.20)
    val_s1_set = set(unique_s1[:n_val_s1])
    
    val_ground_truth = {}
    for _, row in sampled_gt_df.iterrows():
        s1 = row["source1_entity_id"]
        if s1 in val_s1_set:
            raw = row["matched_entity_ids"].strip()
            val_ground_truth[s1] = {m.strip() for m in raw.split(",") if m.strip()} if raw else set()

    # Filter features for validation set
    val_mask = df_feat["source1_entity_id"].isin(val_s1_set)
    val_df = df_feat[val_mask].reset_index(drop=True)
    
    feature_cols = [c for c in val_df.columns if c not in ("source1_entity_id", "candidate_entity_id", "label")]
    X_val = val_df[feature_cols].values
    y_val = val_df["label"].values
    
    val_probs = model.predict_proba(X_val)[:, 1]
    val_s1_ids = val_df["source1_entity_id"].values
    val_cand_ids = val_df["candidate_entity_id"].values

    print(f"Validation Set: {len(val_s1_set):,} S1 entities | {len(X_val):,} candidate pairs | {np.sum(y_val):,} true matches.")

    # Fine-grained threshold sweep [0.90 to 0.99]
    fine_thresholds = [0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
    
    sweep_results = []
    best_thresh = 0.95
    best_macro_f05 = -1.0
    best_pred_map = {}

    print("\n" + "-" * 75)
    print(f"{'Threshold':<10}{'Macro F0.5':<14}{'Macro Prec':<14}{'Macro Rec':<14}{'Pairwise F0.5':<14}{'Singletons Acc'}")
    print("-" * 75)

    for thresh in fine_thresholds:
        pred_map = {s1: set() for s1 in val_s1_set}
        for s1, cid, p in zip(val_s1_ids, val_cand_ids, val_probs):
            if p >= thresh:
                pred_map[s1].add(cid)
                
        macro_res = compute_macro_f05(pred_map, val_ground_truth, beta=0.5)
        pair_res = compute_pairwise_metrics(y_val, (val_probs >= thresh).astype(int), beta=0.5)
        
        row = {
            "threshold": thresh,
            "macro_f05": macro_res["macro_f05"],
            "macro_precision": macro_res["macro_precision"],
            "macro_recall": macro_res["macro_recall"],
            "pairwise_f05": pair_res["f05"],
            "pairwise_precision": pair_res["precision"],
            "pairwise_recall": pair_res["recall"],
            "singletons_total": macro_res["singletons_total"],
            "singletons_correct": macro_res["singletons_correct"],
            "singleton_accuracy": macro_res["singleton_accuracy"],
        }
        sweep_results.append(row)
        
        print(f"{thresh:<10.2f}{row['macro_f05']:<14.4f}{row['macro_precision']:<14.4f}{row['macro_recall']:<14.4f}{row['pairwise_f05']:<14.4f}{row['singleton_accuracy']:.2f}%")

        if row["macro_f05"] > best_macro_f05:
            best_macro_f05 = row["macro_f05"]
            best_thresh = thresh
            best_pred_map = pred_map

    print("-" * 75)
    print(f"Optimal Verified Decision Threshold: {best_thresh:.2f} (Macro F0.5 = {best_macro_f05:.4f})")

    # Group-level breakdown at optimal threshold
    # 1. Singletons (0 true matches)
    # 2. Single match (1 true match)
    # 3. Multi-match (2+ true matches)
    print("\n" + "=" * 75)
    print("STEP 3: DETAILED SINGLETON / SINGLE / MULTI-MATCH BREAKDOWN AT OPTIMAL THRESHOLD")
    print("=" * 75)

    groups = {"singletons_0_matches": [], "single_1_match": [], "multi_2plus_matches": []}

    for s1, true_set in val_ground_truth.items():
        pred_set = best_pred_map.get(s1, set())
        p, r, f = compute_entity_f05(pred_set, true_set, beta=0.5)
        
        is_exact = (pred_set == true_set)
        has_fp = len(pred_set - true_set) > 0
        
        entry = {
            "s1_id": s1,
            "true_count": len(true_set),
            "pred_count": len(pred_set),
            "precision": p,
            "recall": r,
            "f05": f,
            "exact_match": is_exact,
            "false_positive": has_fp,
        }
        
        if len(true_set) == 0:
            groups["singletons_0_matches"].append(entry)
        elif len(true_set) == 1:
            groups["single_1_match"].append(entry)
        else:
            groups["multi_2plus_matches"].append(entry)

    group_stats = {}
    for g_name, items in groups.items():
        n = len(items)
        if n == 0:
            continue
        acc = sum(1 for x in items if x["exact_match"]) / n * 100
        fp_rate = sum(1 for x in items if x["false_positive"]) / n * 100
        mean_f05 = np.mean([x["f05"] for x in items])
        mean_prec = np.mean([x["precision"] for x in items])
        mean_rec = np.mean([x["recall"] for x in items])
        
        group_stats[g_name] = {
            "count_s1_entities": n,
            "percentage_of_val_set": round((n / len(val_ground_truth)) * 100, 2),
            "exact_prediction_accuracy_pct": round(acc, 2),
            "false_positive_rate_pct": round(fp_rate, 2),
            "mean_f05": round(float(mean_f05), 4),
            "mean_precision": round(float(mean_prec), 4),
            "mean_recall": round(float(mean_rec), 4),
        }
        print(f"\nGroup: {g_name.upper()} ({n:,} S1 entities, {group_stats[g_name]['percentage_of_val_set']}%)")
        print(f"  Exact Accuracy:       {acc:.2f}%")
        print(f"  False Positive Rate:  {fp_rate:.2f}%")
        print(f"  Mean F0.5 Score:      {mean_f05:.4f}")
        print(f"  Mean Precision:       {mean_prec:.4f}")
        print(f"  Mean Recall:          {mean_rec:.4f}")

    verification_report = {
        "dataset_source": "datasett11/student_resource/dataset",
        "optimal_threshold": best_thresh,
        "best_macro_f05": best_macro_f05,
        "fine_threshold_sweep": sweep_results,
        "group_breakdown": group_stats,
    }

    out_json = os.path.join(OUTPUT_DIR, "validation_verification_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(verification_report, f, indent=2)
    print(f"\nSaved Validation Verification Report to {out_json}")
    return verification_report


if __name__ == "__main__":
    run_validation_verification()
