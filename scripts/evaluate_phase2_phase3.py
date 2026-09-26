"""
Phase 2 (Normalization) and Phase 3 (Blocking) Pipeline & Evaluation.
Amazon ML Challenge 2026 - Business Entity Resolution.
Strictly uses data originating from `datasett11`.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    OUTPUT_DIR,
    CANDIDATE_PAIRS_PATH,
)
from src.normalize import (
    normalize_dataframe,
    analyze_normalization_quality,
)
from src.blocking import (
    MultiStrategyBlocker,
    evaluate_blocking_recall,
    save_candidate_pairs_tsv,
)


def run_phase2_normalization_benchmarks(sample_size=100_000):
    print("\n" + "=" * 70)
    print("PHASE 2: NORMALIZATION QUALITY BENCHMARKS")
    print("=" * 70)
    
    norm_reports = {}
    for name, path in [("train_source1", TRAIN_SOURCE1_PATH),
                       ("train_source2", TRAIN_SOURCE2_PATH),
                       ("train_source3", TRAIN_SOURCE3_PATH)]:
        print(f"Sampling {sample_size:,} records from {name}...")
        df_sample = pd.read_csv(path, sep="\t", nrows=sample_size, dtype=str, keep_default_na=False)
        df_norm = normalize_dataframe(df_sample)
        quality = analyze_normalization_quality(df_sample, df_norm)
        norm_reports[name] = quality
        
        print(f"[{name}] Total: {quality['total_records']:,} | Names changed: {quality['names_changed_pct']}% | Addrs changed: {quality['addrs_changed_pct']}%")
        print(f"[{name}] Unique Names: {quality['unique_original_names']:,} -> {quality['unique_normalized_names']:,} | Name collisions: {quality['colliding_normalized_name_keys']:,}")
        print(f"[{name}] Empty Names: {quality['empty_normalized_names_pct']}% | Empty Addrs: {quality['empty_normalized_addrs_pct']}%")
        
    return norm_reports


def run_phase3_blocking_evaluation(
    eval_s1_count=10_000,
    background_sample_per_file=200_000,
    random_seed=42,
):
    print("\n" + "=" * 70)
    print("PHASE 3: BLOCKING & CANDIDATE GENERATION EVALUATION")
    print("=" * 70)
    start_time = time.time()
    
    # 1. Load Ground Truth
    print("Loading ground truth mapping...")
    gt_df = pd.read_csv(TRAIN_GROUND_TRUTH_PATH, sep="\t", dtype=str, keep_default_na=False)
    
    rng = np.random.RandomState(random_seed)
    sampled_indices = rng.choice(len(gt_df), size=min(eval_s1_count, len(gt_df)), replace=False)
    sampled_gt_df = gt_df.iloc[sampled_indices].reset_index(drop=True)
    
    eval_s1_ids = set(sampled_gt_df["source1_entity_id"].values)
    
    eval_ground_truth = {}
    target_match_ids = set()
    
    for _, row in sampled_gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        raw = row["matched_entity_ids"].strip()
        if raw:
            mids = {m.strip() for m in raw.split(",") if m.strip()}
            eval_ground_truth[s1_id] = mids
            target_match_ids.update(mids)
        else:
            eval_ground_truth[s1_id] = set()
            
    print(f"Selected {len(eval_s1_ids):,} evaluation S1 entities with {len(target_match_ids):,} true match targets.")
    
    # 2. Load and Normalize S1 Evaluation Records
    print("Loading and normalizing Source 1 evaluation records...")
    s1_records = []
    for chunk in pd.read_csv(TRAIN_SOURCE1_PATH, sep="\t", chunksize=200_000, dtype=str, keep_default_na=False):
        matched = chunk[chunk["entity_id"].isin(eval_s1_ids)]
        if len(matched) > 0:
            s1_records.append(matched)
            
    s1_eval_df = pd.concat(s1_records, ignore_index=True)
    s1_eval_norm_df = normalize_dataframe(s1_eval_df)
    print(f"Source 1 evaluation records ready: {len(s1_eval_norm_df):,}")
    
    # 3. Stream & Index Target Records across all chunks
    print("Streaming and indexing S2 and S3 target records...")
    blocker = MultiStrategyBlocker(max_candidates_per_entity=50)
    
    # S2 Stream: collect all target_match_ids in S2 + background sample
    s2_total_indexed = 0
    s2_bg_collected = 0
    for chunk in pd.read_csv(TRAIN_SOURCE2_PATH, sep="\t", chunksize=250_000, dtype=str, keep_default_na=False):
        is_target = chunk["entity_id"].isin(target_match_ids)
        if s2_bg_collected < background_sample_per_file:
            take_bg = min(background_sample_per_file - s2_bg_collected, len(chunk))
            is_target.iloc[:take_bg] = True
            s2_bg_collected += take_bg
            
        selected = chunk[is_target]
        if len(selected) > 0:
            selected_norm = normalize_dataframe(selected)
            blocker.index_target_records(selected_norm)
            s2_total_indexed += len(selected)
            
    # S3 Stream: collect all target_match_ids in S3 + background sample
    s3_total_indexed = 0
    s3_bg_collected = 0
    for chunk in pd.read_csv(TRAIN_SOURCE3_PATH, sep="\t", chunksize=250_000, dtype=str, keep_default_na=False):
        is_target = chunk["entity_id"].isin(target_match_ids)
        if s3_bg_collected < background_sample_per_file:
            take_bg = min(background_sample_per_file - s3_bg_collected, len(chunk))
            is_target.iloc[:take_bg] = True
            s3_bg_collected += take_bg
            
        selected = chunk[is_target]
        if len(selected) > 0:
            selected_norm = normalize_dataframe(selected)
            blocker.index_target_records(selected_norm)
            s3_total_indexed += len(selected)
            
    print(f"Total Target records indexed: {s2_total_indexed + s3_total_indexed:,} (S2: {s2_total_indexed:,}, S3: {s3_total_indexed:,})")
    
    # 4. Generate Candidates
    print("Generating candidate pairs across multi-strategy blocker...")
    cand_start = time.time()
    candidate_map = blocker.generate_candidates_batch(s1_eval_norm_df)
    cand_time = time.time() - cand_start
    print(f"Candidate generation completed in {cand_time:.2f}s ({len(candidate_map):,} entities).")
    
    # 5. Evaluate Recall
    total_cartesian = len(eval_s1_ids) * (s2_total_indexed + s3_total_indexed)
    metrics = evaluate_blocking_recall(candidate_map, eval_ground_truth, total_cartesian)
    
    print("\n" + "-" * 50)
    print("BLOCKING RECALL METRICS:")
    print("-" * 50)
    print(f"Evaluated S1 Entities:           {metrics['evaluated_s1_entities']:,}")
    print(f"Total True Matches:             {metrics['total_true_matches']:,}")
    print(f"True Matches Survived:          {metrics['true_matches_survived_blocking']:,}")
    print(f"Overall Blocking Recall:        {metrics['overall_blocking_recall_pct']:.2f}%")
    print(f"S1 -> S2 Blocking Recall:       {metrics['s2_blocking_recall_pct']:.2f}% ({metrics['s2_matches_survived']:,} / {metrics['total_true_s2_matches']:,})")
    print(f"S1 -> S3 Blocking Recall:       {metrics['s3_blocking_recall_pct']:.2f}% ({metrics['s3_matches_survived']:,} / {metrics['total_true_s3_matches']:,})")
    print(f"Candidate Pairs Generated:      {metrics['total_candidate_pairs_generated']:,}")
    print(f"Mean Candidates per S1:         {metrics['mean_candidates_per_s1']:.2f}")
    print(f"Median Candidates per S1:       {metrics['median_candidates_per_s1']}")
    print(f"Max Candidates per S1:          {metrics['max_candidates_per_s1']}")
    print(f"Singletons with 0 Candidates:   {metrics['zero_candidate_s1_count']:,} ({metrics['zero_candidate_s1_pct']}%)")
    print(f"Candidate Reduction Ratio:      {metrics['candidate_reduction_ratio_pct']:.4f}%")
    print(f"Total Runtime:                  {time.time() - start_time:.2f} seconds")
    
    # Save candidate pairs to candidate_pairs.tsv
    save_candidate_pairs_tsv(candidate_map, CANDIDATE_PAIRS_PATH)
    
    return metrics


def main():
    print("=" * 80)
    print("PHASE 2 & PHASE 3 EVALUATION")
    print("=" * 80)
    start_total = time.time()
    
    norm_metrics = run_phase2_normalization_benchmarks(sample_size=100_000)
    blocking_metrics = run_phase3_blocking_evaluation(
        eval_s1_count=10_000,
        background_sample_per_file=200_000,
    )
    
    total_report = {
        "dataset_source": "datasett11/student_resource/dataset",
        "phase2_normalization": norm_metrics,
        "phase3_blocking": blocking_metrics,
        "total_benchmark_runtime_seconds": round(time.time() - start_total, 2),
    }
    
    out_file = os.path.join(OUTPUT_DIR, "phase2_phase3_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(total_report, f, indent=2)
    print(f"\nSaved Phase 2 & 3 benchmark report to {out_file}")


if __name__ == "__main__":
    main()
