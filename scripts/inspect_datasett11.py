"""
Comprehensive Dataset Inspection & EDA for datasett11
Amazon ML Challenge 2026 - Business Entity Resolution

This script strictly inspects `datasett11/student_resource` files and saves
structured analysis into `output/datasett11_inspection_report.json` and a formatted text report.
"""

import os
import sys
import json
import time
from collections import Counter, defaultdict
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET11_DIR = os.path.join(BASE_DIR, "datasett11", "student_resource")
TRAIN_DIR = os.path.join(DATASET11_DIR, "dataset", "train")
TEST_DIR = os.path.join(DATASET11_DIR, "dataset", "test")

FILES = {
    "train_source1": os.path.join(TRAIN_DIR, "train_source1.tsv"),
    "train_source2": os.path.join(TRAIN_DIR, "train_source2.tsv"),
    "train_source3": os.path.join(TRAIN_DIR, "train_source3.tsv"),
    "train_ground_truth": os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
    "test_source1": os.path.join(TEST_DIR, "test_source1.tsv"),
    "test_source2": os.path.join(TEST_DIR, "test_source2.tsv"),
    "test_source3": os.path.join(TEST_DIR, "test_source3.tsv"),
}


def get_file_size_info(filepath):
    size_bytes = os.path.getsize(filepath)
    size_mb = size_bytes / (1024 * 1024)
    size_gb = size_bytes / (1024 * 1024 * 1024)
    return size_bytes, size_mb, size_gb


def inspect_source_file(filepath, name, chunksize=200_000):
    print(f"\n{'='*70}\n[datasett11] Inspecting: {name}\nPath: {filepath}\n{'='*70}")
    size_bytes, size_mb, size_gb = get_file_size_info(filepath)
    
    total_rows = 0
    col_names = []
    dtypes = {}
    missing_empty_counts = defaultdict(int)
    
    entity_ids = set()
    dup_entity_ids = 0
    
    country_counts = Counter()
    
    name_lens = []
    name_word_counts = []
    addr_lens = []
    addr_word_counts = []
    
    sample_records = []
    
    for i, chunk in enumerate(pd.read_csv(filepath, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False)):
        if i == 0:
            col_names = list(chunk.columns)
            dtypes = {c: "string (object)" for c in chunk.columns}
            sample_records = chunk.head(5).to_dict(orient="records")
        
        chunk_rows = len(chunk)
        total_rows += chunk_rows
        
        # Entity ID duplicates
        if "entity_id" in chunk.columns:
            ids = chunk["entity_id"].values
            for eid in ids:
                if eid in entity_ids:
                    dup_entity_ids += 1
                else:
                    entity_ids.add(eid)
                    
        # Missing & empty values
        for c in chunk.columns:
            vals = chunk[c].values
            empty_mask = (vals == "") | (vals == "nan") | (vals == "None") | (vals == "NULL")
            missing_empty_counts[c] += int(np.sum(empty_mask))
            
        # Country distribution
        if "country" in chunk.columns:
            country_counts.update(chunk["country"].value_counts().to_dict())
            
        # Sampling length statistics
        if "business_name" in chunk.columns:
            s_name = chunk["business_name"]
            n_len = s_name.str.len().values
            n_words = s_name.str.split().str.len().fillna(0).values
            if len(name_lens) < 300_000:
                name_lens.extend(n_len[:min(50_000, len(n_len))])
                name_word_counts.extend(n_words[:min(50_000, len(n_words))])
                
        if "business_address" in chunk.columns:
            s_addr = chunk["business_address"]
            a_len = s_addr.str.len().values
            a_words = s_addr.str.split().str.len().fillna(0).values
            if len(addr_lens) < 300_000:
                addr_lens.extend(a_len[:min(50_000, len(a_len))])
                addr_word_counts.extend(a_words[:min(50_000, len(a_words))])

    name_lens = np.array(name_lens)
    addr_lens = np.array(addr_lens)
    name_word_counts = np.array(name_word_counts)
    addr_word_counts = np.array(addr_word_counts)

    def stats_summary(arr):
        if len(arr) == 0:
            return {}
        return {
            "min": int(np.min(arr)),
            "max": int(np.max(arr)),
            "mean": round(float(np.mean(arr)), 2),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    # RAM footprint estimation when loaded into DataFrame
    # Using categorical country + Arrow/PyArrow strings or compact objects
    est_ram_mb = (size_bytes * 1.25) / (1024 * 1024)

    results = {
        "file_name": name,
        "file_path": filepath,
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_mb, 2),
        "file_size_gb": round(size_gb, 4),
        "total_rows": total_rows,
        "num_columns": len(col_names),
        "column_names": col_names,
        "dtypes": dtypes,
        "duplicate_entity_ids": dup_entity_ids,
        "unique_entity_ids": len(entity_ids),
        "missing_empty_counts": dict(missing_empty_counts),
        "missing_empty_percentages": {c: round((missing_empty_counts[c] / max(1, total_rows)) * 100, 4) for c in col_names},
        "unique_countries_count": len(country_counts),
        "country_distribution": dict(country_counts),
        "name_char_length_stats": stats_summary(name_lens),
        "name_word_count_stats": stats_summary(name_word_counts),
        "addr_char_length_stats": stats_summary(addr_lens),
        "addr_word_count_stats": stats_summary(addr_word_counts),
        "estimated_dataframe_ram_mb": round(est_ram_mb, 2),
        "sample_records": sample_records,
    }
    
    print(f"Rows: {total_rows:,} | Size: {size_mb:.2f} MB ({size_bytes:,} bytes)")
    print(f"Countries: {dict(country_counts)}")
    print(f"Missing Values (%): {results['missing_empty_percentages']}")
    print(f"Name Lengths: mean={results['name_char_length_stats'].get('mean')}, median={results['name_char_length_stats'].get('median')}, max={results['name_char_length_stats'].get('max')}")
    print(f"Addr Lengths: mean={results['addr_char_length_stats'].get('mean')}, median={results['addr_char_length_stats'].get('median')}, max={results['addr_char_length_stats'].get('max')}")

    return results, entity_ids


def inspect_ground_truth(filepath, train_s1_ids, train_s2_ids, train_s3_ids, chunksize=200_000):
    print(f"\n{'='*70}\n[datasett11] Inspecting Ground Truth: {filepath}\n{'='*70}")
    size_bytes, size_mb, size_gb = get_file_size_info(filepath)
    
    total_rows = 0
    gt_s1_ids = set()
    dup_s1_ids = 0
    
    match_counts = []
    s2_match_counts = []
    s3_match_counts = []
    
    s2_matched_ids = set()
    s3_matched_ids = set()
    
    self_match_count = 0
    unknown_id_count = 0
    
    col_names = []
    sample_records = []
    
    for i, chunk in enumerate(pd.read_csv(filepath, sep="\t", chunksize=chunksize, dtype=str, keep_default_na=False)):
        if i == 0:
            col_names = list(chunk.columns)
            sample_records = chunk.head(5).to_dict(orient="records")
            
        total_rows += len(chunk)
        for _, row in chunk.iterrows():
            s1_id = row["source1_entity_id"]
            if s1_id in gt_s1_ids:
                dup_s1_ids += 1
            else:
                gt_s1_ids.add(s1_id)
                
            raw_matched = row["matched_entity_ids"].strip()
            if not raw_matched:
                match_counts.append(0)
                s2_match_counts.append(0)
                s3_match_counts.append(0)
            else:
                m_ids = [m.strip() for m in raw_matched.split(",") if m.strip()]
                match_counts.append(len(m_ids))
                
                s2_cnt = 0
                s3_cnt = 0
                for mid in m_ids:
                    if mid.startswith("S1-"):
                        self_match_count += 1
                    elif mid.startswith("S2-"):
                        s2_cnt += 1
                        s2_matched_ids.add(mid)
                        if train_s2_ids and mid not in train_s2_ids:
                            unknown_id_count += 1
                    elif mid.startswith("S3-"):
                        s3_cnt += 1
                        s3_matched_ids.add(mid)
                        if train_s3_ids and mid not in train_s3_ids:
                            unknown_id_count += 1
                    else:
                        unknown_id_count += 1
                s2_match_counts.append(s2_cnt)
                s3_match_counts.append(s3_cnt)
                
    match_counts = np.array(match_counts)
    s2_match_counts = np.array(s2_match_counts)
    s3_match_counts = np.array(s3_match_counts)
    
    zero_matches = int(np.sum(match_counts == 0))
    one_match = int(np.sum(match_counts == 1))
    two_matches = int(np.sum(match_counts == 2))
    three_plus = int(np.sum(match_counts >= 3))
    
    missing_in_gt = len(train_s1_ids - gt_s1_ids) if train_s1_ids else 0
    extra_in_gt = len(gt_s1_ids - train_s1_ids) if train_s1_ids else 0
    
    results = {
        "file_name": "train_ground_truth",
        "file_path": filepath,
        "file_size_bytes": size_bytes,
        "file_size_mb": round(size_mb, 2),
        "file_size_gb": round(size_gb, 4),
        "total_s1_rows": total_rows,
        "unique_s1_ids": len(gt_s1_ids),
        "duplicate_s1_ids": dup_s1_ids,
        "s1_in_source1_missing_in_gt": missing_in_gt,
        "s1_in_gt_missing_in_source1": extra_in_gt,
        "zero_matches_count_singletons": zero_matches,
        "zero_matches_pct_singletons": round((zero_matches / max(1, total_rows)) * 100, 2),
        "one_match_count": one_match,
        "one_match_pct": round((one_match / max(1, total_rows)) * 100, 2),
        "two_matches_count": two_matches,
        "two_matches_pct": round((two_matches / max(1, total_rows)) * 100, 2),
        "three_plus_matches_count": three_plus,
        "three_plus_matches_pct": round((three_plus / max(1, total_rows)) * 100, 2),
        "total_matched_pairs": int(np.sum(match_counts)),
        "mean_matches_per_s1": round(float(np.mean(match_counts)), 4),
        "median_matches_per_s1": float(np.median(match_counts)),
        "max_matches_per_s1": int(np.max(match_counts)),
        "total_unique_s2_matched": len(s2_matched_ids),
        "total_unique_s3_matched": len(s3_matched_ids),
        "self_matches_count": self_match_count,
        "unknown_ids_count": unknown_id_count,
        "sample_records": sample_records,
    }
    
    print(f"Total S1 entities: {total_rows:,} | Total Match Links: {results['total_matched_pairs']:,}")
    print(f"Singletons (0 matches): {zero_matches:,} ({results['zero_matches_pct_singletons']}%)")
    print(f"1 match: {one_match:,} ({results['one_match_pct']}%)")
    print(f"2 matches: {two_matches:,} ({results['two_matches_pct']}%)")
    print(f"3+ matches: {three_plus:,} ({results['three_plus_matches_pct']}%)")
    print(f"Unique S2 Matched: {len(s2_matched_ids):,} | Unique S3 Matched: {len(s3_matched_ids):,}")
    print(f"Self-matches: {self_match_count} | Unknown IDs: {unknown_id_count}")

    return results


def run_inspection():
    print("=" * 80)
    print("PHASE 1: INSPECTING datasett11 ONLY")
    print("=" * 80)
    start_time = time.time()
    
    all_results = {}
    id_sets = {}
    
    # 1. Inspect Train Sources
    for name in ["train_source1", "train_source2", "train_source3"]:
        res, ids = inspect_source_file(FILES[name], name)
        all_results[name] = res
        id_sets[name] = ids
        
    # 2. Inspect Train Ground Truth
    gt_res = inspect_ground_truth(
        FILES["train_ground_truth"],
        id_sets["train_source1"],
        id_sets["train_source2"],
        id_sets["train_source3"],
    )
    all_results["train_ground_truth"] = gt_res
    
    # 3. Inspect Test Sources
    for name in ["test_source1", "test_source2", "test_source3"]:
        res, ids = inspect_source_file(FILES[name], name)
        all_results[name] = res
        id_sets[name] = ids
        
    elapsed = time.time() - start_time
    print(f"\nInspection of datasett11 completed in {elapsed:.2f} seconds.")
    
    os.makedirs(os.path.join(BASE_DIR, "output"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "output", "datasett11_inspection_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved complete inspection metrics to {out_path}")
    return all_results


if __name__ == "__main__":
    run_inspection()
