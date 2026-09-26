"""
High-Throughput Production Test Inference Engine.
Amazon ML Challenge 2026 - Business Entity Resolution.
Strictly processes test data originating from `datasett11`.
Reuses the trained Gradient Boosted Trees model at threshold 0.95.
"""

import os
import sys
import time
import pickle
from collections import defaultdict
from typing import Dict, Set, List, Tuple, Optional, Any
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import (
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    OUTPUT_DIR,
    ARTIFACTS_DIR,
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH,
)
from src.normalize import normalize_dataframe
from src.blocking import MultiStrategyBlocker
from src.features import (
    fast_levenshtein_ratio,
    get_ngrams,
    prefix_similarity,
    extract_numbers_list,
)


class FastPairFeatureExtractor:
    """
    Ultra-fast feature extractor utilizing pre-tokenized string representations.
    Produces identical 31-feature vectors as EntityFeatureExtractor.
    """

    @staticmethod
    def prepare_record_cache(df: pd.DataFrame, is_s3: bool = False) -> Dict[str, Dict[str, Any]]:
        """Pre-compute string tokenizations, ngrams, and numbers for a batch of records."""
        cache = {}
        for _, row in df.iterrows():
            eid = row["entity_id"]
            orig_name = str(row.get("business_name", "") or "")
            norm_name = str(row.get("business_name_normalized", "") or "")
            orig_addr = str(row.get("business_address", "") or "")
            norm_addr = str(row.get("business_address_normalized", "") or "")
            country = str(row.get("country_normalized", "UNKNOWN") or "")

            name_tokens = norm_name.split()
            addr_tokens = norm_addr.split()
            
            cache[eid] = {
                "orig_name": orig_name,
                "norm_name": norm_name,
                "len_name": len(norm_name),
                "name_tokens": name_tokens,
                "set_name_tokens": set(name_tokens),
                "name_2gram": get_ngrams(norm_name, 2),
                "name_3gram": get_ngrams(norm_name, 3),
                "first_name_tok": name_tokens[0] if name_tokens else "",
                
                "orig_addr": orig_addr,
                "norm_addr": norm_addr,
                "len_addr": len(norm_addr),
                "addr_tokens": addr_tokens,
                "set_addr_tokens": set(addr_tokens),
                "addr_3gram": get_ngrams(norm_addr, 3),
                "addr_nums": extract_numbers_list(norm_addr),
                "country": country,
                "is_s3": 1.0 if is_s3 else 0.0,
            }
        return cache

    @staticmethod
    def extract_from_cache(s1: Dict[str, Any], cand: Dict[str, Any]) -> List[float]:
        """Compute exact 31 features between pre-cached S1 and candidate dictionaries."""
        # --- Name Features ---
        s1_orig_n = s1["orig_name"]
        c_orig_n = cand["orig_name"]
        name_exact = 1.0 if s1_orig_n == c_orig_n and s1_orig_n != "" else 0.0
        
        s1_norm_n = s1["norm_name"]
        c_norm_n = cand["norm_name"]
        name_norm_exact = 1.0 if s1_norm_n == c_norm_n and s1_norm_n != "" else 0.0
        
        len_s1_n = s1["len_name"]
        len_c_n = cand["len_name"]
        name_len_diff = float(abs(len_s1_n - len_c_n))
        name_len_ratio = (min(len_s1_n, len_c_n) + 1.0) / (max(len_s1_n, len_c_n) + 1.0)

        set_s1_nt = s1["set_name_tokens"]
        set_c_nt = cand["set_name_tokens"]
        name_token_diff = float(abs(len(s1["name_tokens"]) - len(cand["name_tokens"])))
        
        inter_n = len(set_s1_nt & set_c_nt)
        union_n = len(set_s1_nt | set_c_nt)
        name_jaccard = inter_n / union_n if union_n else 0.0
        name_overlap = float(inter_n)

        # 2-gram and 3-gram
        g2_1 = s1["name_2gram"]
        g2_2 = cand["name_2gram"]
        u2 = len(g2_1 | g2_2)
        name_2gram = len(g2_1 & g2_2) / u2 if u2 else 0.0

        g3_1 = s1["name_3gram"]
        g3_2 = cand["name_3gram"]
        u3 = len(g3_1 | g3_2)
        name_3gram = len(g3_1 & g3_2) / u3 if u3 else 0.0

        name_lev = fast_levenshtein_ratio(s1_norm_n, c_norm_n)
        name_prefix = prefix_similarity(s1_norm_n, c_norm_n)

        f1_tok = s1["first_name_tok"]
        fc_tok = cand["first_name_tok"]
        name_first_tok_match = 1.0 if f1_tok == fc_tok and f1_tok != "" else 0.0

        # --- Address Features ---
        s1_orig_a = s1["orig_addr"]
        c_orig_a = cand["orig_addr"]
        s1_norm_a = s1["norm_addr"]
        c_norm_a = cand["norm_addr"]
        
        addr_missing = 1.0 if not c_norm_a else 0.0
        addr_exact = 1.0 if s1_orig_a == c_orig_a and s1_orig_a != "" else 0.0
        addr_norm_exact = 1.0 if s1_norm_a == c_norm_a and s1_norm_a != "" else 0.0

        len_s1_a = s1["len_addr"]
        len_c_a = cand["len_addr"]
        addr_len_diff = float(abs(len_s1_a - len_c_a))
        addr_len_ratio = (min(len_s1_a, len_c_a) + 1.0) / (max(len_s1_a, len_c_a) + 1.0)

        set_s1_at = s1["set_addr_tokens"]
        set_c_at = cand["set_addr_tokens"]
        inter_a = len(set_s1_at & set_c_at)
        union_a = len(set_s1_at | set_c_at)
        addr_jaccard = inter_a / union_a if union_a else 0.0
        addr_overlap = float(inter_a)

        ag3_1 = s1["addr_3gram"]
        ag3_2 = cand["addr_3gram"]
        au3 = len(ag3_1 | ag3_2)
        addr_3gram = len(ag3_1 & ag3_2) / au3 if au3 else 0.0

        addr_lev = fast_levenshtein_ratio(s1_norm_a, c_norm_a)

        s1_nums = s1["addr_nums"]
        c_nums = cand["addr_nums"]
        num_overlap_cnt = float(len(set(s1_nums) & set(c_nums)))

        f_num1 = s1_nums[0] if s1_nums else ""
        f_numc = c_nums[0] if c_nums else ""
        first_num_match = 1.0 if f_num1 == f_numc and f_num1 != "" else 0.0

        s1_postal = {n for n in s1_nums if len(n) in (5, 6)}
        c_postal = {n for n in c_nums if len(n) in (5, 6)}
        postal_match = 1.0 if (s1_postal & c_postal) else 0.0

        country_match = 1.0 if s1["country"] == cand["country"] and s1["country"] != "" else 0.0

        # Composite metrics
        name_sim_avg = (name_jaccard + name_3gram + name_lev + name_prefix) / 4.0
        name_sim_max = max(name_norm_exact, name_jaccard, name_3gram, name_lev)

        if not addr_missing:
            addr_sim_avg = (addr_jaccard + addr_3gram + addr_lev + first_num_match) / 4.0
            addr_sim_max = max(addr_norm_exact, addr_jaccard, addr_3gram, addr_lev)
            combined_score = 0.55 * name_sim_avg + 0.45 * addr_sim_avg
        else:
            addr_sim_avg = 0.0
            addr_sim_max = 0.0
            combined_score = name_sim_avg * 0.85

        source_is_s3 = cand["is_s3"]

        return [
            name_exact,
            name_norm_exact,
            name_len_diff,
            name_len_ratio,
            name_token_diff,
            name_overlap,
            name_jaccard,
            name_2gram,
            name_3gram,
            name_lev,
            name_prefix,
            name_first_tok_match,
            addr_missing,
            addr_exact,
            addr_norm_exact,
            addr_len_diff,
            addr_len_ratio,
            addr_overlap,
            addr_jaccard,
            addr_3gram,
            addr_lev,
            num_overlap_cnt,
            first_num_match,
            postal_match,
            country_match,
            name_sim_avg,
            name_sim_max,
            addr_sim_avg,
            addr_sim_max,
            combined_score,
            source_is_s3,
        ]


class TestInferencePipeline:
    """
    Memory-efficient, pre-cached streaming inference pipeline reusing frozen model at threshold 0.95.
    """

    def __init__(self, threshold: float = 0.95, model_path: Optional[str] = None):
        self.threshold = threshold
        self.extractor = FastPairFeatureExtractor()
        
        if model_path is None:
            model_path = os.path.join(ARTIFACTS_DIR, "best_tree_model.pkl")
            
        print(f"Loading trained Gradient Boosted Trees model from {model_path}...")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
            
        print(f"Inference engine initialized with frozen threshold = {self.threshold}")

    def run_inference_on_test_set(
        self,
        matching_out_path: str = MATCHING_RESULTS_PATH,
        candidate_out_path: str = CANDIDATE_PAIRS_PATH,
        chunksize_s1: int = 50_000,
    ) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(matching_out_path), exist_ok=True)
        os.makedirs(os.path.dirname(candidate_out_path), exist_ok=True)
        
        start_total = time.time()
        
        print("Discovering country partitions in test set...")
        s1_countries = set()
        for chunk in pd.read_csv(TEST_SOURCE1_PATH, sep="\t", chunksize=200_000, usecols=["country"], dtype=str):
            s1_countries.update(chunk["country"].dropna().str.strip().str.upper().unique())
            
        country_list = sorted(s1_countries)
        print(f"Found {len(country_list)} country partition(s): {country_list}")

        # Open output TSV writers
        matching_f = open(matching_out_path, "w", encoding="utf-8")
        candidate_f = open(candidate_out_path, "w", encoding="utf-8")
        
        matching_f.write("source1_entity_id\tmatched_entity_ids\n")
        candidate_f.write("source1_entity_id\tcandidate_entity_ids\n")

        total_s1_processed = 0
        total_candidates_generated = 0
        total_matches_predicted = 0
        s1_singletons_predicted = 0
        s1_single_match_predicted = 0
        s1_multi_match_predicted = 0

        s1_to_s2_candidates = 0
        s1_to_s3_candidates = 0

        for country in country_list:
            print(f"\n{'='*70}\nProcessing Country Partition: {country}\n{'='*70}")
            country_start = time.time()
            
            blocker = MultiStrategyBlocker(max_candidates_per_entity=50)
            target_cache = {}

            # Stream S2 for this country
            print(f"[{country}] Indexing & Pre-Caching Test Source 2...")
            s2_cnt = 0
            for chunk in pd.read_csv(TEST_SOURCE2_PATH, sep="\t", chunksize=250_000, dtype=str, keep_default_na=False):
                c_mask = chunk["country"].str.strip().str.upper() == country
                chunk_c = chunk[c_mask]
                if len(chunk_c) > 0:
                    norm_c = normalize_dataframe(chunk_c)
                    blocker.index_target_records(norm_c)
                    cache_chunk = FastPairFeatureExtractor.prepare_record_cache(norm_c, is_s3=False)
                    target_cache.update(cache_chunk)
                    s2_cnt += len(chunk_c)

            # Stream S3 for this country
            print(f"[{country}] Indexing & Pre-Caching Test Source 3...")
            s3_cnt = 0
            for chunk in pd.read_csv(TEST_SOURCE3_PATH, sep="\t", chunksize=250_000, dtype=str, keep_default_na=False):
                c_mask = chunk["country"].str.strip().str.upper() == country
                chunk_c = chunk[c_mask]
                if len(chunk_c) > 0:
                    norm_c = normalize_dataframe(chunk_c)
                    blocker.index_target_records(norm_c)
                    cache_chunk = FastPairFeatureExtractor.prepare_record_cache(norm_c, is_s3=True)
                    target_cache.update(cache_chunk)
                    s3_cnt += len(chunk_c)

            print(f"[{country}] Indexed {len(target_cache):,} target records in memory (S2: {s2_cnt:,}, S3: {s3_cnt:,})")

            # Stream S1 for this country
            print(f"[{country}] Streaming S1 records and predicting matches...")
            s1_country_count = 0
            
            for chunk in pd.read_csv(TEST_SOURCE1_PATH, sep="\t", chunksize=chunksize_s1, dtype=str, keep_default_na=False):
                c_mask = chunk["country"].str.strip().str.upper() == country
                chunk_s1 = chunk[c_mask]
                if len(chunk_s1) == 0:
                    continue

                chunk_s1_norm = normalize_dataframe(chunk_s1)
                s1_cache = FastPairFeatureExtractor.prepare_record_cache(chunk_s1_norm, is_s3=False)
                candidate_map = blocker.generate_candidates_batch(chunk_s1_norm)

                # Build pair list
                pair_list = []
                for s1_id, cand_set in candidate_map.items():
                    for cid in cand_set:
                        if cid.startswith("S2-"):
                            s1_to_s2_candidates += 1
                        elif cid.startswith("S3-"):
                            s1_to_s3_candidates += 1
                        pair_list.append((s1_id, cid))

                total_candidates_generated += len(pair_list)

                # Fast Feature Extraction
                feature_rows = []
                valid_pairs = []
                for s1_id, cid in pair_list:
                    if s1_id in s1_cache and cid in target_cache:
                        feats = FastPairFeatureExtractor.extract_from_cache(s1_cache[s1_id], target_cache[cid])
                        feature_rows.append(feats)
                        valid_pairs.append((s1_id, cid))

                # Batch Prediction
                s1_matches = defaultdict(list)
                if feature_rows:
                    X_batch = np.array(feature_rows, dtype=np.float32)
                    probs = self.model.predict_proba(X_batch)[:, 1]
                    for (s1_id, cid), prob in zip(valid_pairs, probs):
                        if prob >= self.threshold:
                            s1_matches[s1_id].append(cid)

                # Write batch results
                for s1_id in chunk_s1["entity_id"].values:
                    cand_ids = candidate_map.get(s1_id, set())
                    cand_str = ",".join(sorted(cand_ids))
                    candidate_f.write(f"{s1_id}\t{cand_str}\n")

                    matched_ids = sorted(set(s1_matches.get(s1_id, [])))
                    match_str = ",".join(matched_ids)
                    matching_f.write(f"{s1_id}\t{match_str}\n")

                    n_m = len(matched_ids)
                    total_matches_predicted += n_m
                    if n_m == 0:
                        s1_singletons_predicted += 1
                    elif n_m == 1:
                        s1_single_match_predicted += 1
                    else:
                        s1_multi_match_predicted += 1

                s1_country_count += len(chunk_s1)
                total_s1_processed += len(chunk_s1)
                print(f"  [{country}] Processed {s1_country_count:,} S1 entities... (Overall: {total_s1_processed:,})")

            country_elapsed = time.time() - country_start
            print(f"[{country}] Completed in {country_elapsed:.2f}s.")
            del blocker
            del target_cache

        matching_f.close()
        candidate_f.close()

        total_elapsed = time.time() - start_total
        print(f"\n{'='*70}\nINFERENCE COMPLETE ({total_elapsed:.2f} seconds)\n{'='*70}")

        stats = {
            "model_reused": True,
            "test_s1_entities": total_s1_processed,
            "test_s2_entities": 4_887_273,
            "test_s3_entities": 5_082_316,
            "total_candidate_pairs": total_candidates_generated,
            "s1_to_s2_candidates": s1_to_s2_candidates,
            "s1_to_s3_candidates": s1_to_s3_candidates,
            "average_candidates_per_s1": round(total_candidates_generated / max(1, total_s1_processed), 2),
            "predicted_matched_s1_entities": total_s1_processed - s1_singletons_predicted,
            "predicted_singletons_count": s1_singletons_predicted,
            "predicted_singletons_pct": round(s1_singletons_predicted / max(1, total_s1_processed) * 100, 2),
            "predicted_single_match_count": s1_single_match_predicted,
            "predicted_single_match_pct": round(s1_single_match_predicted / max(1, total_s1_processed) * 100, 2),
            "predicted_multi_match_count": s1_multi_match_predicted,
            "predicted_multi_match_pct": round(s1_multi_match_predicted / max(1, total_s1_processed) * 100, 2),
            "total_predicted_matches": total_matches_predicted,
            "average_matches_per_s1": round(total_matches_predicted / max(1, total_s1_processed), 4),
            "final_threshold": self.threshold,
            "total_inference_runtime_seconds": round(total_elapsed, 2),
            "matching_results_path": matching_out_path,
            "candidate_pairs_path": candidate_out_path,
        }

        with open(os.path.join(OUTPUT_DIR, "test_inference_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return stats
