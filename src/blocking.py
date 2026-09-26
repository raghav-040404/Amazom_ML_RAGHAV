"""
High-Recall Multi-Strategy Blocking Engine (Enhanced).
Amazon ML Challenge 2026 - Business Entity Resolution.
Strictly processes data originating from `datasett11`.
"""

import os
import re
from collections import defaultdict, Counter
from typing import Dict, Set, List, Tuple, Generator, Optional, Any
import pandas as pd
import numpy as np

from src.config import CONFIG, OUTPUT_DIR, CANDIDATE_PAIRS_PATH
from src.normalize import normalize_text_base

COMMON_STOPWORDS = {
    "the", "and", "of", "in", "for", "on", "at", "to", "a", "an", "or", "by", "with",
    "pvt", "ltd", "private", "limited", "inc", "llc", "corp", "corporation",
    "co", "company", "services", "solutions", "enterprises", "group", "holdings",
    "international", "global", "national", "associates", "consultants", "center",
    "road", "street", "avenue", "drive", "lane", "court", "highway", "floor",
    "suite", "apartment", "block", "sector", "phase", "plot", "no", "near", "opp",
    "behind", "beside", "state", "city", "town", "district",
}


def extract_numbers(text: str) -> List[str]:
    """Extract numeric tokens (house numbers, PIN codes, postal codes)."""
    if not text:
        return []
    return [n for n in re.findall(r"\b\d+\b", text) if len(n) <= 10]


def extract_distinctive_tokens(text: str, min_len: int = 3) -> List[str]:
    """Extract distinctive tokens from text excluding common stopwords."""
    if not text:
        return []
    tokens = text.split()
    return [t for t in tokens if len(t) >= min_len and t not in COMMON_STOPWORDS]


def get_first_word(text: str) -> str:
    """Extract first word of text."""
    if not text:
        return ""
    words = text.split()
    return words[0] if words else ""


class MultiStrategyBlocker:
    """
    High-recall, multi-strategy blocking engine.
    Indexes target records (S2, S3) by complementary high-precision & high-recall keys:
    1. Exact Normalized Name: (country, name)
    2. Name Prefix (5 & 4 chars): (country, prefix)
    3. Distinctive Name Tokens: (country, token)
    4. First Name Word: (country, first_word)
    5. Address Number + Street Token: (country, num, token)
    6. Distinctive Address Tokens: (country, addr_token)
    7. Postal / PIN Code: (country, postal_code)
    """

    def __init__(self, max_candidates_per_entity: int = 50):
        self.max_candidates_per_entity = max_candidates_per_entity
        # Inverted index tables: key -> list of candidate entity_ids
        self.idx_exact_name = defaultdict(list)
        self.idx_name_prefix5 = defaultdict(list)
        self.idx_name_prefix4 = defaultdict(list)
        self.idx_name_tokens = defaultdict(list)
        self.idx_first_word = defaultdict(list)
        self.idx_addr_num_token = defaultdict(list)
        self.idx_addr_distinctive = defaultdict(list)
        self.idx_postal_code = defaultdict(list)

    def index_target_records(self, target_df: pd.DataFrame):
        """Index a batch or full DataFrame of target records (S2 or S3)."""
        for _, row in target_df.iterrows():
            eid = row["entity_id"]
            name = row.get("business_name_normalized", "")
            addr = row.get("business_address_normalized", "")
            country = row.get("country_normalized", "UNKNOWN")

            # 1. Exact Name
            if name:
                self.idx_exact_name[(country, name)].append(eid)

            # 2. Name Prefix (5 chars)
            if len(name) >= 5:
                self.idx_name_prefix5[(country, name[:5])].append(eid)
            elif len(name) >= 3:
                self.idx_name_prefix4[(country, name[:4])].append(eid)

            # 3. First Word
            first_w = get_first_word(name)
            if len(first_w) >= 4 and first_w not in COMMON_STOPWORDS:
                self.idx_first_word[(country, first_w)].append(eid)

            # 4. Distinctive Name Tokens
            name_tokens = extract_distinctive_tokens(name, min_len=3)
            for tok in name_tokens[:5]:
                self.idx_name_tokens[(country, tok)].append(eid)

            # 5. Address Numbers + Street/Locality Tokens
            addr_nums = extract_numbers(addr)
            addr_tokens = extract_distinctive_tokens(addr, min_len=3)
            
            if addr_nums and addr_tokens:
                primary_num = addr_nums[0]
                for tok in addr_tokens[:3]:
                    self.idx_addr_num_token[(country, primary_num, tok)].append(eid)

            # 6. Distinctive Address Tokens (localities, unique streets)
            for tok in addr_tokens[:4]:
                self.idx_addr_distinctive[(country, tok)].append(eid)

            # 7. Postal / PIN Code (5 or 6 digits)
            for num in addr_nums:
                if len(num) in (5, 6):
                    self.idx_postal_code[(country, num)].append(eid)

    def generate_candidates_for_entity(
        self,
        s1_id: str,
        name: str,
        addr: str,
        country: str,
    ) -> Set[str]:
        """Generate candidates for a Source 1 entity across all multi-index strategies."""
        scores = Counter()

        # 1. Exact Name Match (weight = 6)
        if name:
            key = (country, name)
            if key in self.idx_exact_name:
                for cid in self.idx_exact_name[key]:
                    scores[cid] += 6

        # 2. First Word Match (weight = 4)
        first_w = get_first_word(name)
        if len(first_w) >= 4 and first_w not in COMMON_STOPWORDS:
            key = (country, first_w)
            if key in self.idx_first_word:
                cands = self.idx_first_word[key]
                if len(cands) <= 150:
                    for cid in cands:
                        scores[cid] += 4

        # 3. Name Prefix 5 Match (weight = 3)
        if len(name) >= 5:
            key = (country, name[:5])
            if key in self.idx_name_prefix5:
                cands = self.idx_name_prefix5[key]
                if len(cands) <= 150:
                    for cid in cands:
                        scores[cid] += 3

        # 4. Distinctive Name Tokens (weight = 3)
        name_tokens = extract_distinctive_tokens(name, min_len=3)
        for tok in name_tokens[:5]:
            key = (country, tok)
            if key in self.idx_name_tokens:
                cands = self.idx_name_tokens[key]
                if len(cands) <= 100:
                    for cid in cands:
                        scores[cid] += 3

        # 5. Address Number + Token (weight = 4)
        addr_nums = extract_numbers(addr)
        addr_tokens = extract_distinctive_tokens(addr, min_len=3)
        if addr_nums and addr_tokens:
            primary_num = addr_nums[0]
            for tok in addr_tokens[:3]:
                key = (country, primary_num, tok)
                if key in self.idx_addr_num_token:
                    cands = self.idx_addr_num_token[key]
                    if len(cands) <= 100:
                        for cid in cands:
                            scores[cid] += 4

        # 6. Distinctive Address Tokens (weight = 2)
        for tok in addr_tokens[:4]:
            key = (country, tok)
            if key in self.idx_addr_distinctive:
                cands = self.idx_addr_distinctive[key]
                if len(cands) <= 60:
                    for cid in cands:
                        scores[cid] += 2

        # 7. Postal/PIN Code Match (weight = 2)
        for num in addr_nums:
            if len(num) in (5, 6):
                key = (country, num)
                if key in self.idx_postal_code:
                    cands = self.idx_postal_code[key]
                    if len(cands) <= 80:
                        for cid in cands:
                            scores[cid] += 2

        if not scores:
            return set()

        top_candidates = [cid for cid, _ in scores.most_common(self.max_candidates_per_entity)]
        return set(top_candidates)

    def generate_candidates_batch(self, s1_df: pd.DataFrame) -> Dict[str, Set[str]]:
        """Generate candidates for a batch of S1 records."""
        candidate_map = {}
        for _, row in s1_df.iterrows():
            s1_id = row["entity_id"]
            name = row.get("business_name_normalized", "")
            addr = row.get("business_address_normalized", "")
            country = row.get("country_normalized", "UNKNOWN")

            candidates = self.generate_candidates_for_entity(s1_id, name, addr, country)
            candidate_map[s1_id] = candidates
        return candidate_map


def evaluate_blocking_recall(
    candidate_map: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    total_possible_cartesian: Optional[int] = None,
) -> Dict[str, Any]:
    """Measure blocking recall against ground truth."""
    total_true = 0
    survived = 0
    total_s2 = 0
    s2_survived = 0
    total_s3 = 0
    s3_survived = 0

    candidate_counts = []
    total_cand_pairs = 0

    for s1_id, true_matches in ground_truth.items():
        cands = candidate_map.get(s1_id, set())
        n_c = len(cands)
        candidate_counts.append(n_c)
        total_cand_pairs += n_c

        for mid in true_matches:
            total_true += 1
            if mid.startswith("S2-"):
                total_s2 += 1
                if mid in cands:
                    s2_survived += 1
                    survived += 1
            elif mid.startswith("S3-"):
                total_s3 += 1
                if mid in cands:
                    s3_survived += 1
                    survived += 1
            else:
                if mid in cands:
                    survived += 1

    candidate_counts = np.array(candidate_counts)
    recall_overall = (survived / max(1, total_true)) * 100
    recall_s2 = (s2_survived / max(1, total_s2)) * 100
    recall_s3 = (s3_survived / max(1, total_s3)) * 100

    reduction = None
    if total_possible_cartesian and total_possible_cartesian > 0:
        reduction = (1.0 - (total_cand_pairs / total_possible_cartesian)) * 100

    return {
        "evaluated_s1_entities": len(ground_truth),
        "total_true_matches": total_true,
        "true_matches_survived_blocking": survived,
        "overall_blocking_recall_pct": round(recall_overall, 4),
        "total_true_s2_matches": total_s2,
        "s2_matches_survived": s2_survived,
        "s2_blocking_recall_pct": round(recall_s2, 4),
        "total_true_s3_matches": total_s3,
        "s3_matches_survived": s3_survived,
        "s3_blocking_recall_pct": round(recall_s3, 4),
        "total_candidate_pairs_generated": total_cand_pairs,
        "mean_candidates_per_s1": round(float(np.mean(candidate_counts)), 2),
        "median_candidates_per_s1": float(np.median(candidate_counts)),
        "max_candidates_per_s1": int(np.max(candidate_counts)) if len(candidate_counts) > 0 else 0,
        "zero_candidate_s1_count": int(np.sum(candidate_counts == 0)),
        "zero_candidate_s1_pct": round((np.sum(candidate_counts == 0) / max(1, len(ground_truth))) * 100, 2),
        "candidate_reduction_ratio_pct": round(reduction, 6) if reduction is not None else None,
    }


def save_candidate_pairs_tsv(
    candidate_map: Dict[str, Set[str]],
    output_path: str = CANDIDATE_PAIRS_PATH,
):
    """Save candidate pairs to TSV format conforming to challenge specification."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id, candidates in candidate_map.items():
            cand_str = ",".join(sorted(candidates))
            f.write(f"{s1_id}\t{cand_str}\n")
    print(f"Saved candidate pairs to {output_path}")
