"""
Feature Engineering Module for Business Entity Resolution.
Amazon ML Challenge 2026.
Strictly processes data originating from `datasett11`.
"""

import re
import math
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np
import pandas as pd


def fast_levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity ratio between 0.0 and 1.0.
    1.0 means identical, 0.0 means completely different.
    """
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    len1, len2 = len(s1), len(s2)
    # Optimized two-row DP
    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    current_row = list(range(len1 + 1))
    for i in range(1, len2 + 1):
        c2 = s2[i - 1]
        previous_row, current_row = current_row, [i] + [0] * len1
        for j in range(1, len1 + 1):
            c1 = s1[j - 1]
            add = previous_row[j] + 1
            delete = current_row[j - 1] + 1
            change = previous_row[j - 1] + (0 if c1 == c2 else 1)
            current_row[j] = min(add, delete, change)

    dist = current_row[len1]
    max_len = max(len1, len2)
    return max(0.0, 1.0 - (dist / max_len))


def get_ngrams(text: str, n: int = 3) -> Set[str]:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """Compute character n-gram Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    g1 = get_ngrams(s1, n)
    g2 = get_ngrams(s2, n)
    if not g1 or not g2:
        return 0.0
    intersection = len(g1 & g2)
    union = len(g1 | g2)
    return intersection / union if union > 0 else 0.0


def token_jaccard(tokens1: List[str], tokens2: List[str]) -> Tuple[float, int]:
    """Compute token Jaccard similarity and common token count."""
    if not tokens1 or not tokens2:
        return 0.0, 0
    s1 = set(tokens1)
    s2 = set(tokens2)
    intersection = s1 & s2
    union = s1 | s2
    jaccard = len(intersection) / len(union) if union else 0.0
    return jaccard, len(intersection)


def prefix_similarity(s1: str, s2: str) -> float:
    """Compute common prefix ratio relative to max string length."""
    if not s1 or not s2:
        return 0.0
    min_len = min(len(s1), len(s2))
    max_len = max(len(s1), len(s2))
    prefix_len = 0
    for i in range(min_len):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    return prefix_len / max_len if max_len > 0 else 0.0


def extract_numbers_list(text: str) -> List[str]:
    """Extract numeric sequences from text."""
    if not text:
        return []
    return re.findall(r"\b\d+\b", text)


class EntityFeatureExtractor:
    """
    Vectorized and cached feature extractor for Candidate Entity Pairs (S1, S2/S3).
    """

    FEATURE_NAMES = [
        # Name Features
        "name_exact_match",
        "name_norm_exact_match",
        "name_len_diff",
        "name_len_ratio",
        "name_token_count_diff",
        "name_token_overlap",
        "name_token_jaccard",
        "name_char_2gram_jaccard",
        "name_char_3gram_jaccard",
        "name_levenshtein_sim",
        "name_prefix_sim",
        "name_first_token_match",
        
        # Address Features
        "addr_is_missing",
        "addr_exact_match",
        "addr_norm_exact_match",
        "addr_len_diff",
        "addr_len_ratio",
        "addr_token_overlap",
        "addr_token_jaccard",
        "addr_char_3gram_jaccard",
        "addr_levenshtein_sim",
        "addr_num_overlap_count",
        "addr_first_num_match",
        "addr_postal_code_match",
        
        # Country Features
        "country_match",
        
        # Combined / Composite Features
        "name_sim_avg",
        "name_sim_max",
        "addr_sim_avg",
        "addr_sim_max",
        "name_addr_combined_score",
        "source_is_s3",  # 0 for S2, 1 for S3
    ]

    def extract_pair_features(
        self,
        s1_orig_name: str,
        s1_norm_name: str,
        s1_orig_addr: str,
        s1_norm_addr: str,
        s1_country: str,
        cand_orig_name: str,
        cand_norm_name: str,
        cand_orig_addr: str,
        cand_norm_addr: str,
        cand_country: str,
        cand_source_prefix: str,
    ) -> List[float]:
        """
        Compute all matching features for a single candidate pair.
        Returns a list of floats corresponding to FEATURE_NAMES.
        """
        s1_norm_name = s1_norm_name or ""
        cand_norm_name = cand_norm_name or ""
        s1_norm_addr = s1_norm_addr or ""
        cand_norm_addr = cand_norm_addr or ""

        # --- Name Features ---
        name_exact = 1.0 if s1_orig_name == cand_orig_name and s1_orig_name != "" else 0.0
        name_norm_exact = 1.0 if s1_norm_name == cand_norm_name and s1_norm_name != "" else 0.0
        
        len_s1_n = len(s1_norm_name)
        len_c_n = len(cand_norm_name)
        name_len_diff = float(abs(len_s1_n - len_c_n))
        name_len_ratio = (min(len_s1_n, len_c_n) + 1.0) / (max(len_s1_n, len_c_n) + 1.0)
        
        tokens_s1_n = s1_norm_name.split()
        tokens_c_n = cand_norm_name.split()
        name_token_diff = float(abs(len(tokens_s1_n) - len(tokens_c_n)))
        name_jaccard, name_overlap = token_jaccard(tokens_s1_n, tokens_c_n)
        
        name_2gram = ngram_jaccard(s1_norm_name, cand_norm_name, n=2)
        name_3gram = ngram_jaccard(s1_norm_name, cand_norm_name, n=3)
        name_lev = fast_levenshtein_ratio(s1_norm_name, cand_norm_name)
        name_prefix = prefix_similarity(s1_norm_name, cand_norm_name)
        
        first_s1_tok = tokens_s1_n[0] if tokens_s1_n else ""
        first_c_tok = tokens_c_n[0] if tokens_c_n else ""
        name_first_tok_match = 1.0 if first_s1_tok == first_c_tok and first_s1_tok != "" else 0.0

        # --- Address Features ---
        addr_missing = 1.0 if not cand_norm_addr else 0.0
        addr_exact = 1.0 if s1_orig_addr == cand_orig_addr and s1_orig_addr != "" else 0.0
        addr_norm_exact = 1.0 if s1_norm_addr == cand_norm_addr and s1_norm_addr != "" else 0.0
        
        len_s1_a = len(s1_norm_addr)
        len_c_a = len(cand_norm_addr)
        addr_len_diff = float(abs(len_s1_a - len_c_a))
        addr_len_ratio = (min(len_s1_a, len_c_a) + 1.0) / (max(len_s1_a, len_c_a) + 1.0)
        
        tokens_s1_a = s1_norm_addr.split()
        tokens_c_a = cand_norm_addr.split()
        addr_jaccard, addr_overlap = token_jaccard(tokens_s1_a, tokens_c_a)
        addr_3gram = ngram_jaccard(s1_norm_addr, cand_norm_addr, n=3)
        addr_lev = fast_levenshtein_ratio(s1_norm_addr, cand_norm_addr)

        # Numbers in address
        s1_nums = extract_numbers_list(s1_norm_addr)
        cand_nums = extract_numbers_list(cand_norm_addr)
        num_overlap_cnt = float(len(set(s1_nums) & set(cand_nums)))
        
        first_num_s1 = s1_nums[0] if s1_nums else ""
        first_num_c = cand_nums[0] if cand_nums else ""
        first_num_match = 1.0 if first_num_s1 == first_num_c and first_num_s1 != "" else 0.0
        
        # Postal code match (5 or 6 digit numbers)
        s1_postal = {n for n in s1_nums if len(n) in (5, 6)}
        cand_postal = {n for n in cand_nums if len(n) in (5, 6)}
        postal_match = 1.0 if (s1_postal & cand_postal) else 0.0

        # --- Country Feature ---
        country_match = 1.0 if s1_country == cand_country and s1_country != "" else 0.0

        # --- Combined Metrics ---
        name_sim_avg = (name_jaccard + name_3gram + name_lev + name_prefix) / 4.0
        name_sim_max = max(name_norm_exact, name_jaccard, name_3gram, name_lev)
        
        if not addr_missing:
            addr_sim_avg = (addr_jaccard + addr_3gram + addr_lev + first_num_match) / 4.0
            addr_sim_max = max(addr_norm_exact, addr_jaccard, addr_3gram, addr_lev)
            combined_score = 0.55 * name_sim_avg + 0.45 * addr_sim_avg
        else:
            addr_sim_avg = 0.0
            addr_sim_max = 0.0
            combined_score = name_sim_avg * 0.85  # slight penalty for missing address

        source_is_s3 = 1.0 if cand_source_prefix == "S3" else 0.0

        features = [
            name_exact,
            name_norm_exact,
            name_len_diff,
            name_len_ratio,
            name_token_diff,
            float(name_overlap),
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
            float(addr_overlap),
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
        return features
