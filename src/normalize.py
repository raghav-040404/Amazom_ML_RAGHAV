"""
Data Preprocessing and Conservative Normalization Module.
Amazon ML Challenge 2026 - Business Entity Resolution.
Strictly processes data originating from `datasett11`.
"""

import re
import unicodedata
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np

# Common legal and corporate abbreviations mapping
CORP_SUFFIX_MAP = {
    r"\bpvt\.?\s*ltd\.?\b": "pvt ltd",
    r"\bprivate\s*limited\b": "pvt ltd",
    r"\bpvt\b": "pvt",
    r"\bltd\.?\b": "ltd",
    r"\blimited\b": "ltd",
    r"\bllc\.?\b": "llc",
    r"\bl\.l\.c\.?\b": "llc",
    r"\binc\.?\b": "inc",
    r"\bincorporated\b": "inc",
    r"\bcorp\.?\b": "corp",
    r"\bcorporation\b": "corp",
    r"\bco\.?\b": "co",
    r"\bcompany\b": "co",
    r"\bsarl\b": "sarl",
    r"\bsas\b": "sas",
    r"\bs\.a\.s\.?\b": "sas",
    r"\bgmbh\b": "gmbh",
}

# Common address token abbreviations mapping
ADDR_TOKEN_MAP = {
    r"\brd\.?\b": "road",
    r"\bst\.?\b": "street",
    r"\bave\.?\b": "avenue",
    r"\bdr\.?\b": "drive",
    r"\bblvd\.?\b": "boulevard",
    r"\bln\.?\b": "lane",
    r"\bct\.?\b": "court",
    r"\bhwy\.?\b": "highway",
    r"\bapt\.?\b": "apartment",
    r"\bste\.?\b": "suite",
    r"\bfl\.?\b": "floor",
    r"\bno\.?\b": "no",
}


def normalize_text_base(text: Optional[str]) -> str:
    """
    Apply conservative baseline normalization:
    1. Unicode NFKC normalization (handles accents, decomposed chars)
    2. Lowercase
    3. Ampersand and plus normalization
    4. Safe punctuation handling
    5. Whitespace collapsing and stripping
    """
    if not text or pd.isna(text):
        return ""
    
    text = str(text)
    # Unicode NFKD normalization
    text = unicodedata.normalize("NFKD", text)
    # Convert to lowercase
    text = text.lower()
    # Normalize & and +
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"\+", " and ", text)
    # Replace separators (/ , - _ ; : |) with spaces
    text = re.sub(r"[/,_\-;:|]", " ", text)
    # Remove remaining punctuation except alphanumeric and spaces
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_name(name: Optional[str]) -> str:
    """
    Normalize business name safely while preserving meaningful tokens and numbers.
    """
    text = normalize_text_base(name)
    if not text:
        return ""
    
    # Standardize corporate suffixes without deleting words
    for pattern, repl in CORP_SUFFIX_MAP.items():
        text = re.sub(pattern, repl, text)
        
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_business_address(address: Optional[str]) -> str:
    """
    Normalize business address while strictly preserving digits, postal codes, and landmarks.
    """
    text = normalize_text_base(address)
    if not text:
        return ""
    
    # Standardize street / road abbreviations
    for pattern, repl in ADDR_TOKEN_MAP.items():
        text = re.sub(pattern, repl, text)
        
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_country(country: Optional[str]) -> str:
    """
    Normalize country as an open-set string label.
    Preserves dynamic values (US, India, France, etc.).
    """
    if not country or pd.isna(country):
        return "unknown"
    return str(country).strip().upper()


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame containing entity records without modifying original columns.
    
    Adds:
    - business_name_normalized
    - business_address_normalized
    - country_normalized
    """
    df = df.copy()
    
    # Normalize names
    if "business_name" in df.columns:
        df["business_name_normalized"] = df["business_name"].fillna("").apply(normalize_business_name)
        
    # Normalize addresses
    if "business_address" in df.columns:
        df["business_address_normalized"] = df["business_address"].fillna("").apply(normalize_business_address)
        
    # Normalize country
    if "country" in df.columns:
        df["country_normalized"] = df["country"].fillna("").apply(normalize_country)
        
    return df


def analyze_normalization_quality(df_original: pd.DataFrame, df_normalized: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute normalization quality metrics and detect potential collisions.
    """
    total_records = len(df_original)
    
    # Original vs Normalized unique counts
    orig_names = df_original["business_name"].fillna("").values
    norm_names = df_normalized["business_name_normalized"].values
    
    orig_addrs = df_original["business_address"].fillna("").values
    norm_addrs = df_normalized["business_address_normalized"].values
    
    n_unique_orig_names = len(set(orig_names))
    n_unique_norm_names = len(set(norm_names))
    
    n_unique_orig_addrs = len(set(orig_addrs))
    n_unique_norm_addrs = len(set(norm_addrs))
    
    names_changed = int(np.sum(orig_names != norm_names))
    addrs_changed = int(np.sum(orig_addrs != norm_addrs))
    
    empty_norm_names = int(np.sum(norm_names == ""))
    empty_norm_addrs = int(np.sum(norm_addrs == ""))
    
    # Collision analysis: distinct original names mapping to same normalized name
    name_mapping = pd.DataFrame({"orig": orig_names, "norm": norm_names})
    collision_groups = name_mapping.groupby("norm")["orig"].nunique()
    n_colliding_norm_names = int((collision_groups > 1).sum())
    
    return {
        "total_records": total_records,
        "unique_original_names": n_unique_orig_names,
        "unique_normalized_names": n_unique_norm_names,
        "names_changed_count": names_changed,
        "names_changed_pct": round((names_changed / max(1, total_records)) * 100, 2),
        "unique_original_addrs": n_unique_orig_addrs,
        "unique_normalized_addrs": n_unique_norm_addrs,
        "addrs_changed_count": addrs_changed,
        "addrs_changed_pct": round((addrs_changed / max(1, total_records)) * 100, 2),
        "empty_normalized_names_count": empty_norm_names,
        "empty_normalized_names_pct": round((empty_norm_names / max(1, total_records)) * 100, 4),
        "empty_normalized_addrs_count": empty_norm_addrs,
        "empty_normalized_addrs_pct": round((empty_norm_addrs / max(1, total_records)) * 100, 4),
        "colliding_normalized_name_keys": n_colliding_norm_names,
    }
