"""
Memory-Efficient Data Loader for Amazon ML Challenge 2026.
Strictly loads data from `datasett11`.
"""

import os
from typing import Generator, Dict, Set, List, Optional, Tuple, Iterator
import pandas as pd
import numpy as np

from src.config import (
    TRAIN_SOURCE1_PATH,
    TRAIN_SOURCE2_PATH,
    TRAIN_SOURCE3_PATH,
    TRAIN_GROUND_TRUTH_PATH,
    TEST_SOURCE1_PATH,
    TEST_SOURCE2_PATH,
    TEST_SOURCE3_PATH,
    CONFIG,
)

# Standard schema types for memory efficiency
SOURCE_DTYPES = {
    "entity_id": "string",
    "business_name": "string",
    "business_address": "string",
    "country": "category",
}


def stream_source_tsv(
    filepath: str,
    chunksize: int = 100_000,
    columns: Optional[List[str]] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream a source TSV file in chunks to prevent memory spikes.
    
    Args:
        filepath: Path to the TSV file.
        chunksize: Number of rows per chunk.
        columns: Specific columns to load (optional).
    
    Yields:
        pd.DataFrame with optimized dtypes.
    """
    usecols = columns if columns else list(SOURCE_DTYPES.keys())
    dtypes = {c: SOURCE_DTYPES[c] for c in usecols if c in SOURCE_DTYPES}
    
    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        chunksize=chunksize,
        usecols=usecols,
        dtype=dtypes,
        keep_default_na=False,
    ):
        yield chunk


def load_source_tsv(
    filepath: str,
    columns: Optional[List[str]] = None,
    country_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a source TSV file into memory with optimized memory types.
    
    Args:
        filepath: Path to the TSV file.
        columns: Specific columns to load.
        country_filter: Optional filter for a specific country (e.g., 'US', 'India', 'France').
    
    Returns:
        pd.DataFrame with optimized dtypes.
    """
    usecols = columns if columns else list(SOURCE_DTYPES.keys())
    dtypes = {c: SOURCE_DTYPES[c] for c in usecols if c in SOURCE_DTYPES}
    
    df = pd.read_csv(
        filepath,
        sep="\t",
        usecols=usecols,
        dtype=dtypes,
        keep_default_na=False,
    )
    if country_filter is not None:
        df = df[df["country"] == country_filter].reset_index(drop=True)
    return df


def load_ground_truth(filepath: str = TRAIN_GROUND_TRUTH_PATH) -> Dict[str, Set[str]]:
    """
    Load train_ground_truth.tsv into a compact dictionary {s1_id: set(matched_ids)}.
    
    Returns:
        Dict mapping source1_entity_id -> set of matched entity IDs (empty set for singletons).
    """
    ground_truth: Dict[str, Set[str]] = {}
    
    for chunk in pd.read_csv(
        filepath,
        sep="\t",
        chunksize=200_000,
        dtype=str,
        keep_default_na=False,
    ):
        for _, row in chunk.iterrows():
            s1_id = row["source1_entity_id"]
            raw_matched = row["matched_entity_ids"].strip()
            if not raw_matched:
                ground_truth[s1_id] = set()
            else:
                ground_truth[s1_id] = {m.strip() for m in raw_matched.split(",") if m.strip()}
                
    return ground_truth


def load_train_validation_split(
    val_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Create a deterministic holdout validation split of Source 1 entity IDs.
    
    Returns:
        (train_s1_ids, val_s1_ids)
    """
    rng = np.random.RandomState(random_seed)
    
    # Read only the S1 entity IDs
    s1_ids = []
    for chunk in pd.read_csv(
        TRAIN_SOURCE1_PATH,
        sep="\t",
        usecols=["entity_id"],
        chunksize=200_000,
        dtype=str,
    ):
        s1_ids.extend(chunk["entity_id"].tolist())
        
    s1_ids = np.array(s1_ids)
    rng.shuffle(s1_ids)
    
    n_val = int(len(s1_ids) * val_size)
    val_ids = s1_ids[:n_val].tolist()
    train_ids = s1_ids[n_val:].tolist()
    
    return train_ids, val_ids
