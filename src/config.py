"""
Configuration Module for Amazon ML Challenge 2026 - Business Entity Resolution.
Single Source of Truth: `datasett11`
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Root directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, "datasett11", "student_resource")

# Dataset subdirectories
DATASET_DIR = os.path.join(DATASET_ROOT, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")

# Raw File Paths strictly from datasett11
TRAIN_SOURCE1_PATH = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_SOURCE2_PATH = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_SOURCE3_PATH = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GROUND_TRUTH_PATH = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")

TEST_SOURCE1_PATH = os.path.join(TEST_DIR, "test_source1.tsv")
TEST_SOURCE2_PATH = os.path.join(TEST_DIR, "test_source2.tsv")
TEST_SOURCE3_PATH = os.path.join(TEST_DIR, "test_source3.tsv")

# Output directory & submission paths
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MATCHING_RESULTS_PATH = os.path.join(OUTPUT_DIR, "matching_results.tsv")
CANDIDATE_PAIRS_PATH = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Intermediate artifacts
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@dataclass
class PipelineConfig:
    """Master configuration class for the Entity Resolution pipeline."""
    # Reproducibility
    random_seed: int = 42
    
    # Chunking & Memory Management
    chunk_size: int = 100_000
    n_jobs: int = -1  # use all available CPU cores
    
    # Data Normalization
    enable_unicode_norm: bool = True
    enable_lower: bool = True
    enable_whitespace_norm: bool = True
    enable_punctuation_norm: bool = True
    enable_ampersand_norm: bool = True
    enable_abbreviation_expansion: bool = True
    preserve_address_numbers: bool = True
    
    # Blocking Strategy (Multi-index candidate generation)
    # Countries must be treated dynamically (open-set: US, India, France, etc.)
    max_candidates_per_entity: int = 50
    blocking_strategies: List[str] = field(default_factory=lambda: [
        "country_and_exact_name",
        "country_and_name_prefix3",
        "country_and_soundex_name",
        "country_and_top_name_token",
        "country_and_pincode_zip",
        "country_and_address_token_overlap",
    ])
    
    # Validation
    val_size: float = 0.15  # 15% holdout validation split from training S1 entities
    beta_fscore: float = 0.5  # Macro F0.5
    
    # Decision Threshold
    decision_threshold: float = 0.65  # precision-heavy optimization target
    threshold_grid: List[float] = field(default_factory=lambda: [
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    ])


# Default singleton config instance
CONFIG = PipelineConfig()
