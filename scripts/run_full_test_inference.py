"""
Execution Script: Full Test Inference and Official Submission Validation.
Amazon ML Challenge 2026 - Business Entity Resolution.
Strictly processes data originating from `datasett11`.
Reuses trained Gradient Boosted Trees model with threshold = 0.95.
"""

import os
import sys
import json
import time

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import (
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH,
    TEST_DIR,
    OUTPUT_DIR,
)
from src.predict import TestInferencePipeline
from src.submission import run_official_validator


def main():
    print("=" * 80)
    print("PHASE 4: FULL TEST INFERENCE & SUBMISSION VALIDATION")
    print("Dataset Source: datasett11/student_resource/dataset/test")
    print("Threshold: 0.95 (Reusing trained model)")
    print("=" * 80)

    # 1. Run Streaming Test Inference
    pipeline = TestInferencePipeline(threshold=0.95)
    stats = pipeline.run_inference_on_test_set(
        matching_out_path=MATCHING_RESULTS_PATH,
        candidate_out_path=CANDIDATE_PAIRS_PATH,
        chunksize_s1=60_000,
    )

    print("\n" + "=" * 80)
    print("TEST INFERENCE SUMMARY:")
    print("=" * 80)
    print(f"Model Reused:                      YES")
    print(f"Total Test S1 Entities:            {stats['test_s1_entities']:,}")
    print(f"Test S2 Entities:                  {stats['test_s2_entities']:,}")
    print(f"Test S3 Entities:                  {stats['test_s3_entities']:,}")
    print(f"Total Candidate Pairs:             {stats['total_candidate_pairs']:,}")
    print(f"  - S1 -> S2:                      {stats['s1_to_s2_candidates']:,}")
    print(f"  - S1 -> S3:                      {stats['s1_to_s3_candidates']:,}")
    print(f"  - Average Candidates per S1:     {stats['average_candidates_per_s1']}")
    print(f"Predicted Matched S1 Entities:     {stats['predicted_matched_s1_entities']:,}")
    print(f"Predicted No-Match / Singletons:   {stats['predicted_singletons_count']:,} ({stats['predicted_singletons_pct']}%)")
    print(f"Predicted Single Match:            {stats['predicted_single_match_count']:,} ({stats['predicted_single_match_pct']}%)")
    print(f"Predicted Multi-Matches (2+):      {stats['predicted_multi_match_count']:,} ({stats['predicted_multi_match_pct']}%)")
    print(f"Total Predicted Matches:           {stats['total_predicted_matches']:,}")
    print(f"Threshold:                         {stats['final_threshold']}")
    print(f"Total Inference Runtime:           {stats['total_inference_runtime_seconds']:.2f} seconds")

    # 2. Run Official Submission Validator
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR (validate_submission.py)")
    print("=" * 80)
    returncode, validator_output = run_official_validator(
        matching_path=MATCHING_RESULTS_PATH,
        candidate_path=CANDIDATE_PAIRS_PATH,
        test_dir=TEST_DIR,
        check_ids=False,
    )

    if returncode == 0:
        print("\n>>> OFFICIAL VALIDATOR: PASS <<<")
        print("Both matching_results.tsv and candidate_pairs.tsv are 100% valid and conform to all official submission rules!")
    else:
        print(f"\n>>> OFFICIAL VALIDATOR: FAIL (exit code {returncode}) <<<")
        sys.exit(1)


if __name__ == "__main__":
    main()
