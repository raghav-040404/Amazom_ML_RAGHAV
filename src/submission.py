"""
Submission Verification & Packaging Module.
Amazon ML Challenge 2026.
Strictly checks compliance against official challenge rules.
"""

import os
import sys
import subprocess
from typing import Tuple, List, Dict, Any

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import (
    MATCHING_RESULTS_PATH,
    CANDIDATE_PAIRS_PATH,
    TEST_DIR,
    BASE_DIR,
)

VALIDATOR_SCRIPT = os.path.join(BASE_DIR, "datasett11", "student_resource", "utils", "validate_submission.py")


def run_official_validator(
    matching_path: str = MATCHING_RESULTS_PATH,
    candidate_path: str = CANDIDATE_PAIRS_PATH,
    test_dir: str = TEST_DIR,
    check_ids: bool = False,
) -> Tuple[int, str]:
    """
    Execute the official validate_submission.py script.
    """
    cmd = [
        sys.executable,
        VALIDATOR_SCRIPT,
        "--matching", matching_path,
        "--candidate", candidate_path,
        "--test-dir", test_dir,
    ]
    if check_ids:
        cmd.append("--check-ids")

    print(f"Running submission validator: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("\n--- VALIDATOR OUTPUT ---")
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
        
    return result.returncode, result.stdout
