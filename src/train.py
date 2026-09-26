"""
Training & Model Comparison Module for Business Entity Resolution.
Amazon ML Challenge 2026.
Strictly processes candidate pairs originating from `datasett11`.
"""

import os
import sys
import json
import time
import pickle
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.config import CONFIG, OUTPUT_DIR, ARTIFACTS_DIR
from src.features import EntityFeatureExtractor
from src.evaluate import compute_macro_f05, compute_pairwise_metrics

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_s1_ids: List[str],
    X_val: np.ndarray,
    y_val: np.ndarray,
    val_s1_ids: List[str],
    val_cand_ids: List[str],
    val_ground_truth: Dict[str, set],
    thresholds: List[float] = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95],
) -> Dict[str, Any]:
    """
    Train baseline Logistic Regression and Tree-based model (HistGradientBoosting / XGBoost),
    sweep decision thresholds, and compute Pairwise + Macro F0.5 metrics.
    """
    feature_names = EntityFeatureExtractor.FEATURE_NAMES
    results = {}
    
    # 1. Feature Scaling for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # =========================================================================
    # Model 1: Logistic Regression (Balanced Class Weights)
    # =========================================================================
    print("\n--- Training Model 1: Logistic Regression ---")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=CONFIG.random_seed)
    lr.fit(X_train_scaled, y_train)
    
    val_probs_lr = lr.predict_proba(X_val_scaled)[:, 1]
    
    lr_best_thresh = 0.5
    lr_best_macro_f05 = -1.0
    lr_thresh_results = []
    
    for thresh in thresholds:
        # Group predictions by S1 ID
        pred_map = {s1: set() for s1 in val_ground_truth.keys()}
        for s1, cid, p in zip(val_s1_ids, val_cand_ids, val_probs_lr):
            if p >= thresh:
                if s1 in pred_map:
                    pred_map[s1].add(cid)
                    
        macro_res = compute_macro_f05(pred_map, val_ground_truth, beta=0.5)
        pair_res = compute_pairwise_metrics(y_val, (val_probs_lr >= thresh).astype(int), beta=0.5)
        
        entry = {
            "threshold": thresh,
            "macro_f05": macro_res["macro_f05"],
            "macro_precision": macro_res["macro_precision"],
            "macro_recall": macro_res["macro_recall"],
            "pair_f05": pair_res["f05"],
            "pair_precision": pair_res["precision"],
            "pair_recall": pair_res["recall"],
            "singletons_correct": macro_res["singletons_correct"],
            "singleton_accuracy": macro_res["singleton_accuracy"],
        }
        lr_thresh_results.append(entry)
        if macro_res["macro_f05"] > lr_best_macro_f05:
            lr_best_macro_f05 = macro_res["macro_f05"]
            lr_best_thresh = thresh

    print(f"Logistic Regression Best Threshold: {lr_best_thresh} | Macro F0.5: {lr_best_macro_f05:.4f}")

    # =========================================================================
    # Model 2: Tree-Based Model (HistGradientBoosting / XGBoost)
    # =========================================================================
    print("\n--- Training Model 2: Gradient Boosted Trees ---")
    pos_weight = (len(y_train) - np.sum(y_train)) / max(1, np.sum(y_train))
    
    if XGB_AVAILABLE:
        print("Using XGBoost Classifier...")
        tree_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            scale_pos_weight=pos_weight,
            random_state=CONFIG.random_seed,
            n_jobs=-1,
        )
        tree_model.fit(X_train, y_train)
        val_probs_tree = tree_model.predict_proba(X_val)[:, 1]
        importances = tree_model.feature_importances_
    else:
        print("Using HistGradientBoostingClassifier...")
        tree_model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=8,
            learning_rate=0.08,
            class_weight="balanced",
            random_state=CONFIG.random_seed,
        )
        tree_model.fit(X_train, y_train)
        val_probs_tree = tree_model.predict_proba(X_val)[:, 1]
        # Feature importances via tree splits approximation or permutation
        importances = np.zeros(len(feature_names))

    tree_best_thresh = 0.5
    tree_best_macro_f05 = -1.0
    tree_thresh_results = []
    best_pred_map = {}

    for thresh in thresholds:
        pred_map = {s1: set() for s1 in val_ground_truth.keys()}
        for s1, cid, p in zip(val_s1_ids, val_cand_ids, val_probs_tree):
            if p >= thresh:
                if s1 in pred_map:
                    pred_map[s1].add(cid)
                    
        macro_res = compute_macro_f05(pred_map, val_ground_truth, beta=0.5)
        pair_res = compute_pairwise_metrics(y_val, (val_probs_tree >= thresh).astype(int), beta=0.5)
        
        entry = {
            "threshold": thresh,
            "macro_f05": macro_res["macro_f05"],
            "macro_precision": macro_res["macro_precision"],
            "macro_recall": macro_res["macro_recall"],
            "pair_f05": pair_res["f05"],
            "pair_precision": pair_res["precision"],
            "pair_recall": pair_res["recall"],
            "tp": pair_res["tp"],
            "fp": pair_res["fp"],
            "fn": pair_res["fn"],
            "singletons_correct": macro_res["singletons_correct"],
            "singleton_accuracy": macro_res["singleton_accuracy"],
        }
        tree_thresh_results.append(entry)
        if macro_res["macro_f05"] > tree_best_macro_f05:
            tree_best_macro_f05 = macro_res["macro_f05"]
            tree_best_thresh = thresh
            best_pred_map = pred_map

    print(f"Gradient Boosted Trees Best Threshold: {tree_best_thresh} | Macro F0.5: {tree_best_macro_f05:.4f}")

    # Ranked Feature Importance
    ranked_importances = []
    if np.sum(importances) > 0:
        indices = np.argsort(importances)[::-1]
        for idx in indices:
            ranked_importances.append({
                "feature": feature_names[idx],
                "importance": round(float(importances[idx]), 4),
            })
    else:
        # Logistic Regression coefficients as importance
        lr_coefs = np.abs(lr.coef_[0])
        indices = np.argsort(lr_coefs)[::-1]
        for idx in indices:
            ranked_importances.append({
                "feature": feature_names[idx],
                "coefficient_magnitude": round(float(lr_coefs[idx]), 4),
            })

    # Save artifacts
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(ARTIFACTS_DIR, "best_tree_model.pkl"), "wb") as f:
        pickle.dump(tree_model, f)
    with open(os.path.join(ARTIFACTS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    # Save validation results TSV
    val_results_path = os.path.join(OUTPUT_DIR, "validation_results.tsv")
    with open(val_results_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matches in best_pred_map.items():
            f.write(f"{s1_id}\t{','.join(sorted(matches))}\n")
    print(f"Saved validation predictions to {val_results_path}")

    return {
        "logistic_regression": {
            "best_threshold": lr_best_thresh,
            "best_macro_f05": lr_best_macro_f05,
            "threshold_sweep": lr_thresh_results,
        },
        "gradient_boosted_trees": {
            "best_threshold": tree_best_thresh,
            "best_macro_f05": tree_best_macro_f05,
            "threshold_sweep": tree_thresh_results,
            "feature_importance": ranked_importances,
        },
    }
