"""
Evaluation Metrics Module for Business Entity Resolution.
Amazon ML Challenge 2026.
Evaluates Macro F0.5, Pairwise F0.5, Precision, Recall, and Singleton Accuracy.
"""

from typing import Dict, Set, List, Tuple, Any
import numpy as np


def compute_entity_f05(pred_ids: Set[str], true_ids: Set[str], beta: float = 0.5) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_beta for a single Source 1 entity.
    
    Singletons:
    - If true_ids is empty and pred_ids is empty: P=1, R=1, F_beta=1.0 (True Negative singleton)
    - If true_ids is empty and pred_ids is non-empty: P=0, R=0, F_beta=0.0 (False Positive merge)
    - If true_ids is non-empty and pred_ids is empty: P=0, R=0, F_beta=0.0 (False Negative missed match)
    """
    if len(true_ids) == 0:
        if len(pred_ids) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if len(pred_ids) == 0:
        return 0.0, 0.0, 0.0

    tp = len(pred_ids & true_ids)
    fp = len(pred_ids - true_ids)
    fn = len(true_ids - pred_ids)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    beta_sq = beta ** 2
    if precision + recall == 0:
        f_beta = 0.0
    else:
        f_beta = (1.0 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall)

    return precision, recall, f_beta


def compute_macro_f05(
    predictions: Dict[str, Set[str]],
    ground_truth: Dict[str, Set[str]],
    beta: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute Macro-Averaged F0.5 across all Source 1 entities in the evaluation split.
    """
    all_s1_ids = list(ground_truth.keys())
    if not all_s1_ids:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0}

    precisions = []
    recalls = []
    f_scores = []
    singleton_correct = 0
    singleton_total = 0

    for s1_id in all_s1_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        
        p, r, f = compute_entity_f05(pred_set, true_set, beta=beta)
        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)

        if len(true_set) == 0:
            singleton_total += 1
            if len(pred_set) == 0:
                singleton_correct += 1

    return {
        "macro_f05": round(float(np.mean(f_scores)), 4),
        "macro_precision": round(float(np.mean(precisions)), 4),
        "macro_recall": round(float(np.mean(recalls)), 4),
        "total_evaluated_s1": len(all_s1_ids),
        "singletons_total": singleton_total,
        "singletons_correct": singleton_correct,
        "singleton_accuracy": round((singleton_correct / max(1, singleton_total)) * 100, 2),
    }


def compute_pairwise_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    beta: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute pairwise classification metrics on candidate pairs.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    beta_sq = beta ** 2
    f_beta = (1.0 + beta_sq) * (precision * recall) / ((beta_sq * precision) + recall) if (precision + recall) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f05": round(f_beta, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_pairs": len(y_true),
    }
