import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from typing import Dict, Tuple

def true_skill_statistic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """TSS = TPR - FPR = recall - false_positive_rate."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    return float(tpr - fpr)

def heidke_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """HSS = 2*(TP*TN - FP*FN) / ((TP+FN)*(FN+TN) + (TP+FP)*(FP+TN))"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    num = 2 * (tp * tn - fp * fn)
    den = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    if den == 0:
        return 0.0
    return float(num / den)

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0,1])

def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0
    
    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'roc_auc': float(roc_auc),
        'pr_auc': float(average_precision_score(y_true, y_prob)),
        'tss': true_skill_statistic(y_true, y_pred),
        'hss': heidke_skill_score(y_true, y_pred)
    }

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, metric: str = 'tss') -> float:
    thresholds = np.linspace(0.01, 0.99, 99)
    best_thresh = 0.5
    best_score = -np.inf
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == 'tss':
            score = true_skill_statistic(y_true, y_pred)
        elif metric == 'hss':
            score = heidke_skill_score(y_true, y_pred)
        elif metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unsupported metric for threshold optimization: {metric}")
            
        if score > best_score:
            best_score = score
            best_thresh = t
            
    return float(best_thresh)

def format_metrics_report(metrics_dict: Dict[str, float]) -> str:
    report = "=== Evaluation Metrics ===\n"
    for k, v in metrics_dict.items():
        report += f"{k.upper():<12}: {v:.4f}\n"
    return report
