import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.calibration import calibration_curve
import os

plt.style.use('dark_background')

def plot_confusion_matrix(cm: np.ndarray, labels: list, save_path: str):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='cyan', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='magenta', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_calibration_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int, save_path: str):
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    plt.figure(figsize=(6, 5))
    plt.plot(prob_pred, prob_true, marker='o', color='yellow', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history: dict, save_path: str):
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.get('train_loss', []), label='Train Loss')
    plt.plot(history.get('val_loss', []), label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if 'val_tss' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history.get('val_tss', []), color='cyan', label='Val TSS')
        plt.title('TSS History')
        plt.xlabel('Epoch')
        plt.ylabel('TSS')
        plt.legend()
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_distribution(labels: list, save_path: str):
    plt.figure(figsize=(6, 5))
    sns.countplot(x=labels, palette='viridis')
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_metric_comparison(results_dict: dict, save_path: str):
    # results_dict format: {'ModelName': {'tss': 0.8, 'hss': 0.7, ...}}
    models = list(results_dict.keys())
    metrics = list(results_dict[models[0]].keys())
    
    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    
    plt.figure(figsize=(10, 6))
    for i, model in enumerate(models):
        scores = [results_dict[model].get(m, 0) for m in metrics]
        plt.bar(x + i*width - 0.4 + width/2, scores, width, label=model)
        
    plt.xticks(x, metrics)
    plt.title('Model Comparison')
    plt.ylabel('Score')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
