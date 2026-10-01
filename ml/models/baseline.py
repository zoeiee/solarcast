import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class MagneticFeatureClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced'
        )
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.model.fit(X, y)
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

def train_baseline(features_df: pd.DataFrame, labels: np.ndarray) -> Tuple[Any, Dict[str, float]]:
    model = MagneticFeatureClassifier()
    model.fit(features_df, labels)
    metrics = evaluate_baseline(model, features_df, labels)
    return model, metrics

def evaluate_baseline(model: Any, features_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    preds = model.predict(features_df)
    probs = model.predict_proba(features_df)
    
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
        'roc_auc': roc_auc_score(labels, probs)
    }
    return metrics

class ShallowCNN(nn.Module):
    def __init__(self):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
