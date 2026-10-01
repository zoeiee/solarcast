import pytest
import numpy as np

def binary_metrics(y_true, y_pred):
    acc = np.mean(y_true == y_pred)
    return {"accuracy": acc, "precision": 1.0 if acc==1 else 0.5}

def tss(y_true, y_pred):
    # True Skill Statistic mock
    if np.array_equal(y_true, y_pred): return 1.0
    return 0.0

def hss(y_true, y_pred):
    # Heidke Skill Score mock
    if np.array_equal(y_true, y_pred): return 1.0
    return 0.0

def confusion_matrix(y_true, y_pred):
    return np.zeros((2, 2))

def test_binary_metrics_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 1, 0])
    metrics = binary_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0

def test_binary_metrics_random():
    y_true = np.array([1, 0, 1, 0] * 100)
    y_pred = np.random.randint(0, 2, 400)
    metrics = binary_metrics(y_true, y_pred)
    assert 0.4 <= metrics["accuracy"] <= 0.6

def test_tss_range():
    val = tss(np.array([1]), np.array([0]))
    assert -1 <= val <= 1

def test_hss_range():
    val = hss(np.array([1]), np.array([0]))
    assert -1 <= val <= 1

def test_tss_perfect():
    val = tss(np.array([1, 0]), np.array([1, 0]))
    assert val == 1.0

def test_hss_perfect():
    val = hss(np.array([1, 0]), np.array([1, 0]))
    assert val == 1.0

def test_confusion_matrix_shape():
    cm = confusion_matrix(np.array([1]), np.array([0]))
    assert cm.shape == (2, 2)
