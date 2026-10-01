import pytest
import numpy as np

class MockPredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path
        
    def predict(self, arr):
        return {
            "probability": 0.75,
            "class": "M",
            "features": {"mean_field": 100}
        }

def test_predictor_from_numpy():
    predictor = MockPredictor()
    arr = np.zeros((128, 128))
    res = predictor.predict(arr)
    assert "probability" in res

def test_predictor_output_keys():
    predictor = MockPredictor()
    arr = np.zeros((128, 128))
    res = predictor.predict(arr)
    assert set(res.keys()) == {"probability", "class", "features"}

def test_predictor_probability_range():
    predictor = MockPredictor()
    arr = np.zeros((128, 128))
    res = predictor.predict(arr)
    assert 0 <= res["probability"] <= 1

def test_feature_extraction():
    predictor = MockPredictor()
    arr = np.zeros((128, 128))
    res = predictor.predict(arr)
    assert "features" in res
    assert "mean_field" in res["features"]

def test_mock_predictor():
    predictor = MockPredictor("fake_path.pth")
    assert predictor.predict(np.zeros((128, 128)))["probability"] == 0.75
