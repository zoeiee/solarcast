import pytest
import numpy as np

def generate_gradcam_heatmap(model, input_tensor):
    return np.random.rand(input_tensor.shape[2], input_tensor.shape[3])

def overlay_heatmap(image, heatmap):
    return np.zeros((image.shape[0], image.shape[1], 3))

def explain_prediction(model, input_tensor):
    heatmap = generate_gradcam_heatmap(model, input_tensor)
    overlay = overlay_heatmap(input_tensor.numpy()[0,0], heatmap)
    return {
        "heatmap": heatmap,
        "overlay": overlay
    }

def test_gradcam_output_shape(trained_model):
    import torch
    x = torch.randn(1, 1, 128, 128)
    heatmap = generate_gradcam_heatmap(trained_model, x)
    assert heatmap.shape == (128, 128)

def test_gradcam_value_range(trained_model):
    import torch
    x = torch.randn(1, 1, 128, 128)
    heatmap = generate_gradcam_heatmap(trained_model, x)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0

def test_gradcam_overlay_shape(trained_model):
    import torch
    x = torch.randn(1, 1, 128, 128)
    heatmap = generate_gradcam_heatmap(trained_model, x)
    overlay = overlay_heatmap(x.numpy()[0,0], heatmap)
    assert overlay.shape == (128, 128, 3)

def test_explain_prediction_keys(trained_model):
    import torch
    x = torch.randn(1, 1, 128, 128)
    explanation = explain_prediction(trained_model, x)
    assert "heatmap" in explanation
    assert "overlay" in explanation

def test_gradcam_different_inputs(trained_model):
    import torch
    x1 = torch.randn(1, 1, 128, 128)
    x2 = torch.randn(1, 1, 128, 128)
    h1 = generate_gradcam_heatmap(trained_model, x1)
    h2 = generate_gradcam_heatmap(trained_model, x2)
    # mock generates random, so should be different
    assert not np.array_equal(h1, h2)
