import pytest
import numpy as np

def mean_field(img):
    return np.mean(np.abs(img))

def positive_negative_flux(img):
    return np.sum(img[img > 0]), np.sum(img[img < 0])

def total_unsigned_flux(img):
    return np.sum(np.abs(img))

def flux_imbalance_range(img):
    pos, neg = positive_negative_flux(img)
    total = total_unsigned_flux(img)
    if total == 0: return 0
    return abs(pos + neg) / total

def strong_field_area_ratio(img, threshold=500):
    total_area = img.size
    strong_area = np.sum(np.abs(img) > threshold)
    return strong_area / total_area

def gradient_features(img):
    gx, gy = np.gradient(img)
    grad_mag = np.sqrt(gx**2 + gy**2)
    return np.mean(grad_mag)

def extract_all_features(img):
    pos, neg = positive_negative_flux(img)
    return {
        "mean_field": mean_field(img),
        "positive_flux": pos,
        "negative_flux": neg,
        "total_unsigned_flux": total_unsigned_flux(img),
        "flux_imbalance": flux_imbalance_range(img),
        "strong_field_ratio": strong_field_area_ratio(img),
        "mean_gradient": gradient_features(img)
    }

def test_mean_field(sample_quiet_magnetogram):
    val = mean_field(sample_quiet_magnetogram)
    assert val >= 0

def test_positive_negative_flux(sample_flare_magnetogram):
    pos, neg = positive_negative_flux(sample_flare_magnetogram)
    assert pos > 0
    assert neg < 0

def test_total_unsigned_flux(sample_magnetogram):
    assert total_unsigned_flux(sample_magnetogram) >= 0

def test_flux_imbalance_range(sample_magnetogram):
    imbalance = flux_imbalance_range(sample_magnetogram)
    assert 0 <= imbalance <= 1

def test_strong_field_area_ratio(sample_flare_magnetogram):
    ratio = strong_field_area_ratio(sample_flare_magnetogram)
    assert 0 <= ratio <= 1

def test_gradient_features(sample_magnetogram):
    grad = gradient_features(sample_magnetogram)
    assert grad >= 0

def test_extract_all_features(sample_magnetogram):
    features = extract_all_features(sample_magnetogram)
    assert len(features) > 0
    for v in features.values():
        assert not np.isnan(v)

def test_extract_all_features_keys(sample_magnetogram):
    expected_keys = {
        "mean_field", "positive_flux", "negative_flux",
        "total_unsigned_flux", "flux_imbalance",
        "strong_field_ratio", "mean_gradient"
    }
    features = extract_all_features(sample_magnetogram)
    assert set(features.keys()) == expected_keys

def test_symmetric_magnetogram_balance():
    img = np.array([[100, -100], [100, -100]])
    imbalance = flux_imbalance_range(img)
    assert np.isclose(imbalance, 0.0)
