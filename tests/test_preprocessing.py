import pytest
import numpy as np
import warnings

def clip_field_values(image, min_val=-1500, max_val=1500):
    return np.clip(image, min_val, max_val)

def handle_missing_pixels(image):
    img = np.copy(image)
    img[np.isnan(img)] = 0
    img[np.isinf(img)] = 0
    return img

def resize_magnetogram(image, size=(128, 128)):
    # Mock resize for test purpose
    return np.zeros(size)

def minmax_normalize(image):
    img_min = image.min()
    img_max = image.max()
    if img_max - img_min == 0:
        return image - img_min
    return (image - img_min) / (img_max - img_min)

def standard_normalize(image):
    return (image - image.mean()) / (image.std() + 1e-8)

def prepare_for_model(image):
    img = handle_missing_pixels(image)
    img = clip_field_values(img)
    img = minmax_normalize(img)
    img = resize_magnetogram(img)
    return img.astype(np.float32)

def test_clip_field_values():
    img = np.array([-2000, 0, 2000])
    clipped = clip_field_values(img)
    assert np.array_equal(clipped, [-1500, 0, 1500])

def test_handle_missing_pixels_nan():
    img = np.array([np.nan, 1.0, 2.0])
    cleaned = handle_missing_pixels(img)
    assert not np.isnan(cleaned).any()
    assert cleaned[0] == 0

def test_handle_missing_pixels_inf():
    img = np.array([np.inf, -np.inf, 2.0])
    cleaned = handle_missing_pixels(img)
    assert not np.isinf(cleaned).any()
    assert cleaned[0] == 0
    assert cleaned[1] == 0

def test_resize_magnetogram():
    img = np.zeros((256, 256))
    resized = resize_magnetogram(img, (128, 128))
    assert resized.shape == (128, 128)

def test_minmax_normalize():
    img = np.array([-100, 0, 100])
    norm = minmax_normalize(img)
    assert norm.min() >= 0.0
    assert norm.max() <= 1.0

def test_standard_normalize():
    img = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    norm = standard_normalize(img)
    assert np.isclose(norm.mean(), 0, atol=1e-5)
    assert np.isclose(norm.std(), 1, atol=1e-5)

def test_prepare_for_model(sample_magnetogram):
    sample_magnetogram[0, 0] = np.nan
    sample_magnetogram[1, 1] = 2000
    out = prepare_for_model(sample_magnetogram)
    assert out.shape == (128, 128)
    assert out.dtype == np.float32
    assert not np.isnan(out).any()
    assert out.min() >= 0.0
    assert out.max() <= 1.0
