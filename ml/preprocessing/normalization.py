import numpy as np
from typing import Callable

def minmax_normalize(img: np.ndarray, clip_range: tuple = (-1500, 1500)) -> np.ndarray:
    img = np.clip(img, clip_range[0], clip_range[1])
    range_val = clip_range[1] - clip_range[0]
    return (img - clip_range[0]) / range_val

def standard_normalize(img: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (img - mean) / (std + 1e-8)

def robust_normalize(img: np.ndarray) -> np.ndarray:
    valid_pixels = img[~np.isnan(img) & ~np.isinf(img)]
    if len(valid_pixels) == 0:
        return np.zeros_like(img)
    median = np.median(valid_pixels)
    q75, q25 = np.percentile(valid_pixels, [75 ,25])
    iqr = q75 - q25
    if iqr == 0:
        iqr = 1e-8
    return (img - median) / iqr

def get_normalization_fn(method: str) -> Callable:
    if method == 'minmax':
        return minmax_normalize
    elif method == 'standard':
        return standard_normalize
    elif method == 'robust':
        return robust_normalize
    else:
        raise ValueError(f"Unknown normalization method: {method}")
