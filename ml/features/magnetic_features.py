import numpy as np
import pandas as pd
from typing import Dict, List
import scipy.ndimage as ndimage

# Note: These are simplified approximations of magnetic features, not official SHARP parameters.

def mean_field(img: np.ndarray) -> float:
    return float(np.nanmean(img))

def std_field(img: np.ndarray) -> float:
    return float(np.nanstd(img))

def mean_abs_field(img: np.ndarray) -> float:
    return float(np.nanmean(np.abs(img)))

def max_abs_field(img: np.ndarray) -> float:
    return float(np.nanmax(np.abs(img)))

def positive_flux(img: np.ndarray, pixel_area: float = 1.0) -> float:
    pos_pixels = img[img > 0]
    return float(np.sum(pos_pixels) * pixel_area)

def negative_flux(img: np.ndarray, pixel_area: float = 1.0) -> float:
    neg_pixels = img[img < 0]
    return float(np.sum(neg_pixels) * pixel_area)

def total_unsigned_flux(img: np.ndarray, pixel_area: float = 1.0) -> float:
    return float(np.sum(np.abs(img)) * pixel_area)

def net_flux(img: np.ndarray, pixel_area: float = 1.0) -> float:
    return positive_flux(img, pixel_area) + negative_flux(img, pixel_area)

def flux_imbalance(img: np.ndarray) -> float:
    total_flux = total_unsigned_flux(img)
    if total_flux == 0:
        return 0.0
    return float(np.abs(net_flux(img)) / total_flux)

def strong_field_area_ratio(img: np.ndarray, threshold: float = 300.0) -> float:
    total_pixels = img.size
    if total_pixels == 0:
        return 0.0
    strong_pixels = np.sum(np.abs(img) > threshold)
    return float(strong_pixels / total_pixels)

def get_gradient_magnitude(img: np.ndarray) -> np.ndarray:
    img_clean = np.nan_to_num(img, nan=0.0)
    grad_x = ndimage.sobel(img_clean, axis=1)
    grad_y = ndimage.sobel(img_clean, axis=0)
    return np.hypot(grad_x, grad_y)

def gradient_mean(img: np.ndarray) -> float:
    grad_mag = get_gradient_magnitude(img)
    return float(np.mean(grad_mag))

def gradient_max(img: np.ndarray) -> float:
    grad_mag = get_gradient_magnitude(img)
    return float(np.max(grad_mag))

def extract_all_features(img: np.ndarray) -> Dict[str, float]:
    img_clean = img.astype(float)
    return {
        'mean_field': mean_field(img_clean),
        'std_field': std_field(img_clean),
        'mean_abs_field': mean_abs_field(img_clean),
        'max_abs_field': max_abs_field(img_clean),
        'positive_flux': positive_flux(img_clean),
        'negative_flux': negative_flux(img_clean),
        'total_unsigned_flux': total_unsigned_flux(img_clean),
        'net_flux': net_flux(img_clean),
        'flux_imbalance': flux_imbalance(img_clean),
        'strong_field_area_ratio': strong_field_area_ratio(img_clean),
        'gradient_mean': gradient_mean(img_clean),
        'gradient_max': gradient_max(img_clean)
    }

def extract_features_batch(images: List[np.ndarray]) -> pd.DataFrame:
    features_list = [extract_all_features(img) for img in images]
    return pd.DataFrame(features_list)
