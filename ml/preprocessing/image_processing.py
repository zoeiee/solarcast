import numpy as np
from PIL import Image
import torch
import warnings

try:
    from astropy.io import fits
    HAS_FITS = True
except ImportError:
    HAS_FITS = False

def load_magnetogram(path: str) -> np.ndarray:
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith('.fits') or path.endswith('.fit'):
        if HAS_FITS:
            with fits.open(path) as hdul:
                return hdul[0].data
        else:
            raise ImportError("astropy is required to load FITS files.")
    else:
        raise ValueError("Unsupported file format. Use .npy or .fits")

def resize_magnetogram(img: np.ndarray, size: tuple) -> np.ndarray:
    img_pil = Image.fromarray(img)
    img_resized = img_pil.resize(size, Image.BILINEAR)
    return np.array(img_resized)

def crop_active_region(img: np.ndarray, center: tuple, size: tuple) -> np.ndarray:
    x_c, y_c = center
    w, h = size
    
    x1 = max(0, x_c - w // 2)
    y1 = max(0, y_c - h // 2)
    x2 = min(img.shape[1], x_c + w // 2)
    y2 = min(img.shape[0], y_c + h // 2)
    
    cropped = img[y1:y2, x1:x2]
    # Pad if necessary
    if cropped.shape != (h, w):
        pad_y = max(0, h - cropped.shape[0])
        pad_x = max(0, w - cropped.shape[1])
        cropped = np.pad(cropped, ((0, pad_y), (0, pad_x)), mode='constant')
        
    return cropped

def clip_field_values(img: np.ndarray, clip_range: tuple = (-1500, 1500)) -> np.ndarray:
    return np.clip(img, clip_range[0], clip_range[1])

def handle_missing_pixels(img: np.ndarray, method: str = 'zero') -> np.ndarray:
    img = img.copy()
    mask = np.isnan(img) | np.isinf(img)
    if method == 'zero':
        img[mask] = 0
    elif method == 'mean':
        mean_val = np.nanmean(img)
        img[mask] = mean_val if not np.isnan(mean_val) else 0
    return img

def prepare_for_model(img: np.ndarray, size: tuple = (128, 128)) -> torch.Tensor:
    img = clip_field_values(img)
    img = handle_missing_pixels(img)
    if img.shape != size:
        img = resize_magnetogram(img, size)
    
    # Normalize to [0, 1] for model input
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
        
    tensor = torch.from_numpy(img).float()
    return tensor.unsqueeze(0) # Add channel dim
