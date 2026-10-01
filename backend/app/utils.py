from fastapi import UploadFile
import numpy as np
import base64
import io
import cv2
from PIL import Image

async def load_image_from_upload(file: UploadFile) -> np.ndarray:
    content = await file.read()
    filename = file.filename.lower()
    
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image = Image.open(io.BytesIO(content)).convert('L')
        img_array = np.array(image, dtype=np.float32)
        img_array /= 255.0
        return img_array
    elif filename.endswith('.npy'):
        return np.load(io.BytesIO(content))
    elif filename.endswith('.fits'):
        try:
            from astropy.io import fits
            with fits.open(io.BytesIO(content)) as hdul:
                for hdu in hdul:
                    if hasattr(hdu, 'data') and hdu.data is not None:
                        return hdu.data.astype(np.float32)
        except ImportError:
            raise Exception("astropy is required to process FITS files")
            
    raise ValueError(f"Unsupported file format: {filename}")

def numpy_to_base64_png(array: np.ndarray, colormap=None) -> str:
    if array.dtype != np.uint8:
        array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    
    if colormap is not None:
        array = cv2.applyColorMap(array, colormap)
        
    img = Image.fromarray(array)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def create_heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, alpha=0.4) -> np.ndarray:
    if image.dtype != np.uint8:
        image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        img_uint8 = (image_norm * 255).astype(np.uint8)
    else:
        img_uint8 = image
        
    if len(img_uint8.shape) == 2:
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2RGB)
    else:
        img_color = img_uint8

    if heatmap.dtype != np.uint8:
        hm_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        hm_uint8 = (hm_norm * 255).astype(np.uint8)
    else:
        hm_uint8 = heatmap
        
    if hm_uint8.shape[:2] != img_color.shape[:2]:
        hm_uint8 = cv2.resize(hm_uint8, (img_color.shape[1], img_color.shape[0]))
        
    heatmap_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    overlay = cv2.addWeighted(img_color, 1 - alpha, heatmap_color, alpha, 0)
    return overlay

def get_confidence_level(probability: float) -> str:
    if probability > 0.8 or probability < 0.2:
        return 'high'
    elif probability > 0.6 or probability < 0.4:
        return 'moderate'
    return 'low'

def validate_image(image: np.ndarray) -> bool:
    if len(image.shape) not in [2, 3]:
        return False
    if image.size == 0:
        return False
    return True
