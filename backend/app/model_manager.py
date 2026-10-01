import numpy as np
import logging
from .utils import numpy_to_base64_png, create_heatmap_overlay, get_confidence_level, validate_image
import os
import sys

logger = logging.getLogger(__name__)

class MockPredictor:
    def __init__(self):
        self.is_mock = True
        
    def _compute_features(self, image: np.ndarray) -> dict:
        mean_abs = float(np.mean(np.abs(image))) * 1000.0
        total_flux = float(np.sum(np.abs(image))) * 10.0
        imbalance = float(np.abs(np.sum(image[image > 0]) - np.abs(np.sum(image[image < 0]))) / (total_flux + 1e-8))
        
        gy, gx = np.gradient(image)
        grad_mag = np.sqrt(gy**2 + gx**2)
        grad_mean = float(np.mean(grad_mag)) * 500.0
        
        strong_field = np.sum(np.abs(image) > 0.5)
        strong_field_ratio = float(strong_field / max(1, image.size))
        
        return {
            "mean_abs_field": round(mean_abs, 2),
            "total_unsigned_flux": round(total_flux, 2),
            "flux_imbalance": round(imbalance, 4),
            "gradient_mean": round(grad_mean, 2),
            "strong_field_area_ratio": round(strong_field_ratio, 4)
        }

    def predict(self, image: np.ndarray) -> dict:
        features = self._compute_features(image)
        
        complexity = (features["gradient_mean"] / 100.0) + features["strong_field_area_ratio"] * 10.0
        prob = float(np.clip(complexity * 0.3 + 0.1, 0.05, 0.95))
        
        is_flare = prob >= 0.5
        
        return {
            "prediction": "flare" if is_flare else "no_flare",
            "probability": round(prob, 4),
            "threshold": 0.5,
            "prediction_window_hours": 24,
            "confidence_level": get_confidence_level(prob),
            "magnetic_features": features,
            "model_loaded": False,
            "note": "Using mock predictions. Train a model for real inference."
        }

    def explain(self, image: np.ndarray) -> dict:
        pred_result = self.predict(image)
        
        gy, gx = np.gradient(image)
        grad_mag = np.sqrt(gy**2 + gx**2)
        heatmap = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-8)
        
        from scipy.ndimage import gaussian_filter
        heatmap = gaussian_filter(heatmap, sigma=5)
        
        heatmap_b64 = numpy_to_base64_png(heatmap, colormap=None)
        
        overlay = create_heatmap_overlay(image, heatmap, alpha=0.4)
        overlay_b64 = numpy_to_base64_png(overlay)
        
        pred_result.update({
            "heatmap": heatmap_b64,
            "overlay": overlay_b64,
            "explanation_note": "Highlighted regions indicate image areas that influenced the network's prediction. They do not prove physical causation."
        })
        return pred_result


class ModelManager:
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance
        
    def __init__(self):
        self.model = None
        self.mock = MockPredictor()
        
    def load_model(self, model_path: str):
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}. Using MockPredictor.")
                self.model = None
                return
                
            try:
                import torch
                from ml.models.solar_cnn import SolarCNN
                
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = SolarCNN()
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                model.eval()
                self.model = model
                logger.info(f"Loaded PyTorch model from {model_path}")
            except ImportError as e:
                logger.error(f"Could not import ml components: {e}")
                self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None

    def load_onnx_model(self, onnx_path: str):
        try:
            if not os.path.exists(onnx_path):
                logger.warning(f"ONNX model not found at {onnx_path}")
                return
                
            import onnxruntime as ort
            self.model = ort.InferenceSession(onnx_path)
            logger.info(f"Loaded ONNX model from {onnx_path}")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            
    def is_loaded(self) -> bool:
        return self.model is not None
        
    def predict(self, image: np.ndarray) -> dict:
        if not validate_image(image):
            raise ValueError("Invalid image for prediction")
            
        if not self.is_loaded():
            return self.mock.predict(image)
        
        try:
            import torch
            from ml.preprocessing.normalization import minmax_normalize
            
            img = minmax_normalize(image, clip_range=(-1500, 1500))
            if img.ndim == 2:
                img = img[np.newaxis, np.newaxis, :, :]
            elif img.ndim == 3:
                img = img[np.newaxis, :, :, :]
            
            tensor = torch.from_numpy(img).float()
            device = next(self.model.parameters()).device
            tensor = tensor.to(device)
            
            with torch.no_grad():
                logits = self.model(tensor)
                prob = float(torch.sigmoid(logits).cpu().item())
            
            features = self.mock._compute_features(image)
            is_flare = prob >= 0.5
            
            return {
                "prediction": "flare" if is_flare else "no_flare",
                "probability": round(prob, 4),
                "threshold": 0.5,
                "prediction_window_hours": 24,
                "confidence_level": get_confidence_level(prob),
                "magnetic_features": features,
                "model_loaded": True
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self.mock.predict(image)
        
    def explain(self, image: np.ndarray) -> dict:
        if not self.is_loaded():
            return self.mock.explain(image)
        
        try:
            import torch
            from ml.explainability.gradcam import GradCAM, generate_gradcam_overlay
            from ml.preprocessing.normalization import minmax_normalize
            
            img = minmax_normalize(image, clip_range=(-1500, 1500))
            if img.ndim == 2:
                img = img[np.newaxis, np.newaxis, :, :]
            elif img.ndim == 3:
                img = img[np.newaxis, :, :, :]
                
            tensor = torch.from_numpy(img).float()
            device = next(self.model.parameters()).device
            tensor = tensor.to(device)
            
            target_layer = self.model.layer3
            grad_cam = GradCAM(self.model, target_layer)
            heatmap = grad_cam.generate(tensor)
            
            with torch.no_grad():
                logits = self.model(tensor)
                prob = float(torch.sigmoid(logits).cpu().item())
            
            original_img = img[0, 0]
            overlay = create_heatmap_overlay(original_img, heatmap, alpha=0.4)
            
            heatmap_b64 = numpy_to_base64_png(heatmap, colormap=None)
            overlay_b64 = numpy_to_base64_png(overlay)
            
            features = self.mock._compute_features(image)
            is_flare = prob >= 0.5
            
            return {
                "prediction": "flare" if is_flare else "no_flare",
                "probability": round(prob, 4),
                "threshold": 0.5,
                "prediction_window_hours": 24,
                "confidence_level": get_confidence_level(prob),
                "magnetic_features": features,
                "model_loaded": True,
                "heatmap": heatmap_b64,
                "overlay": overlay_b64,
                "explanation_note": "Highlighted regions indicate image areas that influenced the network's prediction. They do not prove physical causation."
            }
        except Exception as e:
            logger.error(f"Explanation error: {e}")
            return self.mock.explain(image)
        
    def extract_features(self, image: np.ndarray) -> dict:
        try:
            from ml.features.magnetic_features import extract_all_features
            return extract_all_features(image)
        except ImportError:
            return self.mock._compute_features(image)
        
    def get_model_info(self) -> dict:
        if not self.is_loaded():
            return {
                "model_name": "MockSolarCNN",
                "model_type": "mock",
                "input_size": [1, 128, 128],
                "num_parameters": 0,
                "prediction_target": "M/X flare within 24 hours",
                "classification_type": "binary",
                "threshold": 0.5,
                "training_info": {
                    "dataset": "Mock Data",
                    "split_strategy": "none",
                    "loss_function": "none"
                },
                "note": "This is a mock model because no weights were loaded."
            }
            
        return {
            "model_name": "SolarCNN",
            "model_type": "custom_cnn",
            "input_size": [1, 128, 128],
            "num_parameters": 1241569,
            "prediction_target": "M/X flare within 24 hours",
            "classification_type": "binary",
            "threshold": 0.5,
            "training_info": {
                "dataset": "SDO/HMI Magnetograms",
                "split_strategy": "active_region",
                "loss_function": "weighted_bce"
            }
        }
