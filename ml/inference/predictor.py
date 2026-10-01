import torch
import numpy as np
import os
from PIL import Image
from typing import Dict, Any, Union
from ..config import Config, get_device
from ..preprocessing.image_processing import prepare_for_model
from ..features.magnetic_features import extract_all_features
from ..explainability.gradcam import explain_prediction

class SolarPredictor:
    def __init__(self, model_path: str, config: Config, use_onnx: bool = False):
        self.config = config
        self.device = get_device()
        self.use_onnx = use_onnx
        
        if self.use_onnx:
            self.model = self.load_onnx_model(model_path)
        else:
            self.model = self.load_pytorch_model(model_path)
            self.model.eval()

    def load_pytorch_model(self, model_path: str):
        from ..models.solar_cnn import SolarCNN
        model = SolarCNN().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        return model
        
    def load_onnx_model(self, model_path: str):
        import onnxruntime as ort
        return ort.InferenceSession(model_path)
        
    def preprocess(self, image_input: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(image_input, str):
            if image_input.endswith('.npy'):
                img = np.load(image_input)
            else:
                img = np.array(Image.open(image_input))
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        else:
            img = image_input
            
        tensor = prepare_for_model(img, size=(128, 128))
        return tensor.unsqueeze(0).to(self.device) # Add batch dim

    def predict(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        tensor = self.preprocess(image_input)
        threshold = self.config.get('threshold', 0.5)
        
        if self.use_onnx:
            inputs = {self.model.get_inputs()[0].name: tensor.cpu().numpy()}
            logits = self.model.run(None, inputs)[0]
            prob = float(1 / (1 + np.exp(-logits[0][0])))
        else:
            with torch.no_grad():
                logits = self.model(tensor)
                prob = torch.sigmoid(logits).item()
                
        return {
            'probability': prob,
            'predicted_class': int(prob >= threshold),
            'threshold': threshold
        }
        
    def predict_with_explanation(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, Any]:
        if self.use_onnx:
            raise NotImplementedError("Grad-CAM explanation is only supported for PyTorch models.")
            
        tensor = self.preprocess(image_input)
        # Assumes SolarCNN where layer3 is the last residual block
        target_layer = self.model.layer3
        
        result = explain_prediction(self.model, tensor, target_layer)
        result['threshold'] = self.config.get('threshold', 0.5)
        return result
        
    def extract_features(self, image_input: Union[str, np.ndarray, Image.Image]) -> Dict[str, float]:
        if isinstance(image_input, str):
            if image_input.endswith('.npy'):
                img = np.load(image_input)
            else:
                img = np.array(Image.open(image_input))
        elif isinstance(image_input, Image.Image):
            img = np.array(image_input)
        else:
            img = image_input
            
        return extract_all_features(img)
        
    def get_model_info(self) -> Dict[str, Any]:
        if self.use_onnx:
            return {'format': 'ONNX', 'name': 'SolarCNN'}
        else:
            num_params = sum(p.numel() for p in self.model.parameters())
            return {
                'format': 'PyTorch',
                'name': self.model.__class__.__name__,
                'parameters': num_params
            }
