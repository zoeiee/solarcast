import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.cm as cm
from typing import Dict, Any

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from PIL import Image


class GradCAM:
    """Gradient-weighted Class Activation Mapping for CNN interpretability."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.forward_handle = self.target_layer.register_forward_hook(forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        """Generate Grad-CAM heatmap for the given input.

        Steps:
        1. Forward pass to get prediction and activations
        2. Backward pass to get gradients of target class w.r.t. target layer
        3. Global average pool gradients to get channel weights
        4. Weighted sum of activation maps
        5. ReLU to keep only positive contributions
        6. Resize heatmap to input image size
        """
        self.model.eval()
        self.model.zero_grad()

        # Enable gradient computation for this pass
        input_tensor = input_tensor.requires_grad_(True)
        logits = self.model(input_tensor)

        if logits.dim() > 1:
            score = logits[:, 0]
        else:
            score = logits

        score.backward(retain_graph=True)

        gradients = self.gradients.cpu().numpy()[0]  # (C, H, W)
        activations = self.activations.cpu().numpy()[0]  # (C, H, W)

        # Global average pool gradients to get channel weights
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weighted sum of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU - keep only positive contributions
        cam = np.maximum(cam, 0)

        # Normalize to [0, 1]
        if np.max(cam) != 0:
            cam = cam / np.max(cam)

        # Resize to input spatial dimensions
        h, w_dim = input_tensor.shape[2], input_tensor.shape[3]
        cam_resized = np.array(Image.fromarray(cam).resize((w_dim, h), Image.BILINEAR))

        return cam_resized

    def __del__(self):
        if hasattr(self, 'forward_handle'):
            self.forward_handle.remove()
        if hasattr(self, 'backward_handle'):
            self.backward_handle.remove()


def generate_gradcam_overlay(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> np.ndarray:
    """Overlay Grad-CAM heatmap on the original image."""
    # Normalize image to [0, 255] uint8
    if image.max() <= 1.0:
        image_uint8 = (image * 255).astype(np.uint8)
    else:
        image_uint8 = image.astype(np.uint8)

    # Ensure 3-channel
    if len(image_uint8.shape) == 2:
        image_rgb = np.stack([image_uint8] * 3, axis=-1)
    elif image_uint8.shape[2] == 1:
        image_rgb = np.concatenate([image_uint8] * 3, axis=-1)
    else:
        image_rgb = image_uint8

    # Apply colormap to heatmap
    cmap = cm.get_cmap(colormap)
    heatmap_normalized = heatmap if heatmap.max() <= 1.0 else heatmap / 255.0
    heatmap_colored = (cmap(heatmap_normalized)[:, :, :3] * 255).astype(np.uint8)

    # Blend
    overlay = (alpha * heatmap_colored + (1 - alpha) * image_rgb).astype(np.uint8)
    return overlay


def explain_prediction(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: torch.nn.Module
) -> Dict[str, Any]:
    """Generate a complete explanation for a single prediction."""
    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate(image_tensor)

    with torch.no_grad():
        logits = model(image_tensor)
        prob = torch.sigmoid(logits).item()

    original_image = image_tensor.cpu().numpy()[0, 0]
    overlay = generate_gradcam_overlay(original_image, heatmap)

    return {
        'original_image': original_image,
        'heatmap': heatmap,
        'overlay': overlay,
        'prediction_probability': prob,
        'predicted_class': int(prob >= 0.5)
    }
