from fastapi import APIRouter
import os
import json
import time
from ..model_manager import ModelManager

router = APIRouter(tags=["model_info"])

@router.get("/health")
async def health_check():
    manager = ModelManager.get_instance()
    return {
        "status": "healthy",
        "model_loaded": manager.is_loaded(),
        "timestamp": time.time()
    }

@router.get("/model-info")
async def get_model_info():
    manager = ModelManager.get_instance()
    return manager.get_model_info()

@router.get("/demo-images")
async def get_demo_images():
    static_dir = os.path.join(os.path.dirname(__file__), "..", "static", "samples")
    images = []
    if os.path.exists(static_dir):
        for idx, filename in enumerate(os.listdir(static_dir)):
            if filename.endswith(('.png', '.jpg', '.jpeg', '.npy', '.fits')):
                images.append({
                    "id": f"sample_{idx}",
                    "name": filename,
                    "filename": filename,
                    "url": f"/static/samples/{filename}"
                })
    if not images:
        images = [
            {"id": "quiet_001", "name": "Quiet Region", "filename": "quiet_001.png", "url": "/static/samples/quiet_001.png"},
            {"id": "active_001", "name": "Active Region", "filename": "active_001.png", "url": "/static/samples/active_001.png"}
        ]
    return images

@router.get("/performance")
async def get_performance():
    metrics_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
            
    return {
        "is_synthetic": True,
        "note": "No actual metrics file found. Returning synthetic data.",
        "accuracy": 0.85,
        "f1_score": 0.78,
        "confusion_matrix": [[800, 50], [100, 150]],
        "roc_auc": 0.88,
        "pr_auc": 0.75,
        "per_class_metrics": {
            "no_flare": {"precision": 0.89, "recall": 0.94, "f1": 0.91},
            "flare": {"precision": 0.75, "recall": 0.60, "f1": 0.67}
        }
    }
