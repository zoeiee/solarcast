from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import List
import logging
from ..utils import load_image_from_upload
from ..model_manager import ModelManager

router = APIRouter(prefix="/predict", tags=["predict"])
logger = logging.getLogger(__name__)

@router.post("")
async def predict_single(file: UploadFile = File(...)):
    try:
        image = await load_image_from_upload(file)
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")
    
    manager = ModelManager.get_instance()
    try:
        result = manager.predict(image)
        return result
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    manager = ModelManager.get_instance()
    results = []
    for file in files:
        try:
            image = await load_image_from_upload(file)
            result = manager.predict(image)
            result["filename"] = file.filename
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")
            results.append({"filename": file.filename, "error": str(e)})
            
    return {"predictions": results}
