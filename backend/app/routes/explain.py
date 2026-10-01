from fastapi import APIRouter, File, UploadFile, HTTPException
import logging
from ..utils import load_image_from_upload
from ..model_manager import ModelManager

router = APIRouter(prefix="/explain", tags=["explain"])
logger = logging.getLogger(__name__)

@router.post("")
async def explain_prediction(file: UploadFile = File(...)):
    try:
        image = await load_image_from_upload(file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")
        
    manager = ModelManager.get_instance()
    try:
        result = manager.explain(image)
        return result
    except Exception as e:
        logger.error(f"Error during explanation: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
