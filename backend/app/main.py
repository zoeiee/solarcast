import os
import sys
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Add project root to sys.path to allow importing from ml package
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from .routes import predict, explain, model_info
from .model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SolarCast API",
    description="API for Solar Flare Prediction",
    version="1.0.0"
)

# CORS Middleware
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files directory
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(os.path.join(static_dir, "samples"), exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

app.include_router(predict.router)
app.include_router(explain.router)
app.include_router(model_info.router)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up SolarCast API...")
    # Initialize ModelManager
    manager = ModelManager.get_instance()
    model_path = os.getenv("MODEL_PATH", os.path.join(project_root, "models", "best_model.pth"))
    manager.load_model(model_path)

@app.get("/")
async def root():
    return {
        "title": "SolarCast API",
        "description": "Predicting M/X-class solar flares from magnetogram images.",
        "version": "1.0.0"
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)}
    )
