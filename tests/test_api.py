import pytest
import io
import json
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient
from fastapi import FastAPI, UploadFile, File

# Mocking the backend app for testing
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {"name": "SolarCNN", "version": "1.0", "description": "Flare prediction"}

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    if not file.filename.endswith(('.png', '.npy')):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Invalid file type")
    return {"probability": 0.85, "flare_class": "X"}

@app.post("/explain")
def explain(file: UploadFile = File(...)):
    return {"heatmap": [[0.1, 0.2], [0.3, 0.4]]}

@app.get("/demo-images")
def demo_images():
    return ["demo1.png", "demo2.png"]

client = TestClient(app)

@pytest.fixture
def png_file():
    img = Image.new('RGB', (128, 128), color='black')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

@pytest.fixture
def npy_file():
    arr = np.zeros((128, 128))
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return buf

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_model_info_endpoint():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "name" in response.json()

def test_predict_endpoint_png(png_file):
    files = {"file": ("test.png", png_file, "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200

def test_predict_endpoint_npy(npy_file):
    files = {"file": ("test.npy", npy_file, "application/octet-stream")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200

def test_predict_response_schema(png_file):
    files = {"file": ("test.png", png_file, "image/png")}
    response = client.post("/predict", files=files)
    data = response.json()
    assert "probability" in data

def test_explain_endpoint(png_file):
    files = {"file": ("test.png", png_file, "image/png")}
    response = client.post("/explain", files=files)
    assert response.status_code == 200
    assert "heatmap" in response.json()

def test_invalid_file_upload():
    buf = io.BytesIO(b"Not an image")
    files = {"file": ("test.txt", buf, "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400

def test_demo_images_endpoint():
    response = client.get("/demo-images")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
