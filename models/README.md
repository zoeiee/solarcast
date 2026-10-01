# Models Directory

This directory is used to store trained machine learning model weights and exported inference graphs.

## Contents
Due to file size constraints, trained model weights (`*.pth`, `*.onnx`) are ignored by Git and are not included in the repository.

When you train a model using `scripts/train.py`, the outputs will be saved here:
- `solar_cnn_best.pth`: PyTorch checkpoint containing the model weights with the best validation performance.
- `solar_cnn_latest.pth`: PyTorch checkpoint of the final training epoch.
- `solar_cnn.onnx`: (Optional) The model exported to ONNX format for optimized inference in the backend.

## Training Instructions
To generate these models, please follow the instructions in the main project `README.md` and ensure you have generated sample data or downloaded real dataset files.

Example:
`python scripts/train.py --model cnn --config config.yaml`
