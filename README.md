# SolarCast — Solar Flare Prediction from Magnetograms
*(Python 3.9+ | PyTorch | FastAPI | Next.js)*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
SolarCast is an end-to-end machine learning pipeline and web application designed to predict the occurrence of major solar flares. Given a line-of-sight magnetogram image of a solar active region, the system predicts whether an M-class or X-class solar flare will occur within the following 24 hours.

## Scientific Motivation
Solar flares and associated coronal mass ejections drive space weather, which can disrupt satellite operations, communications, and power grids on Earth. Accurate forecasting of major flares (M and X class) is critical for mitigating these impacts and protecting technological infrastructure and astronaut safety.

## System Architecture
SolarCast consists of a PyTorch-based Deep Learning pipeline (SolarCNN), a FastAPI backend for inference, and a React/Next.js frontend for interactive visualization. For detailed architectural diagrams and data flow, see [Architecture Documentation](docs/architecture.md).

## Key Features
- **CNN-based magnetogram analysis**: Custom ResNet-style architecture for spatial feature extraction.
- **Binary M/X flare prediction**: 24-hour prediction window.
- **Grad-CAM explainability**: Visual heatmaps showing which magnetic features drove the prediction.
- **Interactive web application**: Upload magnetograms and get instant predictions.
- **Baseline model comparison**: Includes Random Forest baselines for performance comparison.
- **Space-weather evaluation metrics**: Evaluates using True Skill Statistic (TSS) and Heidke Skill Score (HSS).

## Dataset
The project is designed to work with SDO/HMI magnetograms. See [Dataset Documentation](docs/dataset.md) for instructions on downloading real data from JSOC.

## Quick Start
```bash
# Clone
git clone https://github.com/yourusername/solarcast.git
cd solarcast

# Generate sample data
python scripts/generate_sample_data.py

# Preprocess
python scripts/preprocess_data.py --data-dir data/sample --output-dir data/processed

# Train
python scripts/train.py --model cnn --config config.yaml

# Evaluate
python scripts/evaluate.py --model-path models/solar_cnn_best.pth --data-dir data/processed

# Start backend
cd backend && pip install -r requirements.txt && uvicorn app.main:app --reload

# Start frontend (new terminal)
cd frontend && npm install && npm run dev
```

## Installation
Detailed installation instructions:
1. **ML Pipeline**: Create a Python virtual environment, install requirements `pip install -r requirements.txt`.
2. **Backend**: Navigate to `backend/`, install requirements, setup `.env`.
3. **Frontend**: Navigate to `frontend/`, run `npm install`.

## Project Structure
```text
solarcast/
├── backend/            # FastAPI server
├── data/               # Raw, processed, and sample datasets
├── docs/               # Documentation files
├── frontend/           # Next.js web application
├── ml/                 # ML pipeline (models, training, evaluation)
├── models/             # Saved model checkpoints
├── scripts/            # Training, evaluation, and data scripts
├── tests/              # Test suite
├── TECHNICAL_REPORT.md # Comprehensive technical report
└── README.md
```

## Model Architecture
**SolarCNN** is a custom convolutional neural network utilizing residual blocks to mitigate vanishing gradients. It extracts spatial features from 128×128 magnetogram arrays and outputs a single probability score via a sigmoid activation function.

## Evaluation Metrics
- **TSS (True Skill Statistic)**: Measures the ability to distinguish between flaring and non-flaring regions, independent of class imbalance.
- **HSS (Heidke Skill Score)**: Measures fractional improvement over a random forecast.
- **PR-AUC**: Precision-Recall Area Under Curve, critical for highly imbalanced datasets.

## Explainability
The system employs **Grad-CAM** (Gradient-weighted Class Activation Mapping) to generate heatmaps. This allows researchers to verify that the model is triggering on physically meaningful features (e.g., strong magnetic polarity inversion lines) rather than image artifacts.

## Deployment
- **Frontend**: Designed for seamless deployment on Vercel.
- **Backend**: Can be containerized using Docker and deployed to services like AWS ECS or Google Cloud Run.
- **Environment variables**: Refer to `.env.example` for required configuration.

## Results
*Note: The results obtained out-of-the-box using the provided `data/sample` directory are based on synthetic data and are not scientifically meaningful. To reproduce actual research results, you must train the model on real SDO/HMI data.*

## Limitations
SolarCast is a prototype utilizing image-only, static snapshots. It does not currently incorporate temporal evolution or physical SHARP parameters. For a full discussion, see [Limitations](docs/limitations.md).

## Future Work
- Integration of temporal sequence modeling (LSTM/3D-CNN).
- Inclusion of SHARP derived physical parameters.
- Multi-modal fusion with AIA extreme ultraviolet imagery.
- Multi-class classification for distinct severity levels (C vs M vs X).
- Extensive chronological operational validation.
- Implementation of model ensembles for uncertainty quantification.

## Citations
- Bobra, M.G. & Couvidat, S. (2015). "Solar Flare Prediction Using SDO/HMI Vector Magnetic Field Data." *The Astrophysical Journal*, 798(2), 135.
- Galvez, R. et al. (2019). "A Machine-Learning Data Set Prepared from the NASA SDO Mission." *The Astrophysical Journal Supplement Series*, 242(1), 7.

## License
MIT License. See [LICENSE](LICENSE) for details.
