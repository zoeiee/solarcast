import pytest
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import os

@pytest.fixture
def sample_magnetogram():
    """Return a 128x128 numpy array with realistic magnetic field values"""
    return np.random.normal(0, 100, (128, 128))

@pytest.fixture
def sample_flare_magnetogram():
    """Return a 128x128 numpy array with strong bipolar pattern"""
    mag = np.zeros((128, 128))
    mag[40:60, 40:60] = 1000  # Strong positive
    mag[60:80, 60:80] = -1000 # Strong negative
    # Add some noise
    mag += np.random.normal(0, 50, (128, 128))
    return mag

@pytest.fixture
def sample_quiet_magnetogram():
    """Return a 128x128 numpy array with weak, smooth field"""
    return np.random.normal(0, 10, (128, 128))

@pytest.fixture
def sample_manifest_df():
    """Return a pandas DataFrame with test manifest data"""
    data = {
        'id': ['ar_1', 'ar_2', 'ar_3', 'ar_4'],
        'timestamp': ['2023-01-01T00:00:00', '2023-01-02T00:00:00', '2023-01-03T00:00:00', '2023-01-04T00:00:00'],
        'flare_class': ['M1.0', 'C1.0', 'X1.0', 'B1.0'],
        'has_flare': [1, 0, 1, 0],
        'file_path': ['ar_1.npy', 'ar_2.npy', 'ar_3.npy', 'ar_4.npy']
    }
    return pd.DataFrame(data)

@pytest.fixture
def trained_model():
    """Return a mocked SolarCNN model (random weights, eval mode)"""
    class DummySolarCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Conv2d(1, 16, 3)
            self.fc = nn.Linear(16 * 126 * 126, 1)
        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return torch.sigmoid(x)
    
    model = DummySolarCNN()
    model.eval()
    return model

@pytest.fixture
def config_dict():
    """Return a test configuration dictionary"""
    return {
        'data': {
            'image_size': 128,
            'batch_size': 32
        },
        'model': {
            'learning_rate': 0.001,
            'epochs': 10
        }
    }

@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temp directory with sample .npy files and manifest.csv"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample npy files
    for i in range(1, 5):
        np.save(data_dir / f"ar_{i}.npy", np.random.normal(0, 100, (128, 128)))
        
    # Create manifest
    df = pd.DataFrame({
        'id': [f'ar_{i}' for i in range(1, 5)],
        'file_path': [f'ar_{i}.npy' for i in range(1, 5)],
        'has_flare': [1, 0, 1, 0]
    })
    df.to_csv(data_dir / "manifest.csv", index=False)
    
    return data_dir
