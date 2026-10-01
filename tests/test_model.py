import pytest
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
    def forward(self, x):
        return x + self.conv(x)

class ShallowCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(1, 8, 3, padding=1)
        self.fc = nn.Linear(8 * 128 * 128, 1)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

class TransferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Conv2d(1, 3, 1) # mock 3 channel input
        self.resnet_mock = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 128 * 128, 1)
    def forward(self, x):
        x = self.features(x)
        x = self.resnet_mock(x)
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.fc(x))

def test_solar_cnn_forward(trained_model):
    batch = torch.randn(4, 1, 128, 128)
    out = trained_model(batch)
    assert out.shape == (4, 1)

def test_solar_cnn_single_sample(trained_model):
    single = torch.randn(1, 1, 128, 128)
    out = trained_model(single)
    assert out.shape == (1, 1)

def test_solar_cnn_output_range(trained_model):
    batch = torch.randn(10, 1, 128, 128)
    out = trained_model(batch)
    assert (out >= 0).all() and (out <= 1).all()

def test_solar_cnn_parameters(trained_model):
    params = sum(p.numel() for p in trained_model.parameters())
    assert params > 0

def test_residual_block():
    block = ResidualBlock(16)
    x = torch.randn(2, 16, 32, 32)
    out = block(x)
    assert out.shape == (2, 16, 32, 32)

def test_shallow_cnn_forward():
    model = ShallowCNN()
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1)

def test_transfer_model_forward():
    model = TransferModel()
    x = torch.randn(2, 1, 128, 128)
    out = model(x)
    assert out.shape == (2, 1)
