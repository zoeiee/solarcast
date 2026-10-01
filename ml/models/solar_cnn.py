import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SolarCNN(nn.Module):
    def __init__(self):
        super(SolarCNN, self).__init__()
        
        self.init_conv = nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.init_bn = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = ResidualBlock(32, 64, stride=2)
        self.layer2 = ResidualBlock(64, 128, stride=2)
        self.layer3 = ResidualBlock(128, 256, stride=2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout1 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)
        
        # Store intermediate feature maps
        self.feature_maps = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(F.relu(self.init_bn(self.init_conv(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        self.feature_maps = x
        
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x
        
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)
        
    def get_feature_maps(self) -> torch.Tensor:
        return self.feature_maps
