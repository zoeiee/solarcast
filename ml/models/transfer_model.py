import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class TransferSolarCNN(nn.Module):
    def __init__(self, freeze_backbone: bool = False):
        super(TransferSolarCNN, self).__init__()
        
        # Load pre-trained ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Modify the first convolutional layer to accept 1 channel instead of 3
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, old_conv.out_channels, 
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )
        
        # Initialize the new weights (e.g., by summing the weights across the RGB channels)
        with torch.no_grad():
            self.backbone.conv1.weight[:] = old_conv.weight.sum(dim=1, keepdim=True)
            
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if 'conv1' not in name and 'fc' not in name:
                    param.requires_grad = False
                    
        # Replace the final fully connected layer for binary classification
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_ftrs, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
        
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)
