import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import Config

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight: float = 1.0):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = torch.tensor([pos_weight])
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.pos_weight.device != logits.device:
            self.pos_weight = self.pos_weight.to(logits.device)
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight)

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def compute_class_weights(labels: list) -> float:
    pos_count = sum(labels)
    neg_count = len(labels) - pos_count
    if pos_count == 0:
        return 1.0
    return float(neg_count / pos_count)

def get_loss_fn(config: Config) -> nn.Module:
    loss_type = config.get('loss_type', 'bce')
    if loss_type == 'focal':
        return FocalLoss(
            alpha=config.get('focal_alpha', 0.25),
            gamma=config.get('focal_gamma', 2.0)
        )
    else:
        return WeightedBCELoss(pos_weight=config.get('pos_weight', 1.0))
