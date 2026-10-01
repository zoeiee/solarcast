import pytest
import torch
import torch.nn.functional as F

def weighted_bce_loss(pred, target, pos_weight=torch.tensor(2.0)):
    return F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

def focal_loss(pred, target, gamma=2.0, alpha=0.25):
    p = torch.sigmoid(pred)
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return loss.mean()

def compute_class_weights(labels):
    pos = sum(labels)
    neg = len(labels) - pos
    return neg / pos if pos > 0 else 1.0

def test_weighted_bce_loss():
    pred = torch.randn(4, 1)
    target = torch.randint(0, 2, (4, 1)).float()
    loss = weighted_bce_loss(pred, target)
    assert loss.item() >= 0

def test_focal_loss():
    pred = torch.tensor([[10.0], [-10.0]])
    target = torch.tensor([[1.0], [0.0]])
    loss = focal_loss(pred, target)
    assert loss.item() < 0.1  # should be very small for correct predictions

def test_focal_loss_gamma_effect():
    pred = torch.tensor([[0.5]])
    target = torch.tensor([[1.0]])
    loss_g0 = focal_loss(pred, target, gamma=0.0)
    loss_g2 = focal_loss(pred, target, gamma=2.0)
    assert loss_g2.item() < loss_g0.item()

def test_class_weights_computation():
    labels = [1, 0, 0, 0] # 1 pos, 3 neg
    w = compute_class_weights(labels)
    assert w == 3.0
