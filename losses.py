import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, samples_per_class, beta=0.9999, gamma=2.0):
        super().__init__()
        counts = torch.tensor(samples_per_class, dtype=torch.float32).clamp_min(1)
        effective = 1.0 - torch.pow(torch.tensor(beta), counts)
        weights = (1.0 - beta) / effective
        weights = weights / weights.sum() * len(counts)
        self.register_buffer("weights", weights)
        self.gamma = gamma

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = targets * p + (1 - targets) * (1 - p)
        focal = (1 - pt).pow(self.gamma)
        return (focal * bce * self.weights.unsqueeze(0)).mean()
