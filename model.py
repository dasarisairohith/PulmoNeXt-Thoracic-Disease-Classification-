import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from config import NUM_CLASSES

class PulmoNeXt(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.backbone = convnext_tiny(weights=weights)
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier[2] = nn.Linear(in_features, NUM_CLASSES)

    def forward(self, x):
        return self.backbone(x)
