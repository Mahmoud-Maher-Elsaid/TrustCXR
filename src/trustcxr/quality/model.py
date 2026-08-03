from __future__ import annotations

import torch
from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class EfficientNetQualityView(nn.Module):
    def __init__(self, view_classes: int = 3, pretrained: bool = True) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        feature_dim = backbone.classifier[1].in_features
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.dropout = nn.Dropout(p=0.25)
        self.view_head = nn.Linear(feature_dim, view_classes)
        self.quality_head = nn.Linear(feature_dim, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.features(inputs)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        features = self.dropout(features)
        return {
            "view_logits": self.view_head(features),
            "quality_logit": self.quality_head(features).squeeze(1),
        }
