from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


def build_densenet121(
    num_labels: int,
    dropout: float,
    pretrained: bool,
) -> nn.Module:
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    input_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(input_features, num_labels),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad_(trainable)
    for parameter in model.classifier.parameters():
        parameter.requires_grad_(True)


def build_optimizer(
    model: nn.Module,
    backbone_learning_rate: float,
    classifier_learning_rate: float,
    weight_decay: float,
    full_finetune: bool,
) -> torch.optim.Optimizer:
    if full_finetune:
        parameter_groups: list[dict[str, Any]] = [
            {
                "params": [
                    parameter
                    for parameter in model.features.parameters()
                    if parameter.requires_grad
                ],
                "lr": backbone_learning_rate,
            },
            {
                "params": [
                    parameter
                    for parameter in model.classifier.parameters()
                    if parameter.requires_grad
                ],
                "lr": classifier_learning_rate,
            },
        ]
    else:
        parameter_groups = [
            {
                "params": [
                    parameter
                    for parameter in model.classifier.parameters()
                    if parameter.requires_grad
                ],
                "lr": classifier_learning_rate,
            }
        ]
    return torch.optim.AdamW(
        parameter_groups,
        weight_decay=weight_decay,
    )


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema = deepcopy(model).eval()
        for parameter in self.ema.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.decay).add_(
                    model_value,
                    alpha=1.0 - self.decay,
                )
            else:
                ema_value.copy_(model_value)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.ema.state_dict()

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> None:
        self.ema.load_state_dict(state_dict)
