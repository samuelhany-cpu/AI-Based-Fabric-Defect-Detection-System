from __future__ import annotations

from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)

from src.models.classifier import replace_classifier_head


def create_model(model_name: str, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    normalized_name = model_name.lower()

    if normalized_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
    elif normalized_name == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
    elif normalized_name == "mobilenet_v3_small":
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    model = replace_classifier_head(model, model_name=normalized_name, out_features=1)

    if freeze_backbone:
        for name, parameter in model.named_parameters():
            if "fc" in name or "classifier" in name:
                parameter.requires_grad = True
            else:
                parameter.requires_grad = False

    return model
