from __future__ import annotations

from torch import nn


def replace_classifier_head(model: nn.Module, model_name: str, out_features: int = 1) -> nn.Module:
    normalized_name = model_name.lower()

    if normalized_name.startswith("resnet"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, out_features)
        return model

    if normalized_name.startswith("efficientnet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, out_features)
        return model

    if normalized_name.startswith("mobilenet"):
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, out_features)
        return model

    raise ValueError(f"Unsupported model name: {model_name}")
