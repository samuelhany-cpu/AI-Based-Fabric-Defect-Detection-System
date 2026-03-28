from __future__ import annotations

import torch
from torch import nn


def compute_pos_weight(labels: list[int]) -> float | None:
    positive_count = sum(labels)
    negative_count = len(labels) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    return negative_count / positive_count


def build_criterion(device: torch.device, pos_weight: float | None = None) -> nn.Module:
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
