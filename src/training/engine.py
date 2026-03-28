from __future__ import annotations

from typing import Any

import torch
from torch.cuda.amp import GradScaler, autocast

from src.training.metrics import compute_classification_metrics


def _run_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    amp_enabled: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    scaler = GradScaler(enabled=(amp_enabled and device.type == "cuda"))

    running_loss = 0.0
    y_true: list[int] = []
    y_prob: list[float] = []
    image_paths: list[str] = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=(amp_enabled and device.type == "cuda")):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)

        if is_training and optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        probabilities = torch.sigmoid(logits).detach().cpu().tolist()
        y_prob.extend(float(probability) for probability in probabilities)
        y_true.extend(int(value) for value in labels.detach().cpu().tolist())
        image_paths.extend(str(path) for path in batch["image_path"])
        running_loss += loss.item() * images.size(0)

    average_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(y_true=y_true, y_prob=y_prob, threshold=threshold)
    metrics["loss"] = average_loss
    metrics["image_paths"] = image_paths
    metrics["y_true"] = y_true
    metrics["y_prob"] = y_prob
    return metrics


def train_one_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
    amp_enabled: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        optimizer=optimizer,
        amp_enabled=amp_enabled,
        threshold=threshold,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    amp_enabled: bool = False,
    threshold: float = 0.5,
) -> dict[str, Any]:
    return _run_epoch(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        optimizer=None,
        amp_enabled=amp_enabled,
        threshold=threshold,
    )
