from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.transforms import build_eval_transform
from src.models.factory import create_model


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(
        model_name=checkpoint["model_name"],
        pretrained=False,
        freeze_backbone=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint


def predict_image(
    image_path: str | Path,
    model: torch.nn.Module,
    config: dict,
    device: torch.device,
    threshold: float | None = None,
) -> dict[str, object]:
    transform = build_eval_transform(config)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor).squeeze(1)
        probability = torch.sigmoid(logits).item()

    class_names = ["non_defective", "defective"]
    decision_threshold = (
        threshold if threshold is not None else config["training"]["decision_threshold"]
    )
    predicted_index = int(probability >= decision_threshold)

    return {
        "image_path": str(Path(image_path).resolve()),
        "predicted_label": class_names[predicted_index],
        "confidence": float(probability if predicted_index == 1 else 1.0 - probability),
        "class_probabilities": {
            "non_defective": float(1.0 - probability),
            "defective": float(probability),
        },
        "decision_threshold": float(decision_threshold),
    }
