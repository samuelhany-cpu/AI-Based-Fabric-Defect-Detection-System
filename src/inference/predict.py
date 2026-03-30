from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from src.data.transforms import build_eval_transform
from src.models.anomaly import (
    create_patch_extractor,
    fit_patch_neighbors,
    score_patch_embeddings,
)


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[dict[str, object], dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("model_type") != "patch_anomaly":
        raise ValueError(
            "This project now expects a patch-anomaly checkpoint. "
            "Train a new checkpoint with scripts/train.py."
        )

    extractor = create_patch_extractor(
        model_name=checkpoint["model_name"],
        pretrained=False,
    )
    extractor.load_state_dict(checkpoint["extractor_state_dict"])
    extractor.to(device)
    extractor.eval()

    memory_bank = checkpoint["memory_bank"]
    if isinstance(memory_bank, torch.Tensor):
        memory_bank_tensor = memory_bank.cpu()
    else:
        memory_bank_tensor = torch.as_tensor(memory_bank, dtype=torch.float32)

    bundle = {
        "extractor": extractor,
        "neighbors": fit_patch_neighbors(
            memory_bank=memory_bank_tensor,
            n_neighbors=int(checkpoint["patch_knn_neighbors"]),
        ),
        "device": device,
    }
    return bundle, checkpoint


def score_image(
    image_path: str | Path,
    model_bundle: dict[str, object],
    config: dict,
) -> tuple[float, list[float], tuple[int, int]]:
    transform = build_eval_transform(config)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(model_bundle["device"])

    extractor = model_bundle["extractor"]
    with torch.no_grad():
        feature_maps = extractor(image_tensor)
        _, channels, height, width = feature_maps.shape
        patch_embeddings = (
            feature_maps.permute(0, 2, 3, 1).contiguous().view(1, height * width, channels)
        )
        patch_embeddings = torch.nn.functional.normalize(patch_embeddings, p=2, dim=-1)

    scores, patch_maps = score_patch_embeddings(
        patch_lists=[patch_embeddings[0].cpu()],
        neighbors=model_bundle["neighbors"],
        top_k=int(config["training"]["patch_top_k"]),
    )
    return float(scores[0]), patch_maps[0].tolist(), (height, width)


def predict_image(
    image_path: str | Path,
    model: dict[str, object],
    config: dict,
    threshold: float | None = None,
) -> dict[str, object]:
    anomaly_score, patch_score_map, feature_map_hw = score_image(
        image_path=image_path,
        model_bundle=model,
        config=config,
    )
    decision_threshold = (
        threshold if threshold is not None else config["training"]["decision_threshold"]
    )
    predicted_index = int(anomaly_score >= decision_threshold)
    class_names = ["non_defective", "defective"]

    return {
        "image_path": str(Path(image_path).resolve()),
        "predicted_label": class_names[predicted_index],
        "confidence": float(anomaly_score / max(decision_threshold, 1e-8) if predicted_index == 1 else max(0.0, 1.0 - (anomaly_score / max(decision_threshold, 1e-8)))),
        "anomaly_score": float(anomaly_score),
        "class_probabilities": {
            "non_defective": None,
            "defective": None,
        },
        "decision_threshold": float(decision_threshold),
        "patch_score_map": patch_score_map,
        "feature_map_hw": list(feature_map_hw),
    }
