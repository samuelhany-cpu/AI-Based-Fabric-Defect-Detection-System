from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from src.inference.predict import score_image


def overlay_heatmap_on_image(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    colored_heatmap = cm.jet(heatmap)[..., :3]
    overlay = (1.0 - alpha) * image_array + alpha * colored_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def generate_patch_anomaly_visualization(
    image_path: str | Path,
    model_bundle: dict[str, object],
    config: dict,
    output_path: str | Path,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    _, patch_score_map, feature_map_hw = score_image(
        image_path=image_path,
        model_bundle=model_bundle,
        config=config,
    )
    heatmap = np.asarray(patch_score_map, dtype=np.float32).reshape(feature_map_hw)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    heatmap_tensor = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    heatmap_array = torch.nn.functional.interpolate(
        heatmap_tensor,
        size=(image.height, image.width),
        mode="bilinear",
        align_corners=False,
    ).squeeze().cpu().numpy()
    base_image = np.asarray(image, dtype=np.float32) / 255.0
    overlay = overlay_heatmap_on_image(base_image, heatmap_array)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(destination)
    return destination
