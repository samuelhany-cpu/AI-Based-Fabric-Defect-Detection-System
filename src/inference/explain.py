from __future__ import annotations

from pathlib import Path

import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image
from torch import nn

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("Could not find a convolutional layer for Grad-CAM.")
    return last_conv


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self._save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def generate(self, image_tensor: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor).squeeze(1)
        score = logits[0]
        score.backward()

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        weighted_activations = pooled_gradients * self.activations
        heatmap = weighted_activations.sum(dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        heatmap /= heatmap.max().clamp(min=1e-8)
        return heatmap.cpu().numpy()

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    image = (image * np.array(IMAGENET_STD)) + np.array(IMAGENET_MEAN)
    return np.clip(image, 0.0, 1.0)


def overlay_heatmap_on_image(
    image_array: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    colored_heatmap = cm.jet(heatmap)[..., :3]
    overlay = (1.0 - alpha) * image_array + alpha * colored_heatmap
    overlay = np.clip(overlay, 0.0, 1.0)
    return Image.fromarray((overlay * 255).astype(np.uint8))


def generate_gradcam_visualization(
    image_path: str | Path,
    model: nn.Module,
    config: dict,
    device: torch.device,
    output_path: str | Path,
) -> Path:
    transform = build_eval_transform(config)
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    gradcam = GradCAM(model=model, target_layer=find_last_conv_layer(model))
    try:
        heatmap = gradcam.generate(image_tensor)
    finally:
        gradcam.close()

    heatmap_image = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(
        image.size, Image.BILINEAR
    )
    heatmap_array = np.asarray(heatmap_image, dtype=np.float32) / 255.0
    base_image = denormalize_image(image_tensor.squeeze(0))
    base_image = np.asarray(Image.fromarray((base_image * 255).astype(np.uint8)).resize(image.size))
    base_image = base_image.astype(np.float32) / 255.0
    overlay = overlay_heatmap_on_image(base_image, heatmap_array)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(destination)
    return destination
