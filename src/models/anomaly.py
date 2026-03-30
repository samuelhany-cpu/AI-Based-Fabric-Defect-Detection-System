from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


@dataclass
class PatchEmbeddingBatch:
    patch_lists: list[torch.Tensor]
    labels: np.ndarray
    image_paths: list[str]
    feature_map_hw: tuple[int, int]


class ResNetPatchExtractor(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = self.layer1(features)
        features = self.layer2(features)
        features = self.layer3(features)
        return features


def create_patch_extractor(model_name: str, pretrained: bool = True) -> ResNetPatchExtractor:
    normalized_name = model_name.lower()
    if normalized_name != "resnet18":
        raise ValueError(
            "Patch anomaly detection currently supports only 'resnet18'."
        )

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    backbone = resnet18(weights=weights)
    return ResNetPatchExtractor(backbone)


def extract_patch_embeddings(
    dataloader,
    model: nn.Module,
    device: torch.device,
    amp_enabled: bool = False,
) -> PatchEmbeddingBatch:
    patch_lists: list[torch.Tensor] = []
    labels: list[int] = []
    image_paths: list[str] = []
    feature_map_hw: tuple[int, int] | None = None

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            with torch.autocast(
                device_type=device.type,
                enabled=(amp_enabled and device.type == "cuda"),
            ):
                feature_maps = model(images)

            batch_size, channels, height, width = feature_maps.shape
            feature_map_hw = (height, width)
            patch_embeddings = (
                feature_maps.permute(0, 2, 3, 1)
                .contiguous()
                .view(batch_size, height * width, channels)
            )
            patch_embeddings = F.normalize(patch_embeddings, p=2, dim=-1)

            for index in range(batch_size):
                patch_lists.append(patch_embeddings[index].cpu())
                labels.append(int(batch["label"][index].item()))
                image_paths.append(str(batch["image_path"][index]))

    if feature_map_hw is None:
        raise ValueError("Could not extract patch embeddings from an empty dataloader.")

    return PatchEmbeddingBatch(
        patch_lists=patch_lists,
        labels=np.asarray(labels, dtype=np.int64),
        image_paths=image_paths,
        feature_map_hw=feature_map_hw,
    )


def build_memory_bank(
    patch_lists: list[torch.Tensor],
    max_patches: int | None = None,
    random_seed: int = 42,
) -> torch.Tensor:
    if not patch_lists:
        raise ValueError("Cannot build a memory bank from an empty patch list.")

    memory_bank = torch.cat(patch_lists, dim=0).reshape(-1, patch_lists[0].shape[-1]).contiguous()

    if max_patches is not None and len(memory_bank) > max_patches:
        generator = torch.Generator()
        generator.manual_seed(random_seed)
        sample_indices = torch.randperm(len(memory_bank), generator=generator)[:max_patches]
        memory_bank = memory_bank[sample_indices]

    return memory_bank


def fit_patch_neighbors(memory_bank: torch.Tensor, n_neighbors: int) -> NearestNeighbors:
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    neighbors.fit(memory_bank.cpu().numpy())
    return neighbors


def score_patch_embeddings(
    patch_lists: list[torch.Tensor],
    neighbors: NearestNeighbors,
    top_k: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    image_scores: list[float] = []
    patch_score_maps: list[np.ndarray] = []

    for patches in patch_lists:
        distances, _ = neighbors.kneighbors(patches.numpy())
        patch_scores = distances.squeeze(axis=1) if distances.ndim == 2 and distances.shape[1] == 1 else distances.mean(axis=1)
        top_scores = np.sort(patch_scores)[-top_k:]
        image_scores.append(float(top_scores.mean()))
        patch_score_maps.append(patch_scores.astype(np.float32, copy=False))

    return np.asarray(image_scores, dtype=np.float32), patch_score_maps
