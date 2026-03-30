from pathlib import Path

import torch

from src.data.dataset import FabricDefectDataset
from src.models.anomaly import build_memory_bank, score_patch_embeddings


def test_dataset_can_filter_to_normal_samples(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "\n".join(
            [
                "image_path,label,label_name,group_id,split",
                f"{(image_dir / 'a.png').as_posix()},0,non_defective,g1,train",
                f"{(image_dir / 'b.png').as_posix()},1,defective,g2,train",
            ]
        ),
        encoding="utf-8",
    )

    # The dataset does not open images until __getitem__, so placeholder paths are enough here.
    dataset = FabricDefectDataset(manifest_path=manifest, split="train", allowed_labels={0})

    assert len(dataset) == 1
    assert int(dataset.samples.iloc[0]["label"]) == 0


def test_patch_scoring_marks_farther_embeddings_as_more_anomalous() -> None:
    memory_bank = build_memory_bank(
        [
            torch.tensor([[1.0, 0.0], [0.9, 0.1]], dtype=torch.float32),
            torch.tensor([[0.95, 0.05]], dtype=torch.float32),
        ]
    )
    from src.models.anomaly import fit_patch_neighbors

    neighbors = fit_patch_neighbors(memory_bank=memory_bank, n_neighbors=1)
    scores, _ = score_patch_embeddings(
        patch_lists=[
            torch.tensor([[1.0, 0.0], [0.98, 0.02]], dtype=torch.float32),
            torch.tensor([[0.0, 1.0], [0.1, 0.9]], dtype=torch.float32),
        ],
        neighbors=neighbors,
        top_k=1,
    )

    assert scores[1] > scores[0]
