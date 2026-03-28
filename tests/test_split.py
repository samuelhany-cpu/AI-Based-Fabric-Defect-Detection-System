from pathlib import Path

import pandas as pd

from src.data.split import (
    create_split_manifest,
    infer_group_id,
    scan_aitex_archive3,
    scan_binary_image_folder,
)


def test_infer_group_id_uses_parent_folder() -> None:
    relative_path = Path("fabric_01/sample_001.png")
    assert infer_group_id(relative_path, group_depth=1) == "fabric_01"


def test_create_split_manifest_keeps_groups_together() -> None:
    manifest = pd.DataFrame(
        [
            {"image_path": "a1.png", "label": 0, "label_name": "non_defective", "group_id": "g1"},
            {"image_path": "a2.png", "label": 0, "label_name": "non_defective", "group_id": "g1"},
            {"image_path": "b1.png", "label": 1, "label_name": "defective", "group_id": "g2"},
            {"image_path": "b2.png", "label": 1, "label_name": "defective", "group_id": "g2"},
            {"image_path": "c1.png", "label": 0, "label_name": "non_defective", "group_id": "g3"},
            {"image_path": "c2.png", "label": 0, "label_name": "non_defective", "group_id": "g3"},
            {"image_path": "d1.png", "label": 1, "label_name": "defective", "group_id": "g4"},
            {"image_path": "d2.png", "label": 1, "label_name": "defective", "group_id": "g4"},
            {"image_path": "e1.png", "label": 0, "label_name": "non_defective", "group_id": "g5"},
            {"image_path": "f1.png", "label": 1, "label_name": "defective", "group_id": "g6"},
        ]
    )

    split_manifest = create_split_manifest(
        manifest=manifest,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
        random_seed=42,
    )

    grouped_splits = split_manifest.groupby("group_id")["split"].nunique()
    assert grouped_splits.max() == 1


def test_scan_binary_image_folder_discovers_images(tmp_path: Path) -> None:
    defective_dir = tmp_path / "defective" / "fabric_01"
    normal_dir = tmp_path / "non_defective" / "fabric_02"
    defective_dir.mkdir(parents=True)
    normal_dir.mkdir(parents=True)

    (defective_dir / "sample_001.jpg").write_bytes(b"fake")
    (normal_dir / "sample_002.png").write_bytes(b"fake")

    manifest = scan_binary_image_folder(
        root_dir=tmp_path,
        class_to_idx={"non_defective": 0, "defective": 1},
        extensions=[".jpg", ".png"],
        group_depth=1,
    )

    assert len(manifest) == 2
    assert set(manifest["label_name"]) == {"defective", "non_defective"}


def test_scan_aitex_archive3_builds_expected_groups(tmp_path: Path) -> None:
    defect_dir = tmp_path / "Defect_images"
    mask_dir = tmp_path / "Mask_images"
    normal_group = tmp_path / "NODefect_images" / "group_a"
    defect_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)
    normal_group.mkdir(parents=True)

    (defect_dir / "0001_019_02.png").write_bytes(b"fake")
    (mask_dir / "0001_019_02_mask.png").write_bytes(b"fake")
    (normal_group / "0001_000_05.png").write_bytes(b"fake")

    manifest = scan_aitex_archive3(tmp_path)

    assert len(manifest) == 2
    defect_row = manifest[manifest["label"] == 1].iloc[0]
    normal_row = manifest[manifest["label"] == 0].iloc[0]
    assert defect_row["group_id"] == "defect_019_02"
    assert "0001_019_02_mask.png" in defect_row["mask_paths"]
    assert normal_row["group_id"] == "normal_group_a"
