from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.split import (
    create_split_manifest,
    scan_aitex_archive3,
    scan_binary_image_folder,
    summarize_manifest,
)
from src.utils.config import load_config, write_json
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a leakage-aware split manifest.")
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to config YAML.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["project"]["seed"])

    logs_dir = Path(config["paths"]["logs_dir"])
    logger = setup_logger("create_splits", logs_dir / "create_splits.log")
    dataset_cfg = config["dataset"]

    dataset_format = dataset_cfg.get("format", "binary_image_folder")
    if dataset_format == "aitex_archive3":
        manifest = scan_aitex_archive3(dataset_cfg["root_dir"])
    elif dataset_format == "binary_image_folder":
        manifest = scan_binary_image_folder(
            root_dir=dataset_cfg["root_dir"],
            class_to_idx=dataset_cfg["class_names"],
            extensions=dataset_cfg["extensions"],
            group_depth=dataset_cfg["group_depth"],
        )
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_format}")

    split_manifest = create_split_manifest(
        manifest=manifest,
        train_ratio=dataset_cfg["train_ratio"],
        val_ratio=dataset_cfg["val_ratio"],
        test_ratio=dataset_cfg["test_ratio"],
        random_seed=config["project"]["seed"],
    )
    summary = summarize_manifest(split_manifest)

    manifest_path = Path(dataset_cfg["manifest_path"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    split_manifest.to_csv(manifest_path, index=False)
    write_json(summary, dataset_cfg["split_summary_path"])

    logger.info("Saved manifest to %s", manifest_path)
    logger.info("Split summary: %s", summary)


if __name__ == "__main__":
    main()
