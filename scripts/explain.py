from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.explain import generate_patch_anomaly_visualization
from src.inference.predict import load_checkpoint
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.runtime import configure_runtime_environment, describe_torch_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a patch-anomaly heatmap visualization.")
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--image", required=True, help="Path to image file.")
    parser.add_argument("--output", default=None, help="Optional output path for the heatmap image.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime_info = configure_runtime_environment(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("explain", Path(config["paths"]["logs_dir"]) / "explain.log")
    logger.info("Runtime directories: %s", runtime_info)
    logger.info("Torch runtime: %s", describe_torch_runtime(torch))

    model_bundle, _ = load_checkpoint(args.checkpoint, device=device)
    output_path = (
        Path(args.output)
        if args.output is not None
        else Path(config["paths"]["figures_dir"]) / f"anomaly_heatmap_{Path(args.image).stem}.png"
    )
    saved_path = generate_patch_anomaly_visualization(
        image_path=args.image,
        model_bundle=model_bundle,
        config=config,
        output_path=output_path,
    )
    logger.info("Saved anomaly heatmap visualization to %s", saved_path)


if __name__ == "__main__":
    main()
