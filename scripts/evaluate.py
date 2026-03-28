from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FabricDefectDataset
from src.data.transforms import build_eval_transform
from src.inference.predict import load_checkpoint
from src.training.engine import evaluate
from src.training.loss import build_criterion
from src.training.metrics import extract_failure_cases, save_confusion_matrix_figure, save_metrics
from src.utils.config import load_config
from src.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"], help="Split to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    split = args.split or config["evaluation"]["split"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logger("evaluate", Path(config["paths"]["logs_dir"]) / "evaluate.log")
    model, checkpoint = load_checkpoint(args.checkpoint, device=device)

    dataset = FabricDefectDataset(
        manifest_path=config["dataset"]["manifest_path"],
        split=split,
        transform=build_eval_transform(config),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["dataset"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    criterion = build_criterion(device=device, pos_weight=None)
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
        amp_enabled=config["training"]["amp"],
        threshold=checkpoint.get("threshold", config["training"]["decision_threshold"]),
    )
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["split"] = split
    metrics["best_val_metrics"] = checkpoint.get("best_val_metrics", {})
    metrics["failure_cases"] = extract_failure_cases(
        y_true=metrics["y_true"],
        y_prob=metrics["y_prob"],
        image_paths=metrics["image_paths"],
        threshold=checkpoint.get("threshold", config["training"]["decision_threshold"]),
    )

    metrics_dir = Path(config["paths"]["metrics_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    metrics_output_path = metrics_dir / f"{split}_metrics.json"
    confusion_output_path = figures_dir / f"{split}_confusion_matrix.png"

    save_metrics(metrics, metrics_output_path)
    save_confusion_matrix_figure(metrics["confusion_matrix"], confusion_output_path)

    logger.info("Saved evaluation metrics to %s", metrics_output_path)
    logger.info("Saved confusion matrix to %s", confusion_output_path)
    logger.info(
        "Evaluation complete | split=%s accuracy=%.4f f1=%.4f recall=%.4f",
        split,
        metrics["accuracy"],
        metrics["f1"],
        metrics["recall"],
    )


if __name__ == "__main__":
    main()
