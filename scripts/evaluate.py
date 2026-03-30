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
from src.models.anomaly import extract_patch_embeddings, score_patch_embeddings
from src.training.metrics import (
    build_portfolio_summary,
    compute_classification_metrics,
    extract_failure_cases,
    save_confusion_matrix_figure,
    save_metrics,
)
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.runtime import configure_runtime_environment, describe_torch_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a patch-anomaly checkpoint.")
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument("--split", default=None, choices=["train", "val", "test"], help="Split to evaluate.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime_info = configure_runtime_environment(config)
    split = args.split or config["evaluation"]["split"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logger("evaluate", Path(config["paths"]["logs_dir"]) / "evaluate.log")
    logger.info("Runtime directories: %s", runtime_info)
    logger.info("Torch runtime: %s", describe_torch_runtime(torch))
    model_bundle, checkpoint = load_checkpoint(args.checkpoint, device=device)

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

    embeddings = extract_patch_embeddings(
        dataloader=dataloader,
        model=model_bundle["extractor"],
        device=device,
        amp_enabled=config["training"]["amp"],
    )
    image_scores, _ = score_patch_embeddings(
        patch_lists=embeddings.patch_lists,
        neighbors=model_bundle["neighbors"],
        top_k=checkpoint["patch_top_k"],
    )
    threshold = float(checkpoint.get("threshold", config["training"]["decision_threshold"]))
    metrics = compute_classification_metrics(
        y_true=embeddings.labels.tolist(),
        y_prob=image_scores.tolist(),
        threshold=threshold,
    )
    metrics["checkpoint"] = str(Path(args.checkpoint).resolve())
    metrics["split"] = split
    metrics["threshold"] = threshold
    metrics["best_val_metrics"] = checkpoint.get("best_val_metrics", {})
    metrics["image_paths"] = embeddings.image_paths
    metrics["y_true"] = embeddings.labels.tolist()
    metrics["y_prob"] = image_scores.tolist()
    metrics["failure_cases"] = extract_failure_cases(
        y_true=metrics["y_true"],
        y_prob=metrics["y_prob"],
        image_paths=metrics["image_paths"],
        threshold=threshold,
    )

    metrics_dir = Path(config["paths"]["metrics_dir"])
    figures_dir = Path(config["paths"]["figures_dir"])
    metrics_output_path = metrics_dir / f"{split}_metrics.json"
    confusion_output_path = figures_dir / f"{split}_confusion_matrix.png"

    save_metrics(metrics, metrics_output_path)
    save_metrics(
        build_portfolio_summary(
            experiment_name="Patch-level ResNet18 + kNN",
            metrics=metrics,
            threshold=threshold,
            threshold_strategy=checkpoint.get("threshold_strategy", "unknown"),
        ),
        metrics_dir / f"{split}_portfolio_summary.json",
    )
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
