from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FabricDefectDataset
from src.data.transforms import build_eval_transform
from src.models.anomaly import (
    build_memory_bank,
    create_patch_extractor,
    extract_patch_embeddings,
    fit_patch_neighbors,
    score_patch_embeddings,
)
from src.training.metrics import (
    build_portfolio_summary,
    compute_best_f1_threshold,
    compute_classification_metrics,
    save_metrics,
)
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.runtime import configure_runtime_environment, describe_torch_runtime
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the patch-level anomaly detector from experiment 3."
    )
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to config YAML.")
    return parser.parse_args()


def build_dataloader(dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    runtime_info = configure_runtime_environment(config)
    seed_everything(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("train", Path(config["paths"]["logs_dir"]) / "train.log")
    logger.info("Runtime directories: %s", runtime_info)
    logger.info("Torch runtime: %s", describe_torch_runtime(torch))

    manifest_path = Path(config["dataset"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run scripts/create_splits.py first."
        )

    transform = build_eval_transform(config)
    training_cfg = config["training"]
    batch_size = int(training_cfg["batch_size"])
    num_workers = int(config["dataset"]["num_workers"])

    train_dataset = FabricDefectDataset(
        manifest_path=manifest_path,
        split="train",
        transform=transform,
        allowed_labels={0},
    )
    val_dataset = FabricDefectDataset(
        manifest_path=manifest_path,
        split="val",
        transform=transform,
    )

    train_loader = build_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    val_loader = build_dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    extractor = create_patch_extractor(
        model_name=training_cfg["model_name"],
        pretrained=training_cfg["pretrained"],
    ).to(device)
    extractor.eval()

    logger.info("Extracting train patch embeddings from non-defective images only.")
    train_embeddings = extract_patch_embeddings(
        dataloader=train_loader,
        model=extractor,
        device=device,
        amp_enabled=training_cfg["amp"],
    )
    logger.info("Train normal images: %s", len(train_embeddings.patch_lists))

    memory_bank = build_memory_bank(
        patch_lists=train_embeddings.patch_lists,
        max_patches=training_cfg["max_patches"],
        random_seed=config["project"]["seed"],
    )
    neighbors = fit_patch_neighbors(
        memory_bank=memory_bank,
        n_neighbors=training_cfg["patch_knn_neighbors"],
    )

    logger.info("Extracting validation patch embeddings for threshold calibration.")
    val_embeddings = extract_patch_embeddings(
        dataloader=val_loader,
        model=extractor,
        device=device,
        amp_enabled=training_cfg["amp"],
    )
    val_scores, _ = score_patch_embeddings(
        patch_lists=val_embeddings.patch_lists,
        neighbors=neighbors,
        top_k=training_cfg["patch_top_k"],
    )
    val_labels = val_embeddings.labels.tolist()

    threshold_strategy = training_cfg.get("threshold_strategy", "best_f1")
    if threshold_strategy == "best_f1":
        threshold, best_f1 = compute_best_f1_threshold(val_labels, val_scores.tolist())
    elif threshold_strategy == "normal_quantile":
        normal_scores = val_scores[val_embeddings.labels == 0]
        threshold = float(torch.quantile(torch.tensor(normal_scores), training_cfg["threshold_quantile"]).item())
        best_f1 = compute_classification_metrics(
            y_true=val_labels,
            y_prob=val_scores.tolist(),
            threshold=threshold,
        )["f1"]
    else:
        raise ValueError(f"Unsupported threshold strategy: {threshold_strategy}")

    val_metrics = compute_classification_metrics(
        y_true=val_labels,
        y_prob=val_scores.tolist(),
        threshold=threshold,
    )
    val_metrics["image_paths"] = val_embeddings.image_paths
    val_metrics["y_true"] = val_labels
    val_metrics["y_prob"] = val_scores.tolist()
    val_metrics["threshold_strategy"] = threshold_strategy
    val_metrics["threshold"] = float(threshold)
    val_metrics["best_f1_at_threshold"] = float(best_f1)

    model_dir = Path(config["paths"]["model_dir"])
    metrics_dir = Path(config["paths"]["metrics_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_type": "patch_anomaly",
        "model_name": training_cfg["model_name"],
        "extractor_state_dict": extractor.state_dict(),
        "memory_bank": memory_bank.cpu(),
        "patch_knn_neighbors": int(training_cfg["patch_knn_neighbors"]),
        "patch_top_k": int(training_cfg["patch_top_k"]),
        "threshold": float(threshold),
        "threshold_strategy": threshold_strategy,
        "feature_map_hw": list(val_embeddings.feature_map_hw),
        "config": config,
        "best_val_metrics": {
            "accuracy": val_metrics["accuracy"],
            "precision": val_metrics["precision"],
            "recall": val_metrics["recall"],
            "f1": val_metrics["f1"],
            "roc_auc": val_metrics["roc_auc"],
            "confusion_matrix": val_metrics["confusion_matrix"],
        },
    }

    checkpoint_path = model_dir / "best_model.pt"
    torch.save(checkpoint, checkpoint_path)
    save_metrics(val_metrics, metrics_dir / "val_metrics.json")
    save_metrics(
        build_portfolio_summary(
            experiment_name="Patch-level ResNet18 + kNN",
            metrics=val_metrics,
            threshold=float(threshold),
            threshold_strategy=threshold_strategy,
        ),
        metrics_dir / "val_portfolio_summary.json",
    )

    summary = {
        "train_normal_images": len(train_embeddings.patch_lists),
        "memory_bank_size": int(memory_bank.shape[0]),
        "embedding_dim": int(memory_bank.shape[1]),
        "feature_map_hw": list(val_embeddings.feature_map_hw),
        "threshold": float(threshold),
        "threshold_strategy": threshold_strategy,
        "best_val_f1": float(best_f1),
        "checkpoint": str(checkpoint_path.resolve()),
    }
    with (model_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Saved anomaly checkpoint to %s", checkpoint_path)
    logger.info(
        "Validation metrics | accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f roc_auc=%s",
        val_metrics["accuracy"],
        val_metrics["precision"],
        val_metrics["recall"],
        val_metrics["f1"],
        f"{val_metrics['roc_auc']:.4f}" if val_metrics["roc_auc"] is not None else "n/a",
    )


if __name__ == "__main__":
    main()
