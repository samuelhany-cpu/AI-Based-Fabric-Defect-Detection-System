from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import FabricDefectDataset
from src.data.transforms import build_eval_transform, build_train_transform
from src.models.factory import create_model
from src.training.engine import evaluate, train_one_epoch
from src.training.loss import build_criterion, compute_pos_weight
from src.training.metrics import save_history_figure
from src.utils.config import load_config
from src.utils.logger import setup_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 1 baseline model.")
    parser.add_argument("--config", default="src/config/config.yaml", help="Path to config YAML.")
    return parser.parse_args()


def build_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config["project"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths_cfg = config["paths"]
    logs_dir = Path(paths_cfg["logs_dir"])
    logger = setup_logger("train", logs_dir / "train.log")

    manifest_path = Path(config["dataset"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. Run scripts/create_splits.py first."
        )

    train_dataset = FabricDefectDataset(
        manifest_path=manifest_path,
        split="train",
        transform=build_train_transform(config),
    )
    val_dataset = FabricDefectDataset(
        manifest_path=manifest_path,
        split="val",
        transform=build_eval_transform(config),
    )

    batch_size = config["training"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]
    train_loader = build_dataloader(train_dataset, batch_size, num_workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, batch_size, num_workers, shuffle=False)

    training_cfg = config["training"]
    model = create_model(
        model_name=training_cfg["model_name"],
        pretrained=training_cfg["pretrained"],
        freeze_backbone=training_cfg["freeze_backbone"],
    ).to(device)

    train_manifest = pd.read_csv(manifest_path)
    train_labels = train_manifest[train_manifest["split"] == "train"]["label"].astype(int).tolist()
    criterion = build_criterion(device=device, pos_weight=compute_pos_weight(train_labels))
    optimizer = AdamW(
        params=[parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=training_cfg["scheduler_factor"],
        patience=training_cfg["scheduler_patience"],
    )

    model_dir = Path(paths_cfg["model_dir"])
    figures_dir = Path(paths_cfg["figures_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = float("-inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_checkpoint_path = model_dir / "best_model.pt"

    for epoch in range(1, training_cfg["epochs"] + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            amp_enabled=training_cfg["amp"],
            threshold=training_cfg["decision_threshold"],
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            amp_enabled=training_cfg["amp"],
            threshold=training_cfg["decision_threshold"],
        )

        scheduler.step(val_metrics["f1"])

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_f1"].append(train_metrics["f1"])
        history["val_f1"].append(val_metrics["f1"])

        logger.info(
            "Epoch %s | train_loss=%.4f train_f1=%.4f | val_loss=%.4f val_f1=%.4f",
            epoch,
            train_metrics["loss"],
            train_metrics["f1"],
            val_metrics["loss"],
            val_metrics["f1"],
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            patience_counter = 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": training_cfg["model_name"],
                "threshold": training_cfg["decision_threshold"],
                "config": config,
                "best_val_metrics": {
                    "loss": val_metrics["loss"],
                    "accuracy": val_metrics["accuracy"],
                    "precision": val_metrics["precision"],
                    "recall": val_metrics["recall"],
                    "f1": val_metrics["f1"],
                    "roc_auc": val_metrics["roc_auc"],
                    "confusion_matrix": val_metrics["confusion_matrix"],
                },
            }
            torch.save(checkpoint, best_checkpoint_path)
            logger.info("Saved new best checkpoint to %s", best_checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= training_cfg["early_stopping_patience"]:
                logger.info("Early stopping triggered at epoch %s", epoch)
                break

    history_path = model_dir / "training_history.json"
    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    save_history_figure(history, figures_dir / "training_curves.png")
    logger.info("Training finished. Best validation F1 = %.4f", best_f1)


if __name__ == "__main__":
    main()
