from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: list[int],
    y_prob: list[float],
    threshold: float = 0.5,
) -> dict[str, object]:
    y_pred = [int(prob >= threshold) for prob in y_prob]
    metrics: dict[str, object] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=["non_defective", "defective"],
            zero_division=0,
            output_dict=True,
        ),
    }

    if len(set(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["roc_auc"] = None

    return metrics


def extract_failure_cases(
    y_true: list[int],
    y_prob: list[float],
    image_paths: list[str],
    threshold: float = 0.5,
    limit: int = 10,
) -> list[dict[str, object]]:
    mistakes: list[dict[str, object]] = []

    for truth, probability, image_path in zip(y_true, y_prob, image_paths):
        prediction = int(probability >= threshold)
        if prediction == truth:
            continue

        confidence = probability if prediction == 1 else (1.0 - probability)
        mistakes.append(
            {
                "image_path": image_path,
                "true_label": "defective" if truth == 1 else "non_defective",
                "predicted_label": "defective" if prediction == 1 else "non_defective",
                "defective_probability": float(probability),
                "confidence": float(confidence),
            }
        )

    mistakes.sort(key=lambda item: item["confidence"], reverse=True)
    return mistakes[:limit]


def save_metrics(metrics: dict[str, object], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)


def save_confusion_matrix_figure(
    matrix: list[list[int]],
    output_path: str | Path,
    class_names: tuple[str, str] = ("non_defective", "defective"),
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    figure = plt.figure(figsize=(5, 4))
    sns.heatmap(
        np.array(matrix),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_history_figure(history: dict[str, list[float]], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_f1"], label="train_f1")
    axes[1].plot(epochs, history["val_f1"], label="val_f1")
    axes[1].set_title("F1")
    axes[1].legend()

    plt.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)
