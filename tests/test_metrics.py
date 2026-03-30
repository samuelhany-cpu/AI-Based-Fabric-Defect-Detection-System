from src.training.metrics import (
    build_portfolio_summary,
    compute_best_f1_threshold,
    compute_classification_metrics,
)


def test_compute_classification_metrics_returns_expected_keys() -> None:
    metrics = compute_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_prob=[0.1, 0.4, 0.7, 0.9],
        threshold=0.5,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert "classification_report" in metrics


def test_compute_best_f1_threshold_returns_reasonable_cutoff() -> None:
    threshold, best_f1 = compute_best_f1_threshold(
        y_true=[0, 0, 1, 1],
        y_score=[0.1, 0.3, 0.8, 0.9],
    )

    assert 0.3 <= threshold <= 0.8
    assert best_f1 > 0.9


def test_build_portfolio_summary_uses_repo_metric_names() -> None:
    metrics = compute_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_prob=[0.1, 0.4, 0.7, 0.9],
        threshold=0.5,
    )

    summary = build_portfolio_summary(
        experiment_name="Patch-level ResNet18 + kNN",
        metrics=metrics,
        threshold=0.5,
        threshold_strategy="best_f1",
    )

    assert summary["experiment"] == "Patch-level ResNet18 + kNN"
    assert summary["f1_normal"] == 1.0
    assert summary["f1_defect"] == 1.0
    assert summary["threshold_strategy"] == "best_f1"
