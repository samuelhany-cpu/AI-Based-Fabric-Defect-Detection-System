from src.training.metrics import compute_classification_metrics


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
