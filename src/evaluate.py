"""Evaluate a saved CNN and write metrics and plots."""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.ndvi import CLASS_NAMES


def evaluate(model_path, data_npz, figures_dir="outputs/figures", reports_dir="outputs/reports"):
    import tensorflow as tf
    os.makedirs(figures_dir, exist_ok=True); os.makedirs(reports_dir, exist_ok=True)
    with np.load(data_npz) as data:
        X_test, y_test = data["X_test"], data["y_test"]
    model = tf.keras.models.load_model(model_path)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    metrics = {"accuracy": accuracy_score(y_test, y_pred), "precision": precision_score(y_test, y_pred, average="macro", zero_division=0),
               "recall": recall_score(y_test, y_pred, average="macro", zero_division=0), "f1": f1_score(y_test, y_pred, average="macro", zero_division=0)}
    report = classification_report(y_test, y_pred, labels=list(range(4)), target_names=CLASS_NAMES, zero_division=0)
    with open(os.path.join(reports_dir, "classification_report.txt"), "w", encoding="utf-8") as output:
        output.write("\n".join(f"{key}: {value:.4f}" for key, value in metrics.items()) + "\n\n" + report)
    plt.figure(figsize=(8, 6)); sns.heatmap(confusion_matrix(y_test, y_pred, labels=list(range(4))), annot=True, fmt="d", cmap="Greens", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES); plt.tight_layout(); plt.savefig(os.path.join(figures_dir, "confusion_matrix.png"), dpi=150); plt.close()
    metrics["report"] = report
    return metrics
