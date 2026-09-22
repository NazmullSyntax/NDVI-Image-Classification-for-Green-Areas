"""Train and save the NDVI CNN."""
from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
from src.model import build_cnn, INPUT_SHAPE
from src.dataset import prepare_dataset
from src.preprocessing import load_ndvi_geotiff


def _plot_history(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for metric, filename, title in (("accuracy", "training_accuracy.png", "Training vs Validation Accuracy"),
                                    ("loss", "training_loss.png", "Training vs Validation Loss")):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label="Train")
        plt.plot(history.history[f"val_{metric}"], label="Validation")
        plt.title(title); plt.xlabel("Epoch"); plt.ylabel(metric.title()); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(output_dir, filename), dpi=150); plt.close()


def train(ndvi_path=None, data_npz=None, epochs=20, batch_size=32,
          model_out="models/ndvi_green_space_cnn.keras", figures_dir="outputs/figures"):
    if data_npz and os.path.exists(data_npz):
        with np.load(data_npz) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    elif ndvi_path:
        data = prepare_dataset(load_ndvi_geotiff(ndvi_path)[0])
    else:
        raise ValueError("Provide either ndvi_path or data_npz.")
    model = build_cnn(INPUT_SHAPE)
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    history = model.fit(data["X_train"], data["y_train"], validation_data=(data["X_val"], data["y_val"]),
                        epochs=epochs, batch_size=batch_size,
                        callbacks=[__import__("tensorflow").keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
                                   __import__("tensorflow").keras.callbacks.ModelCheckpoint(model_out, monitor="val_accuracy", save_best_only=True)], verbose=1)
    model.save(model_out)
    _plot_history(history, figures_dir)
    return model, history
