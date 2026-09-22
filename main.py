"""Command-line pipeline for the NDVI Green Space CNN project."""
from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.ndvi import compute_ndvi, read_bands, classify_ndvi, ndvi_statistics, class_percentages, CLASS_NAMES
from src.preprocessing import clean_ndvi, load_ndvi_geotiff, save_geotiff
from src.dataset import prepare_dataset

RED_BAND_PATH = "data/raw/red_band.tif"
NIR_BAND_PATH = "data/raw/nir_band.tif"
NDVI_OUT_PATH = "data/processed/ndvi.tif"
PATCHES_NPZ = "data/processed/ndvi_patches.npz"
MODEL_PATH = "models/ndvi_green_space_cnn.keras"


def _create_demo_ndvi():
    import rasterio
    from rasterio.transform import from_origin
    os.makedirs(os.path.dirname(NDVI_OUT_PATH), exist_ok=True)
    rng = np.random.default_rng(42)
    size = 512
    rows, cols = np.mgrid[0:size, 0:size]
    zones = np.select([cols < 128, cols < 256, cols < 384], [-0.15, 0.28, 0.52], default=0.75)
    ndvi = np.clip(zones + 0.10 * np.sin(rows / 18) + 0.08 * np.cos(cols / 23) + rng.normal(0, 0.035, (size, size)), -1, 1).astype(np.float32)
    profile = {"driver": "GTiff", "height": size, "width": size, "count": 1, "dtype": "float32", "crs": "EPSG:32646", "transform": from_origin(0, 0, 10, 10), "nodata": np.nan, "compress": "lzw"}
    save_geotiff(ndvi, profile, NDVI_OUT_PATH)
    print(f"DEMO NDVI raster saved to {NDVI_OUT_PATH}")


def stage_ndvi():
    print(">>> Stage: NDVI calculation")
    os.makedirs("outputs/figures", exist_ok=True); os.makedirs("outputs/maps", exist_ok=True)
    if not (os.path.exists(RED_BAND_PATH) and os.path.exists(NIR_BAND_PATH)):
        print("Real Red/NIR bands not found; generating clearly labeled synthetic demo data.")
        _create_demo_ndvi()
    else:
        red, nir, profile, _, _ = read_bands(RED_BAND_PATH, NIR_BAND_PATH)
        save_geotiff(clean_ndvi(compute_ndvi(red, nir)), profile, NDVI_OUT_PATH)
    ndvi, _, _, _ = load_ndvi_geotiff(NDVI_OUT_PATH)
    print("NDVI statistics:", ndvi_statistics(ndvi))
    valid = ndvi[np.isfinite(ndvi)]
    plt.figure(figsize=(9, 5)); sns.histplot(valid, bins=60, kde=True, color="green"); plt.title("NDVI Distribution"); plt.xlabel("NDVI"); plt.tight_layout(); plt.savefig("outputs/figures/ndvi_histogram.png", dpi=150); plt.close()
    classes = classify_ndvi(ndvi); percentages = class_percentages(classes)
    plt.figure(figsize=(9, 5)); sns.barplot(x=list(percentages), y=list(percentages.values()), color="forestgreen"); plt.title("Vegetation Class Distribution"); plt.xlabel("Class"); plt.ylabel("Percentage (%)"); plt.xticks(rotation=15); plt.tight_layout(); plt.savefig("outputs/figures/class_distribution.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 8)); plt.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8); plt.colorbar(label="NDVI"); plt.title("NDVI Map"); plt.axis("off"); plt.tight_layout(); plt.savefig("outputs/maps/ndvi_map.png", dpi=150); plt.close()
    plt.figure(figsize=(8, 8)); plt.imshow(np.where(classes < 0, np.nan, classes), cmap="Greens", vmin=0, vmax=3); colorbar = plt.colorbar(ticks=range(4)); colorbar.ax.set_yticklabels(CLASS_NAMES); plt.title("Vegetation Classification Map"); plt.axis("off"); plt.tight_layout(); plt.savefig("outputs/maps/classification_map.png", dpi=150); plt.close()


def stage_dataset():
    if not os.path.exists(NDVI_OUT_PATH): stage_ndvi()
    data = prepare_dataset(load_ndvi_geotiff(NDVI_OUT_PATH)[0], out_dir="data/processed")
    for split in ("train", "val", "test"): print(f"{split}: {data[f'X_{split}'].shape}")


def stage_train():
    from src.train import train
    train(data_npz=PATCHES_NPZ, model_out=MODEL_PATH)


def stage_evaluate():
    from src.evaluate import evaluate
    results = evaluate(MODEL_PATH, PATCHES_NPZ)
    print("\n".join(f"{key}: {value:.4f}" for key, value in results.items() if isinstance(value, float)))


def stage_predict(input_path):
    from src.predict import predict_many, pretty_print
    for result in predict_many(MODEL_PATH, [input_path]): pretty_print(result)


def main():
    parser = argparse.ArgumentParser(description="NDVI Green Space CNN - Dhaka City")
    parser.add_argument("--stage", required=True, choices=["ndvi", "dataset", "train", "evaluate", "predict"])
    parser.add_argument("--input", help=".npy or image patch for prediction")
    args = parser.parse_args()
    if args.stage == "ndvi": stage_ndvi()
    elif args.stage == "dataset": stage_dataset()
    elif args.stage == "train": stage_train()
    elif args.stage == "evaluate": stage_evaluate()
    elif args.stage == "predict":
        if not args.input: parser.error("--input is required for predict")
        stage_predict(args.input)


if __name__ == "__main__": main()
