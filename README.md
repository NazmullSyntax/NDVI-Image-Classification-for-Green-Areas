# NDVI Green Space CNN

A complete portfolio pipeline for analyzing green space in Dhaka from satellite imagery. It computes NDVI from Red and NIR GeoTIFF bands, classifies vegetation into four classes, extracts 64x64 patches, trains a CNN, and evaluates predictions.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --stage ndvi
python main.py --stage dataset
python main.py --stage train
python main.py --stage evaluate
```

The first stage creates a clearly labeled synthetic demo raster when `data/raw/red_band.tif` and `data/raw/nir_band.tif` are absent. Replace those files with aligned Sentinel-2 B04/B08 or Landsat red/NIR bands for real analysis. The demo is not real Dhaka data.

## Prediction

```powershell
python main.py --stage predict --input data/samples/DEMO_patch.npy
```

The input can be a 2D `.npy` NDVI patch or a grayscale image. A trained model is required.

## Project layout

- `data/raw`: original Red and NIR bands
- `data/processed`: NDVI raster and compressed patch dataset
- `data/samples`: clearly labeled demo inputs
- `notebooks`: analysis, processing, dataset, and training walkthroughs
- `src`: reusable pipeline modules
- `outputs`: figures, maps, and evaluation reports
- `models`: saved Keras model

NDVI is calculated as `(NIR - Red) / (NIR + Red)`. Classes are Very Low, Low, Moderate, and High Vegetation using thresholds 0.20, 0.40, and 0.60. Spatial ordering is used for the train/validation/test split to reduce nearby-patch leakage.
