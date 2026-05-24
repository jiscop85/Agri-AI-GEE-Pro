# Agri-AI GEE Pro

A high-quality, real-world crop/field segmentation pipeline built around:
- Google Earth Engine
- Sentinel-2 SR Harmonized
- Real polygon boundaries from a shapefile/GeoJSON/GPKG
- Multi-season feature stacks
- Residual Attention U-Net
- Strong segmentation losses and sliding-window inference

## What this project does

1. Reads your real field boundary polygon.
2. Builds a buffered Area of Interest.
3. Exports a multi-season Sentinel-2 chip from Earth Engine.
4. Exports a cropland probability layer from Dynamic World for QA / weak supervision.
5. Rasterizes the polygon to create a ground-truth mask.
6. Builds balanced training patches.
7. Trains an improved segmentation model.
8. Runs overlap-aware inference and evaluates the result.

## Recommended use case

Use this repository with a real boundary file:
- a single farm polygon, or
- many field polygons from a field-boundary dataset.

For a single polygon, the repo can still generate an inference chip and mask,
but training works best when you have multiple fields or many patches.

## Folder layout

- `config.py` – all parameters
- `src/gee_export.py` – Earth Engine export
- `src/build_dataset.py` – rasterize mask and build patches
- `src/model.py` – Residual Attention U-Net
- `src/train.py` – model training
- `src/infer.py` – overlap-aware prediction
- `src/evaluate.py` – IoU / Dice / Boundary F1
- `src/utils.py` – shared utilities

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Put your polygon in `data/field_boundary.geojson`
Supported formats:
- `.geojson`
- `.gpkg`
- `.shp`

### 3. Configure Earth Engine
Edit `config.py` and set:
- `PROJECT_ID`
- `BOUNDARY_PATH`
- `START_DATE`
- `END_DATE`

### 4. Export chip from Earth Engine
```bash
python src/gee_export.py
```

This creates:
- `outputs/agri_multiseason_stack.tif`
- `outputs/agri_crop_prob.tif`

### 5. Build training data
```bash
python src/build_dataset.py
```

This creates:
- `outputs/mask.tif`
- `outputs/dataset/train.npz`
- `outputs/dataset/val.npz`
- `outputs/dataset/test.npz`

### 6. Train the model
```bash
python src/train.py
```

### 7. Predict
```bash
python src/infer.py
```

### 8. Evaluate
```bash
python src/evaluate.py
```

## Notes

- `Dynamic World` is used only as a QA/auxiliary layer.
- The actual ground truth should come from your polygon boundary.
- For best results, use a dataset with multiple fields.
- Multi-season imagery is much stronger than a single median composite.

## Stronger research extensions

- add crop-type classification after segmentation
- add temporal transformer features
- add boundary-aware loss
- add post-processing with morphology and polygon smoothing
- add active learning on uncertain patches
