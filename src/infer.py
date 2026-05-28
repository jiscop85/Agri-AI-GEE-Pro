from __future__ import annotations

from pathlib import Path
import numpy as np
import tensorflow as tf
import rasterio
from rasterio.features import sieve
from skimage import morphology
from shapely.geometry import shape
from rasterio.features import shapes
import geopandas as gpd

from config import N_SEASONS, PRED_THRESHOLD, MIN_COMPONENT_SIZE
from src.utils import normalize_stack, remove_small_components

MODEL_PATH = Path("outputs/best_model.keras")
STACK_TIF = Path("outputs/agri_multiseason_stack.tif")
PRED_TIF = Path("outputs/predicted_mask.tif")
PRED_GEOJSON = Path("outputs/predicted_mask.geojson")

PATCH_SIZE = 256
STRIDE = 128

def hann_window(size):
    w = np.hanning(size)
    win = np.outer(w, w).astype(np.float32)
    win = win / np.max(win)
    return win

def predict_overlap(model, image, patch_size=PATCH_SIZE, stride=STRIDE):
    bands, h, w = image.shape
    prob = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    win = hann_window(patch_size)

    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            patch = image[:, row:row+patch_size, col:col+patch_size]
            x = np.transpose(patch, (1, 2, 0))[np.newaxis, ...]
            pred = model.predict(x, verbose=0)[0, :, :, 0].astype(np.float32)

            prob[row:row+patch_size, col:col+patch_size] += pred * win
            weight[row:row+patch_size, col:col+patch_size] += win

    prob = prob / np.maximum(weight, 1e-6)
    mask = (prob >= PRED_THRESHOLD).astype(np.uint8)
    return prob, mask

def polygonize_mask(mask, transform, crs):
    geoms = []
    for geom, val in shapes(mask.astype(np.uint8), mask=mask.astype(bool), transform=transform):
        if val == 1:
            geoms.append(shape(geom))
    if not geoms:
        return gpd.GeoDataFrame(geometry=[], crs=crs)

    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    gdf["area_px_geom"] = gdf.geometry.area
    return gdf

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing model. Run src/train.py first.")
    if not STACK_TIF.exists():
        raise FileNotFoundError("Missing exported stack. Run src/gee_export.py and download the TIF first.")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    with rasterio.open(STACK_TIF) as src:
        stack = src.read()
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    bands_per_season = stack.shape[0] // N_SEASONS
    stack = normalize_stack(stack, n_seasons=N_SEASONS, bands_per_season=bands_per_season)

    prob, mask = predict_overlap(model, stack)

    # Clean tiny components and speckle
    mask = remove_small_components(mask, min_size=MIN_COMPONENT_SIZE)

    profile.update(count=1, dtype=rasterio.uint8, compress="deflate")
    with rasterio.open(PRED_TIF, "w", **profile) as dst:
        dst.write(mask, 1)

    # Optional polygon output
    gdf = polygonize_mask(mask, transform, crs)
    if len(gdf) > 0:
        gdf = gdf.dissolve().explode(index_parts=False).reset_index(drop=True)
        gdf.to_file(PRED_GEOJSON, driver="GeoJSON")

    # save probability for inspection
    np.save("outputs/predicted_prob.npy", prob)

    print(f"Saved mask: {PRED_TIF}")
    if len(gdf) > 0:
        print(f"Saved polygons: {PRED_GEOJSON}")

if __name__ == "__main__":
    main()
