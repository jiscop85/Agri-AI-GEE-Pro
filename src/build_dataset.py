from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd

from config import (
    BOUNDARY_PATH, USE_AOI_FALLBACK, AOI_FALLBACK,
    N_SEASONS, PATCH_SIZE, STRIDE, RANDOM_SEED
)
from src.utils import read_boundary, save_rasterized_mask, normalize_stack, extract_patches

STACK_TIF = Path("outputs/agri_multiseason_stack.tif")
MASK_TIF = Path("outputs/field_mask.tif")
DATASET_DIR = Path("outputs/dataset")

np.random.seed(RANDOM_SEED)

def main():
    if not STACK_TIF.exists():
        raise FileNotFoundError(
            f"Missing {STACK_TIF}. Run src/gee_export.py first and download the GeoTIFF from Drive."
        )

    # Read boundary and rasterize it exactly onto the exported chip.
    gdf, geom = read_boundary(
        Path(BOUNDARY_PATH),
        fallback_aoi=AOI_FALLBACK if USE_AOI_FALLBACK else None
    )
    save_rasterized_mask(geom, STACK_TIF, MASK_TIF)

    with rasterio.open(STACK_TIF) as src_img, rasterio.open(MASK_TIF) as src_mask:
        stack = src_img.read()
        mask = src_mask.read(1)

    bands_per_season = stack.shape[0] // N_SEASONS
    stack = normalize_stack(stack, n_seasons=N_SEASONS, bands_per_season=bands_per_season)
    mask = (mask > 0).astype(np.uint8)

    X, y, meta = extract_patches(stack, mask, patch_size=PATCH_SIZE, stride=STRIDE)

    if len(X) == 0:
        raise RuntimeError("No patches were created. Try a larger buffer or different stride.")
    # Shuffle deterministically
    idx = np.arange(len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(idx)
    X, y = X[idx], y[idx]
    meta = [meta[i] for i in idx]

    n = len(X)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test, y_test = X[n_train+n_val:], y[n_train+n_val:]

    DATASET_DIR.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(DATASET_DIR / "train.npz", X=X_train, y=y_train)
    np.savez_compressed(DATASET_DIR / "val.npz", X=X_val, y=y_val)
    np.savez_compressed(DATASET_DIR / "test.npz", X=X_test, y=y_test)

    pd.DataFrame(meta).to_csv(DATASET_DIR / "patch_meta.csv", index=False)

    print(f"Saved dataset to {DATASET_DIR}")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

if __name__ == "__main__":
    main()

