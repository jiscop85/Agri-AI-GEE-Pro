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

