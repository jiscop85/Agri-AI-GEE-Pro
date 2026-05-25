from __future__ import annotations

from pathlib import Path
import math
import random
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window
from rasterio.transform import from_bounds
from shapely.geometry import mapping, box
from shapely.ops import unary_union
from skimage import morphology, measure

from config import RANDOM_SEED, BUFFER_M, PATCH_SIZE, STRIDE, MIN_POSITIVE_RATIO, KEEP_NEGATIVE_PROB

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def read_boundary(boundary_path: Path, fallback_aoi=None):
    """
    Read a local vector boundary file and return:
    - GeoDataFrame in EPSG:4326
    - dissolved geometry in EPSG:4326
    """
    if boundary_path.exists():
        gdf = gpd.read_file(boundary_path)
        if gdf.empty:
            raise ValueError(f"Boundary file is empty: {boundary_path}")
        if gdf.crs is None:
            # default assumption if source lacks CRS
            gdf = gdf.set_crs("EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")
        geom = unary_union(gdf.geometry)
        return gdf, geom

    if fallback_aoi is not None:
        geom = box(*fallback_aoi)
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[geom], crs="EPSG:4326")
        return gdf, geom

    raise FileNotFoundError(
        f"Boundary file not found: {boundary_path}. "
        "Place a .geojson/.gpkg/.shp there or enable AOI fallback."
    )

def buffered_geometry(geom, buffer_m=BUFFER_M):
    """
    Buffer the geometry in meters using EPSG:3857, then convert back to EPSG:4326.
    """
    gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326")
    gdf_3857 = gdf.to_crs("EPSG:3857")
    gdf_3857["geometry"] = gdf_3857.buffer(buffer_m)
    return gdf_3857.to_crs("EPSG:4326").geometry.iloc[0]

def save_rasterized_mask(vector_geom, reference_tif, out_mask_tif, burn_value=1):
    """
    Rasterize a polygon/multipolygon to the exact grid of the reference raster.
    """
    with rasterio.open(reference_tif) as src:
        profile = src.profile.copy()
        transform = src.transform
        shape = (src.height, src.width)
        crs = src.crs

    gdf = gpd.GeoDataFrame(geometry=[vector_geom], crs="EPSG:4326").to_crs(crs)
    mask = rasterize(
        [(mapping(gdf.geometry.iloc[0]), burn_value)],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8
    )

    profile.update(count=1, dtype=rasterio.uint8, compress="deflate")
    with rasterio.open(out_mask_tif, "w", **profile) as dst:
        dst.write(mask, 1)

    return out_mask_tif

def normalize_stack(stack, n_seasons, bands_per_season):
    """
    Reflectance bands scale to [0,1] approximately.
    Vegetation indices scale from [-1, 1] to [0, 1].
    """
    x = stack.astype(np.float32).copy()
    for s in range(n_seasons):
        start = s * bands_per_season
        # Reflectance bands (first 8 bands except indices)
        reflect = slice(start, start + 8)
        idx = slice(start + 8, start + bands_per_season)
        x[reflect] = np.clip(x[reflect] / 10000.0, 0.0, 1.0)
        x[idx] = np.clip((x[idx] + 1.0) / 2.0, 0.0, 1.0)
    return x

def patch_grid(h, w, patch_size=PATCH_SIZE, stride=STRIDE):
    for row in range(0, h - patch_size + 1, stride):
        for col in range(0, w - patch_size + 1, stride):
            yield row, col

def extract_patches(image, mask, patch_size=PATCH_SIZE, stride=STRIDE):
    """
    Balanced patch extraction for segmentation.
    image: (bands, H, W)
    mask:  (H, W)
    Returns X, y, meta
    """
    X, y, meta = [], [], []
    bands, h, w = image.shape

    for row, col in patch_grid(h, w, patch_size, stride):
        img_patch = image[:, row:row+patch_size, col:col+patch_size]
        mask_patch = mask[row:row+patch_size, col:col+patch_size]

        if img_patch.shape[1] != patch_size or img_patch.shape[2] != patch_size:
            continue

        # Empty-ness check on first eight bands.
        valid_ratio = np.mean(img_patch[:8] > 0)
        if valid_ratio < 0.10:
            continue

        positive_ratio = mask_patch.mean()

        # keep patches containing target
        if positive_ratio >= MIN_POSITIVE_RATIO:
            keep = True
        else:
            keep = (np.random.rand() < KEEP_NEGATIVE_PROB)

        if not keep:
            continue

        X.append(np.transpose(img_patch, (1, 2, 0)).astype(np.float32))
        y.append(mask_patch[..., np.newaxis].astype(np.uint8))
        meta.append({"row": row, "col": col, "positive_ratio": float(positive_ratio)})

    return np.array(X), np.array(y), meta

def dice_coef_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.reshape(-1).astype(np.float32)
    y_pred_f = y_pred.reshape(-1).astype(np.float32)
    inter = np.sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou_np(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.reshape(-1).astype(np.bool_)
    y_pred_f = y_pred.reshape(-1).astype(np.bool_)
    inter = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (inter + smooth) / (union + smooth)

def boundary_f1_score(y_true, y_pred, dilation=2):
    """
    Boundary F1 using skimage boundaries and a small dilation.
    """
    true_b = measure.find_contours(y_true.astype(np.uint8), 0.5)
    pred_b = measure.find_contours(y_pred.astype(np.uint8), 0.5)
    # Fallback: approximate with morphological boundaries if contours absent
    true_edge = morphology.binary_dilation(y_true, morphology.disk(1)) ^ morphology.binary_erosion(y_true, morphology.disk(1))
    pred_edge = morphology.binary_dilation(y_pred, morphology.disk(1)) ^ morphology.binary_erosion(y_pred, morphology.disk(1))

    if true_edge.sum() == 0 and pred_edge.sum() == 0:
        return 1.0

    tp = np.logical_and(true_edge, pred_edge).sum()
    p = pred_edge.sum()
    r = true_edge.sum()

    precision = tp / (p + 1e-6)
    recall = tp / (r + 1e-6)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

def remove_small_components(mask, min_size=64):
    lbl = measure.label(mask > 0, connectivity=2)
    cleaned = morphology.remove_small_objects(lbl, min_size=min_size) > 0
    return cleaned.astype(np.uint8)
