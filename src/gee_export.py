from __future__ import annotations

import math
from pathlib import Path
import ee
import geemap.core as geemap

from config import (
    PROJECT_ID, BOUNDARY_PATH, USE_AOI_FALLBACK, AOI_FALLBACK,
    START_DATE, END_DATE, CLOUDY_PIXEL_PERCENTAGE,
    BUFFER_M, EXPORT_SCALE, EXPORT_CRS, EXPORT_FOLDER,
    STACK_NAME, CROP_PROB_NAME, N_SEASONS
)
from src.utils import read_boundary, buffered_geometry

REFLECTANCE_BANDS = ["B2", "B3", "B4", "B5", "B8", "B8A", "B11", "B12"]

def mask_s2_sr(img):
    """
    Cloud + shadow masking using SCL and QA60.
    """
    scl = img.select("SCL")
    qa60 = img.select("QA60")

    cloud_bits = qa60.bitwiseAnd(1 << 10).eq(0).And(qa60.bitwiseAnd(1 << 11).eq(0))
    scl_mask = (
        scl.neq(3)   # cloud shadow
        .And(scl.neq(8))   # medium prob cloud
        .And(scl.neq(9))   # high prob cloud
        .And(scl.neq(10))  # thin cirrus
        .And(scl.neq(11))  # snow/ice
    )
    return img.updateMask(cloud_bits.And(scl_mask))

def add_features(img):
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndre = img.normalizedDifference(["B8A", "B5"]).rename("NDRE")
    nbr = img.normalizedDifference(["B8", "B12"]).rename("NBR")
    evi = img.expression(
        "2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))",
        {
            "NIR": img.select("B8"),
            "RED": img.select("B4"),
            "BLUE": img.select("B2"),
        },
    ).rename("EVI")
    base = img.select(REFLECTANCE_BANDS)
    return base.addBands([ndvi, ndwi, ndre, nbr, evi])

def season_windows(start_date, end_date, n_seasons):
    start = ee.Date(start_date)
    end = ee.Date(end_date)
    total_days = end.difference(start, "day")
    step = total_days.divide(n_seasons)
    windows = []
    for i in range(n_seasons):
        s = start.advance(step.multiply(i), "day")
        e = start.advance(step.multiply(i + 1), "day")
        windows.append((s, e))
    return windows

def seasonal_composite(aoi, s, e):
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(s, e)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUDY_PIXEL_PERCENTAGE))
        .map(mask_s2_sr)
        .map(add_features)
    )
    return col.median().clip(aoi)

def build_stack(aoi):
    windows = season_windows(START_DATE, END_DATE, N_SEASONS)
    layers = []

    for idx, (s, e) in enumerate(windows, start=1):
        season = seasonal_composite(aoi, s, e)
        season = season.rename([
            f"B2_s{idx}", f"B3_s{idx}", f"B4_s{idx}", f"B5_s{idx}",
            f"B8_s{idx}", f"B8A_s{idx}", f"B11_s{idx}", f"B12_s{idx}",
            f"NDVI_s{idx}", f"NDWI_s{idx}", f"NDRE_s{idx}", f"NBR_s{idx}", f"EVI_s{idx}"
        ])
        layers.append(season)

    return ee.Image.cat(layers)

def main():
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)

    _, boundary_geom = read_boundary(
        Path(BOUNDARY_PATH),
        fallback_aoi=AOI_FALLBACK if USE_AOI_FALLBACK else None
    )
    aoi_geom = buffered_geometry(boundary_geom, BUFFER_M)

    ee_aoi = ee.Geometry(aoi_geom.__geo_interface__)

    stack = build_stack(ee_aoi)

    # Auxiliary cropland probability for QA or weak supervision.
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(ee_aoi)
        .filterDate(START_DATE, END_DATE)
    )
    crop_prob = dw.select("crops").median().clip(ee_aoi)

    # quick visual check
    m = geemap.Map()
    m.centerObject(ee_aoi, 14)
    m.addLayer(stack, {"bands": ["B4_s1", "B3_s1", "B2_s1"], "min": 0, "max": 3000}, "RGB season 1")
    m.addLayer(crop_prob, {"min": 0, "max": 1, "palette": ["white", "yellow", "green"]}, "Crop probability")
    display(m)

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True, parents=True)

    task_stack = ee.batch.Export.image.toDrive(
        image=stack,
        description=STACK_NAME,
        folder=EXPORT_FOLDER,
        fileNamePrefix=STACK_NAME,
        region=ee_aoi,
        scale=EXPORT_SCALE,
        crs=EXPORT_CRS,
        maxPixels=1e13
    )

    task_prob = ee.batch.Export.image.toDrive(
        image=crop_prob,
        description=CROP_PROB_NAME,
        folder=EXPORT_FOLDER,
        fileNamePrefix=CROP_PROB_NAME,
        region=ee_aoi,
        scale=EXPORT_SCALE,
        crs=EXPORT_CRS,
        maxPixels=1e13
    )

    task_stack.start()
    task_prob.start()

    print("Stack task:", task_stack.status())
    print("Crop-prob task:", task_prob.status())

if __name__ == "__main__":
    main()
