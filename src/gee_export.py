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

