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

