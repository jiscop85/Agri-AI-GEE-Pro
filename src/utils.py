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
