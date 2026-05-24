# config.py
from pathlib import Path

# -----------------------------
# USER SETTINGS
# -----------------------------
PROJECT_ID = "your-gcp-project-id"

# Put your real polygon here (GeoJSON, GPKG, SHP all supported by geopandas)
BOUNDARY_PATH = Path("data/field_boundary.geojson")

# Optional: if you do not have a boundary file, you can use an AOI fallback.
USE_AOI_FALLBACK = False
AOI_FALLBACK = [147.20, -35.25, 147.60, -34.85]  # [minLon, minLat, maxLon, maxLat]

# Date range should match the crop season of the region.
# For Australia/NZ, use local growing season.
START_DATE = "2024-04-01"
END_DATE = "2024-11-30"

# Earth Engine export
CLOUDY_PIXEL_PERCENTAGE = 20
BUFFER_M = 750                # buffer around field polygon in meters
EXPORT_SCALE = 10
EXPORT_CRS = "EPSG:4326"
EXPORT_FOLDER = "EarthEngine"
STACK_NAME = "agri_multiseason_stack"
CROP_PROB_NAME = "agri_crop_prob"

# Multiseason stack
N_SEASONS = 4

# Patch generation
PATCH_SIZE = 256
STRIDE = 128
MIN_POSITIVE_RATIO = 0.002     # keep patches containing at least 0.2% target pixels
KEEP_NEGATIVE_PROB = 0.20      # randomly keep some empty patches for class balance

# Training
RANDOM_SEED = 42
BATCH_SIZE = 8
EPOCHS = 60
LEARNING_RATE = 1e-4
MODEL_OUT = "outputs/best_model.keras"
LOGS_DIR = "outputs/logs"

# Prediction
PRED_THRESHOLD = 0.5
MIN_COMPONENT_SIZE = 64         # remove tiny false positives at inference
