from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio

from src.utils import dice_coef_np, iou_np, boundary_f1_score

TEST_PATH = Path("outputs/dataset/test.npz")
PRED_TIF = Path("outputs/predicted_mask.tif")
MASK_TIF = Path("outputs/field_mask.tif")

def main():
    if not TEST_PATH.exists():
        raise FileNotFoundError("Missing test split. Run src/build_dataset.py first.")
    if not PRED_TIF.exists():
        raise FileNotFoundError("Missing prediction. Run src/infer.py first.")
    if not MASK_TIF.exists():
        raise FileNotFoundError("Missing rasterized ground truth mask.")

    test = np.load(TEST_PATH)
    y_true = test["y"].astype(np.uint8)

    # If a full-scene prediction is available, compare at raster level.
    with rasterio.open(PRED_TIF) as pred_src, rasterio.open(MASK_TIF) as mask_src:
        pred = pred_src.read(1).astype(np.uint8)
        truth = mask_src.read(1).astype(np.uint8)

    # Full-scene scores
    iou = iou_np(truth, pred)
    dice = dice_coef_np(truth, pred)
    bf1 = boundary_f1_score(truth, pred)

    print("Full-scene IoU:", round(float(iou), 4))
    print("Full-scene Dice:", round(float(dice), 4))
    print("Boundary F1:", round(float(bf1), 4))

if __name__ == "__main__":
    main()
