from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from config import (
    PATCH_SIZE, BATCH_SIZE, EPOCHS, LEARNING_RATE,
    MODEL_OUT, LOGS_DIR, RANDOM_SEED
)
from src.model import build_attention_resunet, dice_coef, bce_dice_loss
