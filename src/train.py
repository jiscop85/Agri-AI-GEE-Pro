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

TRAIN_PATH = Path("outputs/dataset/train.npz")
VAL_PATH = Path("outputs/dataset/val.npz")

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

def load_npz(path):
    data = np.load(path)
    return data["X"].astype(np.float32), data["y"].astype(np.float32)

def augment(x, y):
    # random flips
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)

    # 90-degree rotations
    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    y = tf.image.rot90(y, k)

    # mild brightness & contrast for robust generalization
    x = tf.image.random_brightness(x, max_delta=0.05)
    x = tf.image.random_contrast(x, lower=0.9, upper=1.1)

    # Gaussian noise
    noise = tf.random.normal(tf.shape(x), mean=0.0, stddev=0.02)
    x = tf.clip_by_value(x + noise, 0.0, 1.0)
    return x, y

def make_dataset(X, y, training=True):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(2048, seed=RANDOM_SEED, reshuffle_each_iteration=True)
        ds = ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

def main():
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError("Missing dataset splits. Run src/build_dataset.py first.")

    X_train, y_train = load_npz(TRAIN_PATH)
    X_val, y_val = load_npz(VAL_PATH)

    input_shape = (PATCH_SIZE, PATCH_SIZE, X_train.shape[-1])
    model = build_attention_resunet(input_shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss=bce_dice_loss,
        metrics=[
            dice_coef,
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()

    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    Path("outputs").mkdir(exist_ok=True, parents=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODEL_OUT,
            monitor="val_dice_coef",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice_coef",
            mode="max",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_dice_coef",
            mode="max",
            patience=5,
            factor=0.5,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR),
    ]

    train_ds = make_dataset(X_train, y_train, training=True)
    val_ds = make_dataset(X_val, y_val, training=False)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    model.save(MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")

    # save training curves
    hist_path = Path("outputs/history.json")
    with hist_path.open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    plt.figure(figsize=(10, 4))
    plt.plot(history.history["dice_coef"], label="train_dice")
    plt.plot(history.history["val_dice_coef"], label="val_dice")
    plt.legend()
    plt.title("Dice score")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

