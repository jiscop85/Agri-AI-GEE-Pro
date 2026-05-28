from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, Model

def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    inter = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )

def tversky_index(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(tf.cast(y_pred, tf.float32))
    tp = tf.keras.backend.sum(y_true_f * y_pred_f)
    fp = tf.keras.backend.sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.keras.backend.sum(y_true_f * (1.0 - y_pred_f))
    return (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)

def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    ti = tversky_index(y_true, y_pred, alpha=alpha, beta=beta)
    return tf.pow((1.0 - ti), gamma)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return bce + focal_tversky_loss(y_true, y_pred)

def conv_bn_relu(x, filters, kernel=3, dropout=0.0):
    x = layers.Conv2D(filters, kernel, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, kernel, padding="same", kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    return x
def residual_block(x, filters, dropout=0.0):
    shortcut = x
    x = conv_bn_relu(x, filters, dropout=dropout)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x

def attention_gate(skip, gating, inter_filters):
    theta = layers.Conv2D(inter_filters, 1, padding="same")(skip)
    phi = layers.Conv2D(inter_filters, 1, padding="same")(gating)
    add = layers.Add()([theta, phi])
    act = layers.Activation("relu")(add)
    psi = layers.Conv2D(1, 1, padding="same")(act)
    psi = layers.Activation("sigmoid")(psi)
    return layers.Multiply()([skip, psi])

def build_attention_resunet(input_shape):
    inputs = layers.Input(shape=input_shape)

    c1 = residual_block(inputs, 32)
    p1 = layers.MaxPooling2D()(c1)

    c2 = residual_block(p1, 64)
    p2 = layers.MaxPooling2D()(c2)

    c3 = residual_block(p2, 128)
    p3 = layers.MaxPooling2D()(c3)

    c4 = residual_block(p3, 256, dropout=0.15)
    p4 = layers.MaxPooling2D()(c4)

    bn = residual_block(p4, 512, dropout=0.25)

    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(bn)
    a4 = attention_gate(c4, u4, 128)
    u4 = layers.Concatenate()([u4, a4])
    c5 = residual_block(u4, 256)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
    a3 = attention_gate(c3, u3, 64)
    u3 = layers.Concatenate()([u3, a3])
    c6 = residual_block(u3, 128)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
    a2 = attention_gate(c2, u2, 32)
    u2 = layers.Concatenate()([u2, a2])
    c7 = residual_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
    a1 = attention_gate(c1, u1, 16)
    u1 = layers.Concatenate()([u1, a1])
    c8 = residual_block(u1, 32)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c8)

    return Model(inputs, outputs, name="AttentionResUNet")

