"""Keras CNN for four NDVI vegetation classes."""
from __future__ import annotations

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except ImportError as exc:
    tf = None
    layers = models = None
    _TF_ERROR = exc

NUM_CLASSES = 4
INPUT_SHAPE = (64, 64, 1)


def build_cnn(input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    if tf is None:
        raise ImportError("TensorFlow is required for CNN training. Install dependencies from requirements.txt.") from _TF_ERROR
    inputs = layers.Input(shape=input_shape, name="ndvi_patch")
    x = inputs
    for filters, dropout in ((32, 0.25), (64, 0.25), (128, 0.30)):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(2)(x)
        x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="vegetation_class")(x)
    model = models.Model(inputs, outputs, name="NDVI_GreenSpace_CNN")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model
