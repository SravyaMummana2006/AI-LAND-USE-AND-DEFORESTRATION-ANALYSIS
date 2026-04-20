"""
train_model.py
--------------
TensorFlow CNN Model Training Module for Land Use Classification.

Responsibilities:
- Define a production-grade CNN architecture (custom + optional MobileNetV2 transfer learning)
- Build data ingestion pipeline with augmentation
- Compile with appropriate loss and metrics
- Train with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
- Save trained model and class label mapping
- Plot and export training history

Land Use Classes:
    0 → Forest
    1 → Water
    2 → Urban
    3 → Agriculture
    4 → Barren

Author: AI Land Use Analysis System
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants & Configuration
# ─────────────────────────────────────────────
IMG_SIZE        = (256, 256)
IMG_CHANNELS    = 3
INPUT_SHAPE     = (*IMG_SIZE, IMG_CHANNELS)
NUM_CLASSES     = 5
BATCH_SIZE      = 32
EPOCHS          = 50
LEARNING_RATE   = 1e-3
DROPOUT_RATE    = 0.4
L2_REG          = 1e-4
VALIDATION_SPLIT = 0.2

# Output paths
MODEL_SAVE_PATH     = "saved_model/land_use_model.h5"
LABELS_SAVE_PATH    = "saved_model/class_labels.json"
HISTORY_SAVE_PATH   = "saved_model/training_history.csv"
HISTORY_PLOT_PATH   = "saved_model/training_history.png"

# Class label mapping
CLASS_LABELS = {
    0: "Forest",
    1: "Water",
    2: "Urban",
    3: "Agriculture",
    4: "Barren"
}

# Class weights to handle imbalanced datasets (adjust based on your data)
CLASS_WEIGHTS = {
    0: 1.0,   # Forest
    1: 1.2,   # Water (slightly rarer)
    2: 1.1,   # Urban
    3: 1.0,   # Agriculture
    4: 1.3    # Barren (often underrepresented)
}


# ─────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────

def build_custom_cnn(input_shape: tuple = INPUT_SHAPE,
                     num_classes: int = NUM_CLASSES) -> tf.keras.Model:
    """
    Build a custom deep CNN optimized for satellite image classification.

    Architecture:
        3 × ConvBlock (Conv2D → BatchNorm → ReLU → MaxPool → Dropout)
        → Global Average Pooling
        → Dense(256) → BatchNorm → Dropout
        → Dense(num_classes, softmax)

    Uses L2 regularization to prevent overfitting on small satellite datasets.

    Parameters:
        input_shape (tuple): (H, W, C) input dimensions.
        num_classes (int): Number of land-use categories.

    Returns:
        tf.keras.Model: Compiled-ready CNN model.
    """
    inputs = layers.Input(shape=input_shape, name="satellite_input")

    # ── Block 1: Low-level edge & texture features ──
    x = layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG),
                      name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.Activation('relu', name='relu1_1')(x)
    x = layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG),
                      name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.Activation('relu', name='relu1_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool1')(x)
    x = layers.Dropout(0.2, name='drop1')(x)

    # ── Block 2: Mid-level shape & color features ──
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG),
                      name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.Activation('relu', name='relu2_1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG),
                      name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.Activation('relu', name='relu2_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool2')(x)
    x = layers.Dropout(0.3, name='drop2')(x)

    # ── Block 3: High-level land pattern features ──
    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG),
                      name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.Activation('relu', name='relu3_1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG),
                      name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    x = layers.Activation('relu', name='relu3_2')(x)
    x = layers.MaxPooling2D((2, 2), name='pool3')(x)
    x = layers.Dropout(0.3, name='drop3')(x)

    # ── Block 4: Deep semantic features ──
    x = layers.Conv2D(256, (3, 3), padding='same',
                      kernel_regularizer=regularizers.l2(L2_REG),
                      name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.Activation('relu', name='relu4_1')(x)
    x = layers.MaxPooling2D((2, 2), name='pool4')(x)
    x = layers.Dropout(DROPOUT_RATE, name='drop4')(x)

    # ── Global Average Pooling → reduces spatial dims, retains feature richness ──
    x = layers.GlobalAveragePooling2D(name='gap')(x)

    # ── Fully Connected Classifier Head ──
    x = layers.Dense(256,
                     kernel_regularizer=regularizers.l2(L2_REG),
                     name='fc1')(x)
    x = layers.BatchNormalization(name='bn_fc1')(x)
    x = layers.Activation('relu', name='relu_fc1')(x)
    x = layers.Dropout(DROPOUT_RATE, name='drop_fc1')(x)

    x = layers.Dense(128,
                     kernel_regularizer=regularizers.l2(L2_REG),
                     name='fc2')(x)
    x = layers.Activation('relu', name='relu_fc2')(x)
    x = layers.Dropout(0.3, name='drop_fc2')(x)

    # ── Output Layer ──
    outputs = layers.Dense(num_classes,
                           activation='softmax',
                           name='predictions')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="LandUseCNN")
    logger.info(f"Custom CNN built | Params: {model.count_params():,}")
    return model


def build_transfer_model(input_shape: tuple = INPUT_SHAPE,
                         num_classes: int = NUM_CLASSES,
                         fine_tune_at: int = 100) -> tf.keras.Model:
    """
    Build a transfer learning model using MobileNetV2 pretrained on ImageNet.
    Ideal when training data is limited (< 5,000 images per class).

    Strategy:
        - Freeze base layers initially
        - Unfreeze layers after `fine_tune_at` index for fine-tuning
        - Attach custom classification head

    Parameters:
        input_shape (tuple): (H, W, C) input dimensions.
        num_classes (int): Number of land-use categories.
        fine_tune_at (int): Layer index from which to unfreeze for fine-tuning.

    Returns:
        tf.keras.Model: Transfer learning model ready for compilation.
    """
    # Load MobileNetV2 base (no top classification layer)
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model initially
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    logger.info(
        f"MobileNetV2 base | "
        f"Total layers: {len(base_model.layers)} | "
        f"Trainable from layer: {fine_tune_at}"
    )

    # Build model on top of base
    inputs = layers.Input(shape=input_shape, name="satellite_input")

    # MobileNetV2 expects inputs preprocessed to [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)

    # Classification head
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG),
                     name='fc1')(x)
    x = layers.BatchNormalization(name='bn_fc')(x)
    x = layers.Dropout(DROPOUT_RATE, name='dropout')(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           name='predictions')(x)

    model = models.Model(inputs=inputs, outputs=outputs,
                         name="LandUseTransferModel")
    logger.info(f"Transfer model built | Trainable params: "
                f"{sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    return model


# ─────────────────────────────────────────────
# Data Pipeline
# ─────────────────────────────────────────────

def build_data_generators(data_dir: str,
                           batch_size: int = BATCH_SIZE,
                           validation_split: float = VALIDATION_SPLIT):
    """
    Build training and validation data generators with augmentation.

    Expected directory structure:
        data_dir/
        ├── Forest/
        ├── Water/
        ├── Urban/
        ├── Agriculture/
        └── Barren/

    Augmentation applied (training only):
        - Horizontal & vertical flips (satellite images have no canonical orientation)
        - Rotation up to 30°
        - Width & height shifts up to 10%
        - Zoom up to 15%
        - Brightness adjustment (simulates different lighting/seasons)
        - Fill mode: reflect (avoids black border artifacts)

    Parameters:
        data_dir (str): Root directory with class-named subdirectories.
        batch_size (int): Number of images per batch.
        validation_split (float): Fraction of data for validation.

    Returns:
        tuple: (train_generator, val_generator, class_indices dict)
    """
    # Training generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect',
        validation_split=validation_split
    )

    # Validation generator — only rescaling, no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=validation_split
    )

    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )

    logger.info(f"Training samples   : {train_generator.n}")
    logger.info(f"Validation samples : {val_generator.n}")
    logger.info(f"Class indices      : {train_generator.class_indices}")

    return train_generator, val_generator, train_generator.class_indices


def build_callbacks(model_save_path: str = MODEL_SAVE_PATH) -> list:
    """
    Build a set of training callbacks for robust model training:

        1. ModelCheckpoint  — saves best model based on val_accuracy
        2. EarlyStopping    — halts training if val_loss stops improving (patience=10)
        3. ReduceLROnPlateau — halves LR when val_loss plateaus (patience=5)
        4. TensorBoard      — optional: logs metrics for visualization

    Parameters:
        model_save_path (str): File path to save the best model weights.

    Returns:
        list: List of tf.keras.callbacks objects.
    """
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

    checkpoint = callbacks.ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=False,
        mode='max',
        verbose=1
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )

    csv_logger = callbacks.CSVLogger(
        filename=HISTORY_SAVE_PATH,
        append=False
    )

    logger.info("Training callbacks configured.")
    return [checkpoint, early_stop, reduce_lr, csv_logger]


# ─────────────────────────────────────────────
# Training Orchestration
# ─────────────────────────────────────────────

def compile_model(model: tf.keras.Model,
                  learning_rate: float = LEARNING_RATE) -> tf.keras.Model:
    """
    Compile the model with Adam optimizer and categorical cross-entropy loss.

    Metrics tracked:
        - accuracy
        - AUC (Area Under ROC Curve)
        - Precision & Recall (useful for imbalanced land-use datasets)

    Parameters:
        model (tf.keras.Model): Uncompiled model.
        learning_rate (float): Initial learning rate.

    Returns:
        tf.keras.Model: Compiled model.
    """
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )
    logger.info(f"Model compiled | LR: {learning_rate} | Loss: categorical_crossentropy")
    return model


def train_model(data_dir: str,
                use_transfer_learning: bool = False,
                epochs: int = EPOCHS,
                batch_size: int = BATCH_SIZE) -> tuple:
    """
    Full training pipeline:
        1. Build data generators
        2. Build model (custom CNN or transfer)
        3. Compile
        4. Train with callbacks
        5. Save model and class labels

    Parameters:
        data_dir (str): Directory with class-named subdirectories.
        use_transfer_learning (bool): If True, uses MobileNetV2 backbone.
        epochs (int): Maximum training epochs.
        batch_size (int): Training batch size.

    Returns:
        tuple: (trained_model, history_object)
    """
    logger.info("=" * 60)
    logger.info("STARTING LAND USE MODEL TRAINING")
    logger.info(f"Data dir              : {data_dir}")
    logger.info(f"Transfer learning     : {use_transfer_learning}")
    logger.info(f"Max epochs            : {epochs}")
    logger.info(f"Batch size            : {batch_size}")
    logger.info("=" * 60)

    # Step 1: Data generators
    train_gen, val_gen, class_indices = build_data_generators(
        data_dir, batch_size=batch_size
    )

    # Step 2: Build model
    if use_transfer_learning:
        model = build_transfer_model()
    else:
        model = build_custom_cnn()

    model.summary(print_fn=logger.info)

    # Step 3: Compile
    model = compile_model(model, learning_rate=LEARNING_RATE)

    # Step 4: Train
    training_callbacks = build_callbacks(MODEL_SAVE_PATH)

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        class_weight=CLASS_WEIGHTS,
        callbacks=training_callbacks,
        verbose=1
    )

    # Step 5: Save class label mapping
    save_class_labels(class_indices)

    # Step 6: Plot and save training history
    plot_training_history(history)

    logger.info(f"Training complete. Model saved to: {MODEL_SAVE_PATH}")
    return model, history


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def save_class_labels(class_indices: dict,
                       save_path: str = LABELS_SAVE_PATH) -> None:
    """
    Save the class index → label mapping to JSON for use during inference.

    Parameters:
        class_indices (dict): e.g. {'Agriculture': 0, 'Barren': 1, ...}
        save_path (str): Output JSON path.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Invert: index → class name
    inverted = {str(v): k for k, v in class_indices.items()}

    with open(save_path, 'w') as f:
        json.dump(inverted, f, indent=2)

    logger.info(f"Class labels saved to: {save_path}")
    logger.info(f"Mapping: {inverted}")


def load_class_labels(save_path: str = LABELS_SAVE_PATH) -> dict:
    """
    Load the class index → label mapping from JSON.

    Parameters:
        save_path (str): Path to saved JSON label file.

    Returns:
        dict: {index_str: class_name} mapping.
    """
    if not Path(save_path).exists():
        logger.warning(
            f"Class labels file not found at {save_path}. "
            f"Using default CLASS_LABELS."
        )
        return {str(k): v for k, v in CLASS_LABELS.items()}

    with open(save_path, 'r') as f:
        labels = json.load(f)

    logger.info(f"Class labels loaded: {labels}")
    return labels


def plot_training_history(history,
                           save_path: str = HISTORY_PLOT_PATH) -> None:
    """
    Plot and save training vs. validation accuracy and loss curves.

    Parameters:
        history: Keras History object from model.fit().
        save_path (str): Path to save the PNG plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Land Use CNN — Training History", fontsize=14, fontweight='bold')

    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', color='steelblue')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy', color='darkorange')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', color='steelblue')
    axes[1].plot(history.history['val_loss'], label='Val Loss', color='darkorange')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Training history plot saved to: {save_path}")


def evaluate_model(model: tf.keras.Model,
                   val_generator) -> dict:
    """
    Evaluate the trained model on the validation set and return metrics.

    Parameters:
        model (tf.keras.Model): Trained model.
        val_generator: Keras validation data generator.

    Returns:
        dict: Evaluation metrics (loss, accuracy, auc, precision, recall).
    """
    logger.info("Evaluating model on validation set...")
    results = model.evaluate(val_generator, verbose=1)
    metric_names = model.metrics_names
    metrics = dict(zip(metric_names, results))

    for name, value in metrics.items():
        logger.info(f"  {name:12s}: {value:.4f}")

    return metrics


def get_model_summary_dict(model: tf.keras.Model) -> dict:
    """
    Return a dictionary summarizing model architecture details.

    Parameters:
        model (tf.keras.Model): Any Keras model.

    Returns:
        dict: Summary with layer count, parameter counts, input/output shapes.
    """
    return {
        "model_name"        : model.name,
        "total_params"      : model.count_params(),
        "trainable_params"  : sum([tf.size(w).numpy() for w in model.trainable_weights]),
        "non_trainable_params": sum([tf.size(w).numpy() for w in model.non_trainable_weights]),
        "input_shape"       : str(model.input_shape),
        "output_shape"      : str(model.output_shape),
        "num_layers"        : len(model.layers)
    }


# ─────────────────────────────────────────────
# Entry point — run training from CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="Train Land Use Classification Model")
    parser.add_argument(
        "--data_dir", type=str, required=True,
        help="Path to dataset root with class subdirectories"
    )
    parser.add_argument(
        "--transfer", action="store_true",
        help="Use MobileNetV2 transfer learning (default: custom CNN)"
    )
    parser.add_argument(
        "--epochs", type=int, default=EPOCHS,
        help=f"Max training epochs (default: {EPOCHS})"
    )
    parser.add_argument(
        "--batch_size", type=int, default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})"
    )

    args = parser.parse_args()

    trained_model, train_history = train_model(
        data_dir=args.data_dir,
        use_transfer_learning=args.transfer,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    summary = get_model_summary_dict(trained_model)
    print("\n Model Summary:")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")