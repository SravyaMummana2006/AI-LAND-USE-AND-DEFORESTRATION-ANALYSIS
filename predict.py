"""
predict.py
----------
Model Inference Module for Land Use Classification.

Responsibilities:
- Load saved TensorFlow model and class label mapping
- Run inference on single or batch satellite images
- Generate per-class probability distributions
- Produce pixel-level land-use segmentation maps (via sliding window)
- Generate color-coded classification overlay images
- Compute land-use area statistics (% coverage per class)

Land Use Classes:
    0 → Forest       (Dark Green)
    1 → Water        (Blue)
    2 → Urban        (Red)
    3 → Agriculture  (Yellow)
    4 → Barren       (Tan/Brown)

Author: AI Land Use Analysis System
"""

import cv2
import json
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from typing import Optional

from preprocess import preprocess_image, preprocess_for_display, TARGET_SIZE

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
MODEL_SAVE_PATH     = "saved_model/land_use_model.h5"
LABELS_SAVE_PATH    = "saved_model/class_labels.json"

# Sliding window configuration for pixel-level segmentation
PATCH_SIZE          = 64    # Size of each patch sent to the model
PATCH_STRIDE        = 32    # Stride between patches (50% overlap for smoother map)

# Color map: BGR colors for each class (used by OpenCV)
CLASS_COLOR_MAP_BGR = {
    "Forest"      : (34,  139, 34),    # Forest Green
    "Water"       : (205, 133, 0),     # Blue (BGR)
    "Urban"       : (60,  20,  220),   # Crimson Red
    "Agriculture" : (0,   200, 200),   # Yellow (BGR)
    "Barren"      : (130, 180, 210),   # Tan/Sandy Brown
}

# RGB version for Streamlit/Matplotlib display
CLASS_COLOR_MAP_RGB = {
    "Forest"      : (34,  139, 34),
    "Water"       : (0,   133, 205),
    "Urban"       : (220, 20,  60),
    "Agriculture" : (200, 200, 0),
    "Barren"      : (210, 180, 130),
}

# Fallback class labels if JSON not found
DEFAULT_CLASS_LABELS = {
    "0": "Forest",
    "1": "Water",
    "2": "Urban",
    "3": "Agriculture",
    "4": "Barren"
}


# ─────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────

def load_model(model_path: str = MODEL_SAVE_PATH) -> tf.keras.Model:
    """
    Load the saved TensorFlow/Keras model from disk.

    Uses a module-level cache so the model is only loaded once per session,
    avoiding repeated expensive disk reads during multi-image inference.

    Parameters:
        model_path (str): Path to the saved .h5 model file.

    Returns:
        tf.keras.Model: Loaded and ready-to-infer model.

    Raises:
        FileNotFoundError: If model file does not exist at given path.
    """
    global _CACHED_MODEL

    # Return cached model if already loaded
    if _CACHED_MODEL is not None:
        logger.debug("Returning cached model.")
        return _CACHED_MODEL

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}\n"
            f"Please run train_model.py first to generate the model."
        )

    logger.info(f"Loading model from: {model_path}")
    _CACHED_MODEL = tf.keras.models.load_model(model_path)
    logger.info(f"Model loaded successfully | "
                f"Input: {_CACHED_MODEL.input_shape} | "
                f"Output: {_CACHED_MODEL.output_shape}")
    return _CACHED_MODEL


# Module-level model cache (initialized as None)
_CACHED_MODEL: Optional[tf.keras.Model] = None


def load_class_labels(labels_path: str = LABELS_SAVE_PATH) -> dict:
    """
    Load class index → label mapping from JSON file saved during training.

    Parameters:
        labels_path (str): Path to class_labels.json.

    Returns:
        dict: e.g. {"0": "Forest", "1": "Water", ...}
    """
    labels_file = Path(labels_path)
    if not labels_file.exists():
        logger.warning(
            f"Class labels not found at {labels_path}. "
            f"Using default mapping."
        )
        return DEFAULT_CLASS_LABELS

    with open(labels_path, 'r') as f:
        labels = json.load(f)

    logger.info(f"Class labels loaded: {labels}")
    return labels


# ─────────────────────────────────────────────
# Core Inference
# ─────────────────────────────────────────────

def predict_single_image(image_source,
                          model: tf.keras.Model,
                          class_labels: dict) -> dict:
    """
    Run inference on a single satellite image and return full prediction results.

    Pipeline:
        Preprocess → Add batch dim → Model.predict() → Decode class → Build result dict

    Parameters:
        image_source (str | Path | bytes): Raw image input.
        model (tf.keras.Model): Loaded classification model.
        class_labels (dict): Index-to-label mapping.

    Returns:
        dict with keys:
            - predicted_class     : str  — dominant land-use class name
            - class_index         : int  — numeric index of predicted class
            - confidence          : float — confidence score [0, 1]
            - probabilities       : dict — {class_name: probability} for all classes
            - top_3               : list — top 3 (class, probability) tuples
    """
    logger.info("Running single-image inference...")

    # Step 1: Preprocess to (256, 256, 3) float32
    img_array = preprocess_image(image_source, use_feature_stack=False)

    # Step 2: Add batch dimension → (1, 256, 256, 3)
    img_batch = np.expand_dims(img_array, axis=0)

    # Step 3: Model inference
    raw_probs = model.predict(img_batch, verbose=0)[0]  # Shape: (num_classes,)

    # Step 4: Decode predictions
    class_index = int(np.argmax(raw_probs))
    confidence  = float(raw_probs[class_index])
    predicted_class = class_labels.get(str(class_index), f"Class_{class_index}")

    # Step 5: Build probability dict mapped to class names
    probabilities = {
        class_labels.get(str(i), f"Class_{i}"): float(raw_probs[i])
        for i in range(len(raw_probs))
    }

    # Step 6: Top-3 predictions
    top_indices = np.argsort(raw_probs)[::-1][:3]
    top_3 = [
        (class_labels.get(str(i), f"Class_{i}"), float(raw_probs[i]))
        for i in top_indices
    ]

    result = {
        "predicted_class" : predicted_class,
        "class_index"     : class_index,
        "confidence"      : confidence,
        "probabilities"   : probabilities,
        "top_3"           : top_3
    }

    logger.info(
        f"Prediction → {predicted_class} "
        f"(confidence: {confidence:.2%})"
    )
    return result


def predict_batch(image_sources: list,
                  model: tf.keras.Model,
                  class_labels: dict) -> list:
    """
    Run inference on a batch of images efficiently using a single model call.

    Parameters:
        image_sources (list): List of image paths or byte streams.
        model (tf.keras.Model): Loaded classification model.
        class_labels (dict): Index-to-label mapping.

    Returns:
        list: List of prediction result dicts (same format as predict_single_image).
    """
    logger.info(f"Running batch inference on {len(image_sources)} images...")

    # Preprocess all images and stack into a single batch tensor
    batch = np.stack([
        preprocess_image(src, use_feature_stack=False)
        for src in image_sources
    ], axis=0)   # Shape: (N, 256, 256, 3)

    raw_probs_batch = model.predict(batch, verbose=0)  # Shape: (N, num_classes)

    results = []
    for raw_probs in raw_probs_batch:
        class_index     = int(np.argmax(raw_probs))
        confidence      = float(raw_probs[class_index])
        predicted_class = class_labels.get(str(class_index), f"Class_{class_index}")

        probabilities = {
            class_labels.get(str(i), f"Class_{i}"): float(raw_probs[i])
            for i in range(len(raw_probs))
        }

        top_indices = np.argsort(raw_probs)[::-1][:3]
        top_3 = [
            (class_labels.get(str(i), f"Class_{i}"), float(raw_probs[i]))
            for i in top_indices
        ]

        results.append({
            "predicted_class" : predicted_class,
            "class_index"     : class_index,
            "confidence"      : confidence,
            "probabilities"   : probabilities,
            "top_3"           : top_3
        })

    logger.info(f"Batch inference complete for {len(results)} images.")
    return results


# ─────────────────────────────────────────────
# Pixel-Level Segmentation (Sliding Window)
# ─────────────────────────────────────────────

def generate_segmentation_map(image_source,
                               model: tf.keras.Model,
                               class_labels: dict,
                               patch_size: int = PATCH_SIZE,
                               stride: int = PATCH_STRIDE) -> np.ndarray:
    """
    Generate a pixel-level land-use segmentation map using a sliding window
    approach. Each patch is classified independently and the result is
    assembled into a full-resolution class index map.

    The sliding window uses 50% overlap (stride = patch_size / 2) and
    averages overlapping predictions for smoother boundaries.

    Parameters:
        image_source: Raw image input.
        model (tf.keras.Model): Loaded classification model.
        class_labels (dict): Index-to-label mapping.
        patch_size (int): Size of each square patch in pixels.
        stride (int): Step size between patches.

    Returns:
        np.ndarray: 2D array of shape (H, W) containing class indices (int).
    """
    logger.info(f"Generating segmentation map | "
                f"patch={patch_size}, stride={stride}")

    # Load and preprocess the full image
    img = preprocess_image(image_source, use_feature_stack=False)
    H, W, C = img.shape
    num_classes = len(class_labels)

    # Accumulator: sum of probabilities at each pixel position
    prob_accumulator = np.zeros((H, W, num_classes), dtype=np.float32)
    # Count: number of patches covering each pixel (for averaging)
    count_map        = np.zeros((H, W), dtype=np.float32)

    patches = []
    positions = []

    # Collect all patches
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            patch = img[y:y + patch_size, x:x + patch_size, :]
            # Resize patch to model input size
            patch_resized = cv2.resize(patch, TARGET_SIZE,
                                       interpolation=cv2.INTER_LINEAR)
            patches.append(patch_resized)
            positions.append((y, x))

    if not patches:
        logger.warning("No patches extracted — image may be smaller than patch size.")
        return np.zeros((H, W), dtype=np.int32)

    # Batch-predict all patches
    patch_batch  = np.stack(patches, axis=0)  # (N, 256, 256, 3)
    all_probs    = model.predict(patch_batch, verbose=0, batch_size=32)

    # Accumulate probabilities back to spatial positions
    for idx, (y, x) in enumerate(positions):
        # Downscale prediction back to patch spatial coverage
        probs_spatial = np.ones((patch_size, patch_size, num_classes),
                                dtype=np.float32)
        probs_spatial *= all_probs[idx]  # Broadcast class probs across patch area
        prob_accumulator[y:y + patch_size, x:x + patch_size, :] += probs_spatial
        count_map[y:y + patch_size, x:x + patch_size]            += 1.0

    # Avoid division by zero for uncovered border pixels
    count_map = np.where(count_map == 0, 1, count_map)
    prob_accumulator /= count_map[:, :, np.newaxis]

    # Final class map: argmax of averaged probabilities
    class_map = np.argmax(prob_accumulator, axis=-1).astype(np.int32)

    logger.info(f"Segmentation map generated | Shape: {class_map.shape}")
    return class_map


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def colorize_segmentation_map(class_map: np.ndarray,
                               class_labels: dict) -> np.ndarray:
    """
    Convert a 2D class index map into an RGB color image where each
    land-use class is rendered with its designated color.

    Parameters:
        class_map (np.ndarray): 2D integer array of class indices.
        class_labels (dict): Index-to-name mapping.

    Returns:
        np.ndarray: RGB image (H x W x 3), uint8.
    """
    H, W = class_map.shape
    color_img = np.zeros((H, W, 3), dtype=np.uint8)

    for idx_str, class_name in class_labels.items():
        idx   = int(idx_str)
        color = CLASS_COLOR_MAP_RGB.get(class_name, (128, 128, 128))
        mask  = (class_map == idx)
        color_img[mask] = color

    logger.debug(f"Segmentation map colorized | Unique classes: {np.unique(class_map)}")
    return color_img


def create_overlay_image(original_image_source,
                          class_map: np.ndarray,
                          class_labels: dict,
                          alpha: float = 0.45) -> np.ndarray:
    """
    Blend the original satellite image with the color-coded segmentation map
    to produce a classification overlay for dashboard display.

    Parameters:
        original_image_source: Raw image input (path or bytes).
        class_map (np.ndarray): 2D class index map.
        class_labels (dict): Index-to-name mapping.
        alpha (float): Transparency of the segmentation layer [0=original, 1=full color].

    Returns:
        np.ndarray: RGB blended overlay image (H x W x 3), uint8.
    """
    # Load display-ready original (uint8, RGB)
    original_rgb = preprocess_for_display(original_image_source)

    # Generate color segmentation
    color_seg    = colorize_segmentation_map(class_map, class_labels)

    # Resize color map to match original if needed
    if color_seg.shape[:2] != original_rgb.shape[:2]:
        color_seg = cv2.resize(
            color_seg,
            (original_rgb.shape[1], original_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

    # Alpha blending: overlay = (1-alpha)*original + alpha*segmentation
    overlay = cv2.addWeighted(
        original_rgb.astype(np.float32), 1.0 - alpha,
        color_seg.astype(np.float32),    alpha,
        0
    ).astype(np.uint8)

    logger.debug("Overlay image created.")
    return overlay


# ─────────────────────────────────────────────
# Area Statistics
# ─────────────────────────────────────────────

def compute_area_statistics(class_map: np.ndarray,
                             class_labels: dict) -> pd.DataFrame:
    """
    Compute the percentage area coverage for each land-use class
    from the segmentation map.

    Parameters:
        class_map (np.ndarray): 2D integer array of class indices.
        class_labels (dict): Index-to-name mapping.

    Returns:
        pd.DataFrame: Columns — ['Class', 'Pixel_Count', 'Percentage']
                      Sorted by Percentage descending.
    """
    total_pixels = class_map.size
    records = []

    for idx_str, class_name in class_labels.items():
        idx         = int(idx_str)
        pixel_count = int(np.sum(class_map == idx))
        percentage  = round((pixel_count / total_pixels) * 100, 2)
        records.append({
            "Class"       : class_name,
            "Pixel_Count" : pixel_count,
            "Percentage"  : percentage
        })

    df = pd.DataFrame(records).sort_values("Percentage", ascending=False)
    df = df.reset_index(drop=True)

    logger.info("Area statistics computed:")
    for _, row in df.iterrows():
        logger.info(f"  {row['Class']:12s}: {row['Percentage']:5.1f}%")

    return df


def get_dominant_classes(area_df: pd.DataFrame,
                          top_n: int = 3) -> list:
    """
    Return the top N dominant land-use classes by area coverage.

    Parameters:
        area_df (pd.DataFrame): Output from compute_area_statistics().
        top_n (int): Number of top classes to return.

    Returns:
        list: List of dicts with 'Class' and 'Percentage'.
    """
    top = area_df.head(top_n)[['Class', 'Percentage']].to_dict('records')
    return top


# ─────────────────────────────────────────────
# Full Prediction Pipeline
# ─────────────────────────────────────────────

def run_full_prediction(image_source,
                         model_path: str = MODEL_SAVE_PATH,
                         labels_path: str = LABELS_SAVE_PATH) -> dict:
    """
    End-to-end prediction pipeline for a single satellite image.

    Steps:
        1. Load model and labels
        2. Global classification (whole-image prediction)
        3. Sliding-window segmentation map
        4. Colorized segmentation image
        5. Overlay image
        6. Area statistics DataFrame

    Parameters:
        image_source: Raw image input (path or bytes).
        model_path (str): Path to saved model.
        labels_path (str): Path to class labels JSON.

    Returns:
        dict with keys:
            - global_prediction  : dict (from predict_single_image)
            - class_map          : np.ndarray (H x W int)
            - color_map          : np.ndarray (H x W x 3 RGB)
            - overlay            : np.ndarray (H x W x 3 RGB)
            - area_stats         : pd.DataFrame
            - dominant_classes   : list of dicts
    """
    logger.info("=" * 55)
    logger.info("RUNNING FULL PREDICTION PIPELINE")
    logger.info("=" * 55)

    # Load model and labels
    model        = load_model(model_path)
    class_labels = load_class_labels(labels_path)

    # Global image-level classification
    global_pred = predict_single_image(image_source, model, class_labels)

    # Pixel-level segmentation
    class_map   = generate_segmentation_map(image_source, model, class_labels)

    # Colorized segmentation map (RGB)
    color_map   = colorize_segmentation_map(class_map, class_labels)

    # Blended overlay
    overlay     = create_overlay_image(image_source, class_map, class_labels)

    # Area coverage statistics
    area_stats  = compute_area_statistics(class_map, class_labels)

    # Top dominant classes
    dominant    = get_dominant_classes(area_stats)

    logger.info("Full prediction pipeline complete.")
    logger.info(f"Global class  : {global_pred['predicted_class']} "
                f"({global_pred['confidence']:.2%})")
    logger.info(f"Dominant area : {dominant[0]['Class']} "
                f"({dominant[0]['Percentage']}%)")

    return {
        "global_prediction" : global_pred,
        "class_map"         : class_map,
        "color_map"         : color_map,
        "overlay"           : overlay,
        "area_stats"        : area_stats,
        "dominant_classes"  : dominant
    }


def compare_predictions(result_old: dict,
                         result_new: dict) -> dict:
    """
    Compare land-use statistics between two time-period predictions
    to quantify class-level changes.

    Parameters:
        result_old (dict): Output of run_full_prediction() for older image.
        result_new (dict): Output of run_full_prediction() for newer image.

    Returns:
        dict: Per-class percentage change and absolute delta.
              Keys: class names → {'old_%', 'new_%', 'change_%', 'direction'}
    """
    old_stats = result_old["area_stats"].set_index("Class")["Percentage"].to_dict()
    new_stats = result_new["area_stats"].set_index("Class")["Percentage"].to_dict()

    all_classes = set(old_stats.keys()) | set(new_stats.keys())
    comparison  = {}

    for cls in all_classes:
        old_pct = old_stats.get(cls, 0.0)
        new_pct = new_stats.get(cls, 0.0)
        delta   = round(new_pct - old_pct, 2)
        direction = "increased" if delta > 0 else "decreased" if delta < 0 else "unchanged"

        comparison[cls] = {
            "old_pct"   : old_pct,
            "new_pct"   : new_pct,
            "change_pct": delta,
            "direction" : direction
        }

    logger.info("Prediction comparison complete:")
    for cls, vals in comparison.items():
        logger.info(
            f"  {cls:12s}: {vals['old_pct']:5.1f}% → "
            f"{vals['new_pct']:5.1f}% ({vals['direction']}: {abs(vals['change_pct'])}%)"
        )

    return comparison


# ─────────────────────────────────────────────
# Legend Utility
# ─────────────────────────────────────────────

def get_class_color_legend() -> dict:
    """
    Return the class → RGB color mapping for dashboard legend rendering.

    Returns:
        dict: {class_name: (R, G, B)} for all land-use classes.
    """
    return CLASS_COLOR_MAP_RGB.copy()


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    print(f"\n Running prediction on: {img_path}")

    try:
        result = run_full_prediction(img_path)

        print(f"\n Global Prediction:")
        print(f"  Class      : {result['global_prediction']['predicted_class']}")
        print(f"  Confidence : {result['global_prediction']['confidence']:.2%}")

        print(f"\n Area Statistics:")
        print(result["area_stats"].to_string(index=False))

        print(f"\n Dominant Classes:")
        for d in result["dominant_classes"]:
            print(f"  {d['Class']:12s}: {d['Percentage']}%")

    except FileNotFoundError as e:
        print(f"\n Error: {e}")
        sys.exit(1)