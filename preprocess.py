"""
preprocess.py
-------------
Satellite Image Preprocessing Module for Land Use and Deforestation Analysis.

Responsibilities:
- Load and validate satellite images
- Resize to a standard input shape
- Normalize pixel values
- Apply noise filtering (Gaussian blur)
- Extract spectral feature maps (NDVI-like, RGB ratios)
- Convert to model-ready NumPy arrays

Author: AI Land Use Analysis System
"""

import cv2
import numpy as np
from pathlib import Path
import logging

# ─────────────────────────────────────────────
# Logging configuration
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
TARGET_SIZE = (256, 256)          # Standard input resolution for the model
NORMALIZE_MAX = 255.0             # Max pixel value for normalization
GAUSSIAN_KERNEL = (5, 5)         # Kernel size for noise reduction
GAUSSIAN_SIGMA = 1.0             # Sigma for Gaussian blur


# ─────────────────────────────────────────────
# Core Preprocessing Functions
# ─────────────────────────────────────────────

def load_image(image_source) -> np.ndarray:
    """
    Load a satellite image from a file path or from raw bytes (Streamlit upload).

    Parameters:
        image_source (str | Path | bytes): File path or raw image bytes.

    Returns:
        np.ndarray: Loaded image in BGR format (H x W x 3), uint8.

    Raises:
        ValueError: If the image cannot be loaded or decoded.
    """
    if isinstance(image_source, (str, Path)):
        image_path = str(image_source)
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        logger.info(f"Loaded image from path: {image_path} | Shape: {img.shape}")

    elif isinstance(image_source, bytes):
        # Streamlit UploadedFile → read() returns bytes
        nparr = np.frombuffer(image_source, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image from byte stream.")
        logger.info(f"Loaded image from bytes | Shape: {img.shape}")

    else:
        raise TypeError(f"Unsupported image source type: {type(image_source)}")

    return img


def resize_image(img: np.ndarray, target_size: tuple = TARGET_SIZE) -> np.ndarray:
    """
    Resize image to a fixed target resolution using area interpolation
    (best for downscaling high-resolution satellite imagery).

    Parameters:
        img (np.ndarray): Input image array.
        target_size (tuple): (width, height) target dimensions.

    Returns:
        np.ndarray: Resized image.
    """
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    logger.debug(f"Resized image to {target_size}")
    return resized


def apply_gaussian_filter(img: np.ndarray) -> np.ndarray:
    """
    Apply Gaussian blur to reduce sensor noise and atmospheric interference
    commonly found in satellite imagery.

    Parameters:
        img (np.ndarray): Input image array.

    Returns:
        np.ndarray: Smoothed image.
    """
    filtered = cv2.GaussianBlur(img, GAUSSIAN_KERNEL, GAUSSIAN_SIGMA)
    logger.debug("Applied Gaussian filter for noise reduction.")
    return filtered


def normalize_image(img: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to the [0.0, 1.0] range.
    Required for stable neural network training and inference.

    Parameters:
        img (np.ndarray): Image with pixel values in [0, 255].

    Returns:
        np.ndarray: Float32 image with values in [0.0, 1.0].
    """
    normalized = img.astype(np.float32) / NORMALIZE_MAX
    logger.debug("Normalized pixel values to [0, 1].")
    return normalized


def enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    to each channel individually to enhance local contrast in
    heterogeneous satellite imagery.

    Parameters:
        img (np.ndarray): BGR image (uint8).

    Returns:
        np.ndarray: Contrast-enhanced BGR image (uint8).
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(img)
    enhanced_channels = [clahe.apply(ch) for ch in channels]
    enhanced = cv2.merge(enhanced_channels)
    logger.debug("Applied CLAHE contrast enhancement.")
    return enhanced


def compute_vegetation_index(img: np.ndarray) -> np.ndarray:
    """
    Compute a proxy Vegetation Index using RGB channels since
    true NIR bands are not always available in standard satellite images.

    Formula: VI = (G - R) / (G + R + epsilon)
    where G = Green channel, R = Red channel.

    This approximates NDVI behavior using visible spectrum bands.

    Parameters:
        img (np.ndarray): Normalized float32 BGR image.

    Returns:
        np.ndarray: 2D Vegetation Index map, values in [-1, 1].
    """
    epsilon = 1e-6
    B, G, R = cv2.split(img)  # OpenCV uses BGR ordering
    vi = (G - R) / (G + R + epsilon)
    logger.debug("Computed proxy Vegetation Index (VI).")
    return vi


def compute_water_index(img: np.ndarray) -> np.ndarray:
    """
    Compute a proxy Water Index using RGB channels.

    Formula: WI = (B - G) / (B + G + epsilon)
    Blue-dominant areas (rivers, lakes) return higher values.

    Parameters:
        img (np.ndarray): Normalized float32 BGR image.

    Returns:
        np.ndarray: 2D Water Index map, values in [-1, 1].
    """
    epsilon = 1e-6
    B, G, R = cv2.split(img)
    wi = (B - G) / (B + G + epsilon)
    logger.debug("Computed proxy Water Index (WI).")
    return wi


def build_feature_stack(img_normalized: np.ndarray) -> np.ndarray:
    """
    Build a multi-channel feature stack by combining:
      - Original 3 RGB channels
      - Vegetation Index (1 channel)
      - Water Index (1 channel)

    This gives the model richer spectral information beyond raw RGB.

    Parameters:
        img_normalized (np.ndarray): Float32 normalized BGR image (H x W x 3).

    Returns:
        np.ndarray: Feature stack of shape (H x W x 5), float32.
    """
    vi = compute_vegetation_index(img_normalized)
    wi = compute_water_index(img_normalized)

    # Expand single-channel indices to 3D for stacking
    vi_channel = np.expand_dims(vi, axis=-1)
    wi_channel = np.expand_dims(wi, axis=-1)

    feature_stack = np.concatenate([img_normalized, vi_channel, wi_channel], axis=-1)
    logger.debug(f"Built feature stack | Shape: {feature_stack.shape}")
    return feature_stack


def preprocess_image(image_source, use_feature_stack: bool = False) -> np.ndarray:
    """
    Master preprocessing pipeline. Runs the full sequence:
        Load → Enhance Contrast → Resize → Gaussian Filter → Normalize
        → [Optional: Build Feature Stack]

    Parameters:
        image_source (str | Path | bytes): Image path or byte stream.
        use_feature_stack (bool): If True, returns 5-channel feature array.
                                  If False, returns standard 3-channel RGB.

    Returns:
        np.ndarray: Preprocessed image ready for model input.
                    Shape: (256, 256, 3) or (256, 256, 5)
    """
    logger.info("Starting preprocessing pipeline...")

    # Step 1: Load raw image
    img = load_image(image_source)

    # Step 2: Enhance contrast for better feature visibility
    img = enhance_contrast(img)

    # Step 3: Resize to standard dimensions
    img = resize_image(img, TARGET_SIZE)

    # Step 4: Reduce sensor noise
    img = apply_gaussian_filter(img)

    # Step 5: Normalize pixel values
    img_normalized = normalize_image(img)

    # Step 6 (optional): Append spectral indices as extra channels
    if use_feature_stack:
        result = build_feature_stack(img_normalized)
    else:
        result = img_normalized

    logger.info(f"Preprocessing complete | Output shape: {result.shape}")
    return result


def preprocess_for_display(image_source) -> np.ndarray:
    """
    Lightweight preprocessing for dashboard display purposes only.
    Does NOT normalize (keeps uint8 values for correct rendering).

    Parameters:
        image_source (str | Path | bytes): Image path or byte stream.

    Returns:
        np.ndarray: Resized uint8 BGR image (256, 256, 3).
    """
    img = load_image(image_source)
    img = resize_image(img, TARGET_SIZE)
    # Convert BGR (OpenCV) → RGB (Streamlit/Matplotlib)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb


def get_image_statistics(img: np.ndarray) -> dict:
    """
    Compute basic statistics of a preprocessed image for logging
    and report generation.

    Parameters:
        img (np.ndarray): Preprocessed image array (float32 or uint8).

    Returns:
        dict: Statistics including mean, std, min, max per channel.
    """
    stats = {}
    channel_names = ['Blue', 'Green', 'Red']

    if img.ndim == 3:
        for i, name in enumerate(channel_names[:img.shape[2]]):
            channel = img[:, :, i]
            stats[name] = {
                'mean': float(np.mean(channel)),
                'std': float(np.std(channel)),
                'min': float(np.min(channel)),
                'max': float(np.max(channel))
            }
    else:
        stats['grayscale'] = {
            'mean': float(np.mean(img)),
            'std': float(np.std(img)),
            'min': float(np.min(img)),
            'max': float(np.max(img))
        }

    logger.debug(f"Image statistics computed: {stats}")
    return stats


# ─────────────────────────────────────────────
# Quick test / standalone run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <image_path>")
        sys.exit(1)

    test_path = sys.argv[1]
    print(f"\n Testing preprocessing on: {test_path}")

    processed = preprocess_image(test_path, use_feature_stack=True)
    print(f" Output shape       : {processed.shape}")
    print(f" Output dtype       : {processed.dtype}")
    print(f" Pixel value range  : [{processed.min():.4f}, {processed.max():.4f}]")

    stats = get_image_statistics(processed[:, :, :3])
    for ch, s in stats.items():
        print(f" {ch}: mean={s['mean']:.4f}, std={s['std']:.4f}")