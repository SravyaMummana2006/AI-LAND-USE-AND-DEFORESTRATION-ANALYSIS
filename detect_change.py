"""
detect_change.py
----------------
Deforestation & Land Use Change Detection Module.

Responsibilities:
- Align and compare old vs new satellite images using OpenCV
- Compute pixel-level difference maps
- Isolate and quantify forest loss regions
- Detect urban expansion, agricultural spread, and water body changes
- Generate annotated change-highlight overlays
- Assign risk classification (Low / Medium / High / Critical)
- Produce structured change report data for report_generator.py

Author: AI Land Use Analysis System
"""

import cv2
import logging
import numpy as np
import pandas as pd
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
# Constants & Thresholds
# ─────────────────────────────────────────────

# Pixel difference threshold — changes below this are treated as noise
DIFF_THRESHOLD          = 30        # Out of 255 (uint8 scale)

# Minimum contour area (pixels) to be counted as a real changed region
MIN_CONTOUR_AREA        = 150

# Morphological kernel sizes for cleaning up binary change masks
MORPH_KERNEL_CLOSE      = (7, 7)    # Closing: fills small holes in detected regions
MORPH_KERNEL_OPEN       = (5, 5)    # Opening: removes small noise blobs

# Risk level thresholds (based on forest loss percentage)
RISK_THRESHOLDS = {
    "Low"      : (0.0,  10.0),
    "Medium"   : (10.0, 25.0),
    "High"     : (25.0, 50.0),
    "Critical" : (50.0, 100.0)
}

# Highlight colors for changed regions (RGB)
CHANGE_COLORS = {
    "forest_loss"    : (220, 50,  50),   # Red — deforested areas
    "urban_gain"     : (255, 165, 0),    # Orange — new urban regions
    "water_change"   : (0,   120, 255),  # Blue — water body changes
    "vegetation_gain": (50,  200, 50),   # Green — new vegetation (reforestation)
    "general_change" : (255, 255, 0),    # Yellow — other changes
    "contour"        : (255, 0,   0),    # Red contour borders
}

# Forest detection: HSV hue range for green vegetation
FOREST_HSV_LOWER = np.array([35,  40,  40],  dtype=np.uint8)
FOREST_HSV_UPPER = np.array([90,  255, 255], dtype=np.uint8)

# Water detection: HSV hue range for blue water bodies
WATER_HSV_LOWER  = np.array([90,  50,  50],  dtype=np.uint8)
WATER_HSV_UPPER  = np.array([130, 255, 255], dtype=np.uint8)

# Urban detection: low saturation (grey concrete), moderate value
URBAN_HSV_LOWER  = np.array([0,   0,   80],  dtype=np.uint8)
URBAN_HSV_UPPER  = np.array([180, 50,  220], dtype=np.uint8)


# ─────────────────────────────────────────────
# Image Alignment
# ─────────────────────────────────────────────

def align_images(img_old: np.ndarray,
                 img_new: np.ndarray) -> tuple:
    """
    Align the new image to the old image using feature-based registration
    (ORB keypoints + homography). This corrects for slight positional
    differences between satellite captures at different times.

    If alignment fails (insufficient keypoint matches), the new image
    is returned as-is with a warning.

    Parameters:
        img_old (np.ndarray): Reference image (float32, normalized).
        img_new (np.ndarray): Target image to align (float32, normalized).

    Returns:
        tuple: (aligned_img_old, aligned_img_new) as float32 arrays.
    """
    logger.info("Aligning images using ORB feature matching...")

    # Convert to uint8 for ORB detection
    old_u8 = (img_old * 255).astype(np.uint8)
    new_u8 = (img_new * 255).astype(np.uint8)

    old_gray = cv2.cvtColor(old_u8, cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor(new_u8, cv2.COLOR_BGR2GRAY)

    # ORB: fast and license-free keypoint detector
    orb = cv2.ORB_create(nfeatures=1000)
    kp_old, desc_old = orb.detectAndCompute(old_gray, None)
    kp_new, desc_new = orb.detectAndCompute(new_gray, None)

    if desc_old is None or desc_new is None or len(kp_old) < 4:
        logger.warning("Insufficient keypoints for alignment. Using original images.")
        return img_old, img_new

    # Brute-force matcher with Hamming distance (ORB is binary)
    bf      = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_old, desc_new)
    matches = sorted(matches, key=lambda m: m.distance)

    # Keep best 50% of matches
    good_matches = matches[:max(10, len(matches) // 2)]

    if len(good_matches) < 4:
        logger.warning("Too few good matches for homography. Skipping alignment.")
        return img_old, img_new

    # Extract matched keypoint coordinates
    pts_old = np.float32([kp_old[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_new = np.float32([kp_new[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Compute homography using RANSAC for robustness to outliers
    H_matrix, mask = cv2.findHomography(pts_new, pts_old, cv2.RANSAC, 5.0)

    if H_matrix is None:
        logger.warning("Homography computation failed. Using original images.")
        return img_old, img_new

    # Warp new image to align with old image
    h, w = img_old.shape[:2]
    aligned_new = cv2.warpPerspective(img_new, H_matrix, (w, h))

    inlier_count = int(mask.sum()) if mask is not None else 0
    logger.info(f"Alignment complete | Inliers: {inlier_count}/{len(good_matches)}")

    return img_old, aligned_new


# ─────────────────────────────────────────────
# Difference Map Computation
# ─────────────────────────────────────────────

def compute_difference_map(img_old: np.ndarray,
                            img_new: np.ndarray) -> np.ndarray:
    """
    Compute a per-pixel absolute difference map between old and new images.
    The result captures all regions that have changed between the two captures.

    Parameters:
        img_old (np.ndarray): Preprocessed old image (float32, H x W x 3).
        img_new (np.ndarray): Preprocessed new image (float32, H x W x 3).

    Returns:
        np.ndarray: Grayscale difference magnitude map, uint8 [0-255].
    """
    # Absolute per-channel difference
    diff = np.abs(img_old.astype(np.float32) - img_new.astype(np.float32))

    # Collapse channels by taking maximum change across R, G, B
    diff_gray = np.max(diff, axis=-1)

    # Scale back to uint8 for morphological operations
    diff_u8 = (diff_gray * 255).clip(0, 255).astype(np.uint8)

    logger.debug(f"Difference map | min={diff_u8.min()}, "
                 f"max={diff_u8.max()}, mean={diff_u8.mean():.2f}")
    return diff_u8


def compute_structural_similarity_map(img_old: np.ndarray,
                                       img_new: np.ndarray) -> np.ndarray:
    """
    Compute a structural dissimilarity map using local mean and variance
    to capture texture-level changes beyond simple pixel differences.

    Parameters:
        img_old (np.ndarray): Normalized float32 image.
        img_new (np.ndarray): Normalized float32 image.

    Returns:
        np.ndarray: Grayscale dissimilarity map, uint8 [0-255].
    """
    old_gray = cv2.cvtColor((img_old * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    new_gray = cv2.cvtColor((img_new * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Local mean using box filter
    mu_old = cv2.blur(old_gray.astype(np.float32), (11, 11))
    mu_new = cv2.blur(new_gray.astype(np.float32), (11, 11))

    # Local variance
    var_old = cv2.blur((old_gray.astype(np.float32) ** 2), (11, 11)) - mu_old ** 2
    var_new = cv2.blur((new_gray.astype(np.float32) ** 2), (11, 11)) - mu_new ** 2

    # Covariance between old and new
    cov = cv2.blur(
        old_gray.astype(np.float32) * new_gray.astype(np.float32),
        (11, 11)
    ) - mu_old * mu_new

    # SSIM-like dissimilarity (1 - SSIM)
    C1, C2 = 6.5025, 58.5225
    ssim_map = (
        (2 * mu_old * mu_new + C1) * (2 * cov + C2)
    ) / (
        (mu_old ** 2 + mu_new ** 2 + C1) * (var_old + var_new + C2)
    )
    dissim = ((1.0 - ssim_map) * 127.5).clip(0, 255).astype(np.uint8)

    logger.debug("Structural dissimilarity map computed.")
    return dissim


def fuse_change_maps(diff_map: np.ndarray,
                     ssim_map: np.ndarray,
                     w_diff: float = 0.6,
                     w_ssim: float = 0.4) -> np.ndarray:
    """
    Fuse the raw pixel difference map and structural dissimilarity map
    into a single robust change score map using weighted averaging.

    Parameters:
        diff_map (np.ndarray): Pixel difference map (uint8).
        ssim_map (np.ndarray): Structural dissimilarity map (uint8).
        w_diff (float): Weight for pixel difference map.
        w_ssim (float): Weight for structural map.

    Returns:
        np.ndarray: Fused change map (uint8).
    """
    fused = (
        w_diff * diff_map.astype(np.float32) +
        w_ssim * ssim_map.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    logger.debug(f"Fused change map | mean={fused.mean():.2f}")
    return fused


# ─────────────────────────────────────────────
# Binary Mask Generation
# ─────────────────────────────────────────────

def generate_change_mask(change_map: np.ndarray,
                          threshold: int = DIFF_THRESHOLD) -> np.ndarray:
    """
    Threshold the fused change map to produce a binary mask of changed regions.
    Applies morphological closing and opening to remove noise and fill gaps.

    Parameters:
        change_map (np.ndarray): Grayscale change score map (uint8).
        threshold (int): Pixel intensity threshold for change detection.

    Returns:
        np.ndarray: Binary mask — 255 = changed, 0 = unchanged (uint8).
    """
    # Threshold: pixels above threshold are changed
    _, binary = cv2.threshold(change_map, threshold, 255, cv2.THRESH_BINARY)

    # Morphological closing: connect nearby changed pixels
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, MORPH_KERNEL_CLOSE
    )
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Morphological opening: remove isolated noise pixels
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, MORPH_KERNEL_OPEN
    )
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

    changed_pct = (np.sum(cleaned > 0) / cleaned.size) * 100
    logger.info(f"Change mask generated | Changed area: {changed_pct:.1f}%")
    return cleaned


# ─────────────────────────────────────────────
# Spectral Class Masks (HSV-based)
# ─────────────────────────────────────────────

def extract_forest_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract a binary mask of forest/vegetation pixels using HSV color thresholding.
    Green-dominant pixels in the satellite image indicate forest cover.

    Parameters:
        img_rgb (np.ndarray): Display-ready RGB image (uint8).

    Returns:
        np.ndarray: Binary mask — 255 = forest pixel (uint8).
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask    = cv2.inRange(img_hsv, FOREST_HSV_LOWER, FOREST_HSV_UPPER)

    # Clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel)

    logger.debug(f"Forest mask | Coverage: "
                 f"{(np.sum(mask > 0) / mask.size * 100):.1f}%")
    return mask


def extract_water_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract a binary mask of water body pixels using HSV blue-range thresholding.

    Parameters:
        img_rgb (np.ndarray): Display-ready RGB image (uint8).

    Returns:
        np.ndarray: Binary mask — 255 = water pixel (uint8).
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask    = cv2.inRange(img_hsv, WATER_HSV_LOWER, WATER_HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    logger.debug(f"Water mask | Coverage: "
                 f"{(np.sum(mask > 0) / mask.size * 100):.1f}%")
    return mask


def extract_urban_mask(img_rgb: np.ndarray) -> np.ndarray:
    """
    Extract a binary mask of urban/built-up pixels using low-saturation
    HSV thresholding (concrete, roads, and buildings appear greyish).

    Parameters:
        img_rgb (np.ndarray): Display-ready RGB image (uint8).

    Returns:
        np.ndarray: Binary mask — 255 = urban pixel (uint8).
    """
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask    = cv2.inRange(img_hsv, URBAN_HSV_LOWER, URBAN_HSV_UPPER)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    logger.debug(f"Urban mask | Coverage: "
                 f"{(np.sum(mask > 0) / mask.size * 100):.1f}%")
    return mask


# ─────────────────────────────────────────────
# Forest Loss Quantification
# ─────────────────────────────────────────────

def compute_forest_loss(old_forest_mask: np.ndarray,
                         new_forest_mask: np.ndarray,
                         change_mask: np.ndarray) -> dict:
    """
    Compute deforestation metrics by combining the forest masks from
    both time periods with the general change mask.

    Forest loss pixels: were forest in OLD image AND changed AND not
    forest in NEW image.

    Parameters:
        old_forest_mask (np.ndarray): Binary forest mask from old image.
        new_forest_mask (np.ndarray): Binary forest mask from new image.
        change_mask (np.ndarray): Binary change mask (from fused change map).

    Returns:
        dict with forest loss mask, gain mask, and percentage metrics.
    """
    total_pixels = old_forest_mask.size

    # Forest loss: was forest, changed, no longer forest
    loss_mask = cv2.bitwise_and(
        old_forest_mask,
        cv2.bitwise_and(
            change_mask,
            cv2.bitwise_not(new_forest_mask)
        )
    )

    # Forest gain: was not forest, changed, now forest (reforestation)
    gain_mask = cv2.bitwise_and(
        cv2.bitwise_not(old_forest_mask),
        cv2.bitwise_and(
            change_mask,
            new_forest_mask
        )
    )

    # Area calculations
    old_forest_pct  = round(float(np.sum(old_forest_mask > 0)) / total_pixels * 100, 2)
    new_forest_pct  = round(float(np.sum(new_forest_mask > 0)) / total_pixels * 100, 2)
    loss_pct        = round(float(np.sum(loss_mask > 0)) / total_pixels * 100, 2)
    gain_pct        = round(float(np.sum(gain_mask > 0)) / total_pixels * 100, 2)
    net_change_pct  = round(new_forest_pct - old_forest_pct, 2)

    results = {
        "forest_loss_mask"      : loss_mask,
        "forest_gain_mask"      : gain_mask,
        "old_forest_pct"        : old_forest_pct,
        "new_forest_pct"        : new_forest_pct,
        "forest_loss_pct"       : loss_pct,
        "forest_gain_pct"       : gain_pct,
        "net_forest_change_pct" : net_change_pct,
        "loss_pixel_count"      : int(np.sum(loss_mask > 0)),
        "gain_pixel_count"      : int(np.sum(gain_mask > 0))
    }

    logger.info(f"Forest analysis:")
    logger.info(f"  Old cover  : {old_forest_pct:.1f}%")
    logger.info(f"  New cover  : {new_forest_pct:.1f}%")
    logger.info(f"  Net change : {net_change_pct:+.1f}%")
    logger.info(f"  Loss area  : {loss_pct:.1f}%")
    logger.info(f"  Gain area  : {gain_pct:.1f}%")

    return results


def compute_urban_expansion(old_urban_mask: np.ndarray,
                             new_urban_mask: np.ndarray) -> dict:
    """
    Quantify urban sprawl by comparing urban pixel coverage between
    old and new images.

    Parameters:
        old_urban_mask (np.ndarray): Urban mask from old image.
        new_urban_mask (np.ndarray): Urban mask from new image.

    Returns:
        dict: Urban expansion metrics and pixel masks.
    """
    total_pixels = old_urban_mask.size

    # New urban areas: not urban before, urban now
    expansion_mask = cv2.bitwise_and(
        cv2.bitwise_not(old_urban_mask),
        new_urban_mask
    )

    old_urban_pct  = round(float(np.sum(old_urban_mask > 0)) / total_pixels * 100, 2)
    new_urban_pct  = round(float(np.sum(new_urban_mask > 0)) / total_pixels * 100, 2)
    expansion_pct  = round(float(np.sum(expansion_mask > 0)) / total_pixels * 100, 2)

    logger.info(f"Urban analysis: {old_urban_pct:.1f}% to {new_urban_pct:.1f}% "
                f"(+{expansion_pct:.1f}% expansion)")

    return {
        "expansion_mask"   : expansion_mask,
        "old_urban_pct"    : old_urban_pct,
        "new_urban_pct"    : new_urban_pct,
        "expansion_pct"    : expansion_pct,
        "net_urban_change" : round(new_urban_pct - old_urban_pct, 2)
    }


# ─────────────────────────────────────────────
# Contour Analysis
# ─────────────────────────────────────────────

def extract_change_contours(change_mask: np.ndarray,
                             min_area: int = MIN_CONTOUR_AREA) -> list:
    """
    Extract contours of changed regions from the binary change mask.
    Filters out very small contours that are likely noise.

    Parameters:
        change_mask (np.ndarray): Binary change mask.
        min_area (int): Minimum contour area threshold in pixels.

    Returns:
        list: List of OpenCV contour arrays, sorted by area (largest first).
    """
    contours, _ = cv2.findContours(
        change_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    significant = [c for c in contours if cv2.contourArea(c) >= min_area]
    significant.sort(key=cv2.contourArea, reverse=True)

    logger.info(f"Change contours: {len(significant)} significant regions "
                f"(filtered from {len(contours)} total)")
    return significant


def get_contour_statistics(contours: list,
                            image_shape: tuple) -> pd.DataFrame:
    """
    Compute statistics for each detected change contour region.

    Parameters:
        contours (list): List of OpenCV contour arrays.
        image_shape (tuple): (H, W) of the image.

    Returns:
        pd.DataFrame: Per-contour stats — area, bbox, centroid, relative size.
    """
    total_area = image_shape[0] * image_shape[1]
    records    = []

    for i, contour in enumerate(contours):
        area         = cv2.contourArea(contour)
        x, y, w, h  = cv2.boundingRect(contour)
        M            = cv2.moments(contour)
        cx = int(M['m10'] / M['m00']) if M['m00'] != 0 else x + w // 2
        cy = int(M['m01'] / M['m00']) if M['m00'] != 0 else y + h // 2

        records.append({
            "Region_ID"  : i + 1,
            "Area_px"    : int(area),
            "Area_pct"   : round(area / total_area * 100, 3),
            "Centroid_X" : cx,
            "Centroid_Y" : cy,
            "BBox_X"     : x,
            "BBox_Y"     : y,
            "BBox_W"     : w,
            "BBox_H"     : h
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
# Visualization
# ─────────────────────────────────────────────

def generate_change_highlight_image(img_old_rgb: np.ndarray,
                                     img_new_rgb: np.ndarray,
                                     forest_loss: dict,
                                     urban_expansion: dict,
                                     contours: list) -> np.ndarray:
    """
    Generate a side-by-side annotated comparison image showing:
    - Left panel:  Old image with forest loss regions highlighted in red
    - Right panel: New image with urban expansion in orange and contours drawn

    Parameters:
        img_old_rgb (np.ndarray): Old image in RGB uint8.
        img_new_rgb (np.ndarray): New image in RGB uint8.
        forest_loss (dict): Output of compute_forest_loss().
        urban_expansion (dict): Output of compute_urban_expansion().
        contours (list): Change contours from extract_change_contours().

    Returns:
        np.ndarray: Combined annotated image (RGB, uint8).
    """
    old_annotated = img_old_rgb.copy()
    new_annotated = img_new_rgb.copy()

    # Highlight forest loss on OLD image (red overlay)
    loss_mask_3ch = np.zeros_like(old_annotated)
    loss_mask_3ch[forest_loss["forest_loss_mask"] > 0] = CHANGE_COLORS["forest_loss"]
    old_annotated = cv2.addWeighted(old_annotated, 1.0, loss_mask_3ch, 0.55, 0)

    # Highlight urban expansion on NEW image (orange)
    exp_colored = np.zeros_like(new_annotated)
    exp_colored[urban_expansion["expansion_mask"] > 0] = CHANGE_COLORS["urban_gain"]
    new_annotated = cv2.addWeighted(new_annotated, 1.0, exp_colored, 0.5, 0)

    # Draw change contours on new image
    new_bgr = cv2.cvtColor(new_annotated, cv2.COLOR_RGB2BGR)
    cv2.drawContours(new_bgr, contours, -1, (0, 0, 255), thickness=2)
    new_annotated = cv2.cvtColor(new_bgr, cv2.COLOR_BGR2RGB)

    # Text annotations
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 2

    cv2.putText(old_annotated, "BEFORE (Old)",
                (10, 25), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(new_annotated, "AFTER (New)",
                (10, 25), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(old_annotated,
                f"Forest Loss: {forest_loss['forest_loss_pct']:.1f}%",
                (10, 50), font, 0.5, (255, 80, 80), 1)
    cv2.putText(new_annotated,
                f"Urban +{urban_expansion['expansion_pct']:.1f}%",
                (10, 50), font, 0.5, (255, 165, 0), 1)

    # Combine side by side with white separator
    H = max(old_annotated.shape[0], new_annotated.shape[0])
    W = old_annotated.shape[1] + new_annotated.shape[1] + 4

    combined = np.zeros((H, W, 3), dtype=np.uint8)
    combined[:old_annotated.shape[0], :old_annotated.shape[1]]           = old_annotated
    combined[:new_annotated.shape[0], old_annotated.shape[1] + 4:]       = new_annotated
    combined[:, old_annotated.shape[1]:old_annotated.shape[1] + 4]       = 255

    logger.info("Change highlight image generated.")
    return combined


def generate_difference_heatmap(change_map: np.ndarray) -> np.ndarray:
    """
    Convert the grayscale change map to a colour heatmap (COLORMAP_JET)
    for intuitive visualization of change magnitude.
    Cool colours = low change, warm colours = high change.

    Parameters:
        change_map (np.ndarray): Grayscale fused change map (uint8).

    Returns:
        np.ndarray: RGB heatmap image (uint8).
    """
    heatmap_bgr = cv2.applyColorMap(change_map, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    logger.debug("Difference heatmap generated.")
    return heatmap_rgb


# ─────────────────────────────────────────────
# Risk Classification
# ─────────────────────────────────────────────

def classify_risk_level(forest_loss_pct: float) -> dict:
    """
    Assign a risk level based on the detected forest loss percentage.

    Risk Scale:
        Low      :  0% -  10%  (Minimal deforestation, monitoring recommended)
        Medium   : 10% -  25%  (Moderate loss, intervention advised)
        High     : 25% -  50%  (Severe deforestation, urgent action needed)
        Critical : 50% - 100%  (Ecosystem collapse risk)

    Parameters:
        forest_loss_pct (float): Percentage of forest lost.

    Returns:
        dict: level, color (hex), description, recommended action.
    """
    risk_profiles = {
        "Low": {
            "color"       : "#27AE60",
            "description" : "Minimal forest change detected. Ecosystem is largely intact.",
            "action"      : "Continue regular monitoring. No immediate intervention required."
        },
        "Medium": {
            "color"       : "#F39C12",
            "description" : "Moderate deforestation detected. Some ecosystem disruption present.",
            "action"      : "Investigate cause. Consider issuing environmental alert to authorities."
        },
        "High": {
            "color"       : "#E74C3C",
            "description" : "Severe deforestation detected. Significant habitat and carbon stock loss.",
            "action"      : "Immediate environmental intervention required. Alert conservation agencies."
        },
        "Critical": {
            "color"       : "#8E0000",
            "description" : "Critical forest loss. Ecosystem collapse and biodiversity crisis risk.",
            "action"      : "Emergency response needed. Escalate to national environmental authorities."
        }
    }

    level = "Low"
    for risk_level, (low, high) in RISK_THRESHOLDS.items():
        if low <= forest_loss_pct < high:
            level = risk_level
            break

    profile = risk_profiles[level]
    profile["level"]           = level
    profile["forest_loss_pct"] = forest_loss_pct

    logger.info(f"Risk level: {level} | Forest loss: {forest_loss_pct:.1f}%")
    return profile


# ─────────────────────────────────────────────
# Master Change Detection Pipeline
# ─────────────────────────────────────────────

def run_change_detection(old_image_source,
                          new_image_source) -> dict:
    """
    Full end-to-end change detection pipeline.

    Steps:
        1. Preprocess both images
        2. Align new image to old image
        3. Compute fused change map (pixel diff + structural)
        4. Generate binary change mask
        5. Extract spectral class masks (forest, water, urban)
        6. Compute forest loss and urban expansion metrics
        7. Extract change contours
        8. Generate visualizations (highlight image, heatmap)
        9. Classify risk level
        10. Return structured results dict

    Parameters:
        old_image_source: Old satellite image (path or bytes).
        new_image_source: New satellite image (path or bytes).

    Returns:
        dict: Complete change detection results for dashboard and report.
    """
    logger.info("=" * 60)
    logger.info("STARTING CHANGE DETECTION PIPELINE")
    logger.info("=" * 60)

    # Step 1: Preprocess
    img_old_norm = preprocess_image(old_image_source, use_feature_stack=False)
    img_new_norm = preprocess_image(new_image_source, use_feature_stack=False)
    img_old_rgb  = preprocess_for_display(old_image_source)
    img_new_rgb  = preprocess_for_display(new_image_source)
    logger.info("Images preprocessed.")

    # Step 2: Align images
    img_old_aligned, img_new_aligned = align_images(img_old_norm, img_new_norm)

    # Step 3: Compute change maps
    diff_map  = compute_difference_map(img_old_aligned, img_new_aligned)
    ssim_map  = compute_structural_similarity_map(img_old_aligned, img_new_aligned)
    fused_map = fuse_change_maps(diff_map, ssim_map)
    logger.info("Change maps computed.")

    # Step 4: Binary change mask
    change_mask = generate_change_mask(fused_map)
    total_changed_pct = round(
        float(np.sum(change_mask > 0)) / change_mask.size * 100, 2
    )

    # Step 5: Spectral masks
    old_forest_mask = extract_forest_mask(img_old_rgb)
    new_forest_mask = extract_forest_mask(img_new_rgb)
    old_water_mask  = extract_water_mask(img_old_rgb)
    new_water_mask  = extract_water_mask(img_new_rgb)
    old_urban_mask  = extract_urban_mask(img_old_rgb)
    new_urban_mask  = extract_urban_mask(img_new_rgb)
    logger.info("Spectral masks extracted.")

    # Step 6: Forest loss and urban expansion
    forest_metrics = compute_forest_loss(
        old_forest_mask, new_forest_mask, change_mask
    )
    urban_metrics  = compute_urban_expansion(old_urban_mask, new_urban_mask)

    old_water_pct    = round(float(np.sum(old_water_mask > 0)) / old_water_mask.size * 100, 2)
    new_water_pct    = round(float(np.sum(new_water_mask > 0)) / new_water_mask.size * 100, 2)
    water_change_pct = round(new_water_pct - old_water_pct, 2)

    # Step 7: Contour analysis
    contours         = extract_change_contours(change_mask)
    contour_stats_df = get_contour_statistics(contours, change_mask.shape)

    # Step 8: Visualizations
    highlight_image = generate_change_highlight_image(
        img_old_rgb, img_new_rgb,
        forest_metrics, urban_metrics, contours
    )
    heatmap = generate_difference_heatmap(fused_map)

    # Step 9: Risk classification
    risk_profile = classify_risk_level(forest_metrics["forest_loss_pct"])

    # Step 10: Assemble full results dict
    results = {
        "total_changed_pct"     : total_changed_pct,
        "forest_loss_pct"       : forest_metrics["forest_loss_pct"],
        "forest_gain_pct"       : forest_metrics["forest_gain_pct"],
        "net_forest_change_pct" : forest_metrics["net_forest_change_pct"],
        "old_forest_pct"        : forest_metrics["old_forest_pct"],
        "new_forest_pct"        : forest_metrics["new_forest_pct"],
        "urban_expansion_pct"   : urban_metrics["expansion_pct"],
        "old_urban_pct"         : urban_metrics["old_urban_pct"],
        "new_urban_pct"         : urban_metrics["new_urban_pct"],
        "old_water_pct"         : old_water_pct,
        "new_water_pct"         : new_water_pct,
        "water_change_pct"      : water_change_pct,
        "risk_level"            : risk_profile["level"],
        "risk_color"            : risk_profile["color"],
        "risk_description"      : risk_profile["description"],
        "risk_action"           : risk_profile["action"],
        "change_mask"           : change_mask,
        "fused_change_map"      : fused_map,
        "forest_loss_mask"      : forest_metrics["forest_loss_mask"],
        "forest_gain_mask"      : forest_metrics["forest_gain_mask"],
        "contours"              : contours,
        "contour_stats"         : contour_stats_df,
        "num_changed_regions"   : len(contours),
        "highlight_image"       : highlight_image,
        "heatmap"               : heatmap,
        "img_old_rgb"           : img_old_rgb,
        "img_new_rgb"           : img_new_rgb
    }

    logger.info("=" * 60)
    logger.info("CHANGE DETECTION COMPLETE")
    logger.info(f"  Total change     : {total_changed_pct:.1f}%")
    logger.info(f"  Forest loss      : {forest_metrics['forest_loss_pct']:.1f}%")
    logger.info(f"  Urban expansion  : {urban_metrics['expansion_pct']:.1f}%")
    logger.info(f"  Risk level       : {risk_profile['level']}")
    logger.info("=" * 60)

    return results


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python detect_change.py <old_image> <new_image>")
        sys.exit(1)

    old_path = sys.argv[1]
    new_path = sys.argv[2]

    print(f"\n Running change detection:")
    print(f"  Old image : {old_path}")
    print(f"  New image : {new_path}\n")

    results = run_change_detection(old_path, new_path)

    print(f"\n RESULTS:")
    print(f"  Total changed area   : {results['total_changed_pct']:.1f}%")
    print(f"  Forest cover (old)   : {results['old_forest_pct']:.1f}%")
    print(f"  Forest cover (new)   : {results['new_forest_pct']:.1f}%")
    print(f"  Forest loss          : {results['forest_loss_pct']:.1f}%")
    print(f"  Forest gain          : {results['forest_gain_pct']:.1f}%")
    print(f"  Urban expansion      : {results['urban_expansion_pct']:.1f}%")
    print(f"  Water body change    : {results['water_change_pct']:+.1f}%")
    print(f"  Changed regions      : {results['num_changed_regions']}")
    print(f"  Risk level           : {results['risk_level']}")
    print(f"  Action               : {results['risk_action']}")