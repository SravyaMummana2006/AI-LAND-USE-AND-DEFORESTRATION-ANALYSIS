"""
download_dataset.py
-------------------
Automated Dataset Downloader & Organiser for Land Use Analysis.

This script downloads a SMALL, efficient subset of the EuroSAT dataset
and organises it into the exact folder structure required by train_model.py.

Total size:  ~6–10 MB  (500 images, 100 per class)
Image size:  64x64 pixels (upscaled to 256x256 during training)
Classes:     Forest, Water, Urban, Agriculture, Barren

No Kaggle account needed — downloads directly from public URLs.

Usage:
    python download_dataset.py

Author: AI Land Use Analysis System
"""

import os
import urllib.request
import zipfile
import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

IMAGES_PER_CLASS  = 100          # 100 per class × 5 classes = 500 total images
DATASET_DIR       = "data"       # Output folder expected by train_model.py
RANDOM_SEED       = 42

# EuroSAT RGB — publicly available via Zenodo (no login needed)
# Direct download link for the full RGB zip (~90MB) — we extract only what we need
EUROSAT_URL = "https://madm.dfki.de/files/sentinel/EuroSAT.zip"
EUROSAT_ZIP = "EuroSAT.zip"
EUROSAT_EXTRACTED = "EuroSAT"

# EuroSAT class → our project class mapping
# EuroSAT has 10 classes; we map them to our 5
CLASS_MAPPING = {
    # EuroSAT folder name   →   Our project class
    "Forest"                : "Forest",
    "River"                 : "Water",
    "SeaLake"               : "Water",          # merged into Water
    "Residential"           : "Urban",
    "Industrial"            : "Urban",           # merged into Urban
    "AnnualCrop"            : "Agriculture",
    "PermanentCrop"         : "Agriculture",     # merged into Agriculture
    "HerbaceousVegetation"  : "Barren",
    "Pasture"               : "Barren",          # merged into Barren
    "Highway"               : "Urban",           # roads → Urban
}

# Target class list
TARGET_CLASSES = ["Forest", "Water", "Urban", "Agriculture", "Barren"]


# ─────────────────────────────────────────────
# Download with Progress
# ─────────────────────────────────────────────

def download_with_progress(url: str, dest: str):
    """
    Download a file from a URL with a simple CLI progress indicator.

    Parameters:
        url  (str): Source URL.
        dest (str): Local destination file path.
    """
    print(f"\n Downloading: {url}")
    print(f" Saving to  : {dest}")

    def reporthook(count, block_size, total_size):
        if total_size > 0:
            pct = min(int(count * block_size * 100 / total_size), 100)
            bar = "█" * (pct // 2) + "░" * (50 - pct // 2)
            print(f"\r  [{bar}] {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook)
    print(f"\n Download complete: {dest}")


# ─────────────────────────────────────────────
# Dataset Organisation
# ─────────────────────────────────────────────

def organise_dataset(extracted_dir: str,
                     output_dir: str,
                     images_per_class: int = IMAGES_PER_CLASS):
    """
    Organise extracted EuroSAT images into the project folder structure.
    Maps EuroSAT's 10 classes to our 5 project classes and copies a
    random subset of images per class.

    Parameters:
        extracted_dir   (str): Path to extracted EuroSAT folder.
        output_dir      (str): Destination data/ folder.
        images_per_class(int): How many images to copy per target class.
    """
    random.seed(RANDOM_SEED)

    # Accumulate source images per target class
    class_image_pool = {cls: [] for cls in TARGET_CLASSES}

    eurosat_path = Path(extracted_dir)

    for eurosat_cls, project_cls in CLASS_MAPPING.items():
        src_folder = eurosat_path / eurosat_cls
        if not src_folder.exists():
            print(f"  Warning: {src_folder} not found, skipping.")
            continue

        images = list(src_folder.glob("*.jpg")) + \
                 list(src_folder.glob("*.png")) + \
                 list(src_folder.glob("*.tif"))

        class_image_pool[project_cls].extend(images)
        print(f"  Found {len(images):4d} images in '{eurosat_cls}' → '{project_cls}'")

    # Copy selected subset to output folders
    print(f"\n Copying {images_per_class} images per class to '{output_dir}/'...")

    total_copied = 0
    for target_cls in TARGET_CLASSES:
        dest_folder = Path(output_dir) / target_cls
        dest_folder.mkdir(parents=True, exist_ok=True)

        pool = class_image_pool[target_cls]
        if not pool:
            print(f"  ⚠ No images found for class: {target_cls}")
            continue

        # Shuffle and select subset
        random.shuffle(pool)
        selected = pool[:images_per_class]

        for i, src_path in enumerate(selected):
            dest_path = dest_folder / f"{target_cls}_{i:04d}{src_path.suffix}"
            shutil.copy2(src_path, dest_path)

        print(f"  ✓ {target_cls:12s}: {len(selected):3d} images copied")
        total_copied += len(selected)

    print(f"\n Total images in dataset: {total_copied}")


def verify_dataset(data_dir: str):
    """
    Verify the dataset folder structure and print a summary.

    Parameters:
        data_dir (str): Path to the data/ folder.
    """
    print(f"\n Dataset Verification — '{data_dir}/'")
    print("─" * 40)

    total = 0
    all_ok = True

    for cls in TARGET_CLASSES:
        cls_path = Path(data_dir) / cls
        if not cls_path.exists():
            print(f"  ✗ {cls:12s}: MISSING")
            all_ok = False
            continue

        count = len(list(cls_path.glob("*.*")))
        status = "✓" if count >= 50 else "⚠ (low)"
        print(f"  {status} {cls:12s}: {count} images")
        total += count

    print("─" * 40)
    print(f"  Total           : {total} images")
    print(f"  Est. disk usage : ~{total * 5 // 1000} MB")
    print(f"  Status          : {'READY ✓' if all_ok else 'INCOMPLETE ✗'}")

    return all_ok


def cleanup_temp_files():
    """Remove the downloaded zip and extracted folder to save disk space."""
    print("\n Cleaning up temporary files...")

    if Path(EUROSAT_ZIP).exists():
        os.remove(EUROSAT_ZIP)
        print(f"  Removed: {EUROSAT_ZIP}")

    if Path(EUROSAT_EXTRACTED).exists():
        shutil.rmtree(EUROSAT_EXTRACTED)
        print(f"  Removed: {EUROSAT_EXTRACTED}/")


# ─────────────────────────────────────────────
# Alternative: Generate Synthetic Dataset
# (if download fails or no internet)
# ─────────────────────────────────────────────

def generate_synthetic_dataset(output_dir: str,
                                images_per_class: int = IMAGES_PER_CLASS):
    """
    Generate a synthetic colour-based dataset as a fallback when
    internet download is not available.

    Each class is represented by realistic HSV colour ranges
    that mimic real satellite image appearances:
        Forest      → Dark green tones
        Water       → Blue tones
        Urban       → Grey tones with texture
        Agriculture → Yellow-green tones
        Barren      → Brown/tan tones

    Images are 64x64 pixels with added Gaussian noise for realism.

    Parameters:
        output_dir      (str): Destination data/ folder.
        images_per_class(int): Number of synthetic images per class.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("OpenCV not found. Install with: pip install opencv-python")
        return False

    print(f"\n Generating synthetic dataset ({images_per_class} images/class)...")

    # HSV base colours and variance per class
    class_profiles = {
        "Forest": {
            "base_bgr" : (34,  100, 34),
            "noise"    : 25,
            "texture"  : True
        },
        "Water": {
            "base_bgr" : (180, 100, 50),
            "noise"    : 15,
            "texture"  : False
        },
        "Urban": {
            "base_bgr" : (130, 130, 130),
            "noise"    : 40,
            "texture"  : True
        },
        "Agriculture": {
            "base_bgr" : (60,  180, 120),
            "noise"    : 30,
            "texture"  : True
        },
        "Barren": {
            "base_bgr" : (100, 150, 180),
            "noise"    : 35,
            "texture"  : False
        }
    }

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    total = 0
    for cls, profile in class_profiles.items():
        dest_folder = Path(output_dir) / cls
        dest_folder.mkdir(parents=True, exist_ok=True)

        for i in range(images_per_class):
            # Base colour image
            img = np.full((64, 64, 3), profile["base_bgr"], dtype=np.uint8)

            # Add Gaussian noise
            noise = np.random.normal(0, profile["noise"], img.shape).astype(np.int16)
            img   = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            # Add texture pattern (for forest, urban, agriculture)
            if profile["texture"]:
                for _ in range(random.randint(5, 15)):
                    x1, y1 = random.randint(0, 63), random.randint(0, 63)
                    x2, y2 = random.randint(0, 63), random.randint(0, 63)
                    color  = tuple(
                        int(c + random.randint(-30, 30))
                        for c in profile["base_bgr"]
                    )
                    cv2.line(img, (x1, y1), (x2, y2), color, 1)

            # Save image
            out_path = dest_folder / f"{cls}_{i:04d}.jpg"
            cv2.imwrite(str(out_path), img)
            total += 1

        print(f"  ✓ {cls:12s}: {images_per_class} synthetic images generated")

    print(f"\n Synthetic dataset ready | Total: {total} images")
    return True


# ─────────────────────────────────────────────
# Main Entry Point
# ─────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Land Use Dataset Downloader & Organiser")
    print("  Target: ~500 images | ~6MB | 5 classes")
    print("=" * 55)

    print("\n Choose dataset source:")
    print("  1 → Download EuroSAT RGB (real satellite, ~90MB download → ~6MB kept)")
    print("  2 → Generate synthetic dataset (no internet needed, ~2MB)")

    choice = input("\nEnter choice (1 or 2) [default=2]: ").strip() or "2"

    if choice == "1":
        # ── Real EuroSAT Download ──
        try:
            # Download
            download_with_progress(EUROSAT_URL, EUROSAT_ZIP)

            # Extract
            print(f"\n Extracting {EUROSAT_ZIP}...")
            with zipfile.ZipFile(EUROSAT_ZIP, 'r') as zf:
                zf.extractall(".")
            print(" Extraction complete.")

            # Organise into project structure
            organise_dataset(EUROSAT_EXTRACTED, DATASET_DIR, IMAGES_PER_CLASS)

            # Clean up large temp files
            cleanup_temp_files()

        except Exception as e:
            print(f"\n Download failed: {e}")
            print(" Falling back to synthetic dataset generation...")
            generate_synthetic_dataset(DATASET_DIR, IMAGES_PER_CLASS)

    else:
        # ── Synthetic Dataset ──
        generate_synthetic_dataset(DATASET_DIR, IMAGES_PER_CLASS)

    # Verify
    ready = verify_dataset(DATASET_DIR)

    if ready:
        print("\n" + "=" * 55)
        print("  Dataset ready! Next step:")
        print()
        print("  python train_model.py --data_dir ./data --epochs 30")
        print("=" * 55)
    else:
        print("\n Dataset setup incomplete. Please check errors above.")


if __name__ == "__main__":
    main()