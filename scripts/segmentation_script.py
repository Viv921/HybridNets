"""
Image segmentation script for lanes and roads.

- Reads all .jpg images from the `input` directory.
- Creates `output/lanes` and `output/roads` directories if they don't exist.
- Detects light-purple lane markings and outputs a binary mask where lanes are white and everything else is black.
- Detects cyan road regions and outputs a mask where roads are grey and everything else is black.

Adjust HSV ranges below if your dataset uses slightly different colors.
Requires: opencv-python, numpy
"""

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Directories
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")
LANES_DIR = OUTPUT_DIR / "lanes"
ROADS_DIR = OUTPUT_DIR / "roads"

# Output colors (BGR)
WHITE: Tuple[int, int, int] = (255, 255, 255)
GREY: Tuple[int, int, int] = (128, 128, 128)

# HSV thresholds (OpenCV HSV: H ∈ [0,179], S ∈ [0,255], V ∈ [0,255])
# Yellow (lanes) — tune if needed
LANE_LOWER = np.array([20, 40, 120], dtype=np.uint8)
LANE_UPPER = np.array([35, 255, 255], dtype=np.uint8)

# Purple (roads) — tune if needed
ROAD_LOWER = np.array([125, 40, 120], dtype=np.uint8)
ROAD_UPPER = np.array([160, 255, 255], dtype=np.uint8)

# Morphology kernel
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def ensure_dirs() -> None:
    LANES_DIR.mkdir(parents=True, exist_ok=True)
    ROADS_DIR.mkdir(parents=True, exist_ok=True)


def make_mask(hsv_img: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Create a cleaned binary mask for the HSV range."""
    raw = cv2.inRange(hsv_img, lower, upper)
    # Clean small speckles; adjust iterations if needed
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, KERNEL, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, KERNEL, iterations=1)
    return closed


def save_mask_as_color(mask: np.ndarray, color: Tuple[int, int, int]) -> np.ndarray:
    """Return a BGR image where mask==255 is painted with color; else black."""
    h, w = mask.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[mask > 0] = color
    return out


def process_image(img_path: Path) -> None:
    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: Could not read image: {img_path}")
        return

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Lanes (light purple -> white)
    lane_mask = make_mask(hsv, LANE_LOWER, LANE_UPPER)
    lane_out = save_mask_as_color(lane_mask, WHITE)

    # Roads (cyan -> grey)
    road_mask = make_mask(hsv, ROAD_LOWER, ROAD_UPPER)
    road_out = save_mask_as_color(road_mask, GREY)

    lanes_out_path = LANES_DIR / img_path.name
    roads_out_path = ROADS_DIR / img_path.name

    cv2.imwrite(str(lanes_out_path), lane_out)
    cv2.imwrite(str(roads_out_path), road_out)


def main() -> int:
    ensure_dirs()

    if not INPUT_DIR.exists():
        print(f"Error: Input directory not found: {INPUT_DIR}")
        return 1

    images = sorted([p for p in INPUT_DIR.glob("*.jpg")])
    if not images:
        print(f"No .jpg images found in {INPUT_DIR}")
        return 0

    for img_path in images:
        process_image(img_path)

    print(f"Done. Wrote lanes to '{LANES_DIR}' and roads to '{ROADS_DIR}'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

