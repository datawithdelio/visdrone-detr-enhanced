"""
SkelAI preprocessing script.

Purpose:
- Reads images from an input folder (e.g., VisDrone2019-DET-train/images)
- Applies a skeletal / edge-based transformation
- Writes transformed images into an output folder (e.g., images_skel)

Processing steps:
1) Convert image to grayscale
2) Apply Gaussian blur
3) Detect edges using Canny
4) Thin edges using morphological erosion
5) Convert result to 3-channel image (required by DETR)

Expected usage:
python tools/skelai_preprocess.py --input <input_images_dir> --output <output_images_skel_dir>

Validation tip:
- Output file count should match input file count for each dataset split.
"""

from pathlib import Path
import cv2
import numpy as np


def skel_like(gray: np.ndarray) -> np.ndarray:
    """
    Apply skeletal-like preprocessing to a grayscale image.

    Steps:
    - Gaussian blur to reduce noise
    - Canny edge detection
    - Morphological erosion to thin edges
    - Convert to 3-channel image so DETR can read it
    """
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)

    # Thinning (simple morph) to make edges more "skeletal"
    kernel = np.ones((3, 3), np.uint8)
    thin = cv2.erode(edges, kernel, iterations=1)

    # Convert to 3-channel
    out = cv2.cvtColor(thin, cv2.COLOR_GRAY2BGR)
    return out


def run(input_dir: str, output_dir: str):
    """
    Run SkelAI preprocessing on all images in input_dir
    and write results to output_dir while preserving structure.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png"}
    images = [p for p in in_dir.rglob("*") if p.suffix.lower() in exts]
    if not images:
        raise SystemExit(f"No images found in {in_dir}")

    print(f"Found {len(images)} images")
    print(f"Writing to {out_dir}")

    for p in images:
        img = cv2.imread(str(p))
        if img is None:
            print("Skipping unreadable:", p)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = skel_like(gray)

        rel = p.relative_to(in_dir)
        save_path = out_dir / rel
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), out)

    print("Done.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="SkelAI image preprocessing")
    ap.add_argument("--input", required=True, help="Folder with original images")
    ap.add_argument("--output", required=True, help="Folder to write skel-like images")
    args = ap.parse_args()

    run(args.input, args.output)
