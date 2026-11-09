"""
Utility functions for ultrasound image quality assessment.
Handles loading, basic preprocessing and overall quality scoring.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def load_image(
    path: Union[str, Path],
    as_gray: bool = True,
    normalize: bool = False,
) -> np.ndarray:
    """
    Load an ultrasound image from disk.

    Parameters
    ----------
    path : str or Path
        Path to the image file (PNG, JPG, TIFF, BMP…).
    as_gray : boolean
        Convert to single-channel grayscale if True.
    normalize : boolean
        Scale intensities to [0, 1] float32 if True.
        Leave False (default) to keep original 8/16-bit range –
        this is usually preferred for SNR/CNR calculations.

    Returns
    -------
    img : np.ndarray
        2-D array (H, W). dtype is float32 when normalize=True,
        otherwise the original dtype (usually uint8 or uint16).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    # OpenCV handles most common medical image formats well
    flag = cv2.IMREAD_GRAYSCALE if as_gray else cv2.IMREAD_UNCHANGED
    img = cv2.imread(str(path), flag)

    if img is None:
        # Fallback to Pillow (handles some exotic TIFF variants)
        pil_img = Image.open(path)
        if as_gray:
            pil_img = pil_img.convert("L")
        img = np.array(pil_img)

    if img.ndim == 3 and as_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if normalize:
        img = img.astype(np.float32)
        if img.max() > 0:
            img = (img - img.min()) / (img.max() - img.min())

    return img


def ensure_float64(img: np.ndarray) -> np.ndarray:
    """Convert image to float64 for numerical stability."""
    return img.astype(np.float64)


def crop_to_roi(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Crop a rectangular ROI from an image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    roi : (x, y, w, h)
        OpenCV-style rectangle (top-left x, top-left y, width, height).

    Returns
    -------
    cropped : np.ndarray
    """
    x, y, w, h = roi
    return img[y : y + h, x : x + w]


def overall_quality_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute a simple weighted overall quality score (0–100).

    Higher is better.  The function normalises each metric into
    a roughly [0, 1] range using literature-inspired heuristics
    and then applies the supplied weights.

    This is intentionally transparent so you can defend the
    scoring scheme in the thesis.  Adjust the normalisation
    constants after you have looked at a few real clinical
    images from your dataset.

    Parameters
    ----------
    metrics : dict
        Dictionary returned by the analysis pipeline.
    weights : dict, optional
        Relative importance of each component.
        Defaults favour clinical detectability (CNR / contrast)
        slightly over pure sharpness.

    Returns
    -------
    score : float
        Overall quality score in [0, 100].
    """
    if weights is None:
        weights = {
            "sharpness": 0.25,
            "contrast": 0.20,
            "snr": 0.20,
            "cnr": 0.25,
            "uniformity": 0.10,
        }

    # --- Heuristic normalisation to ~[0, 1] ---
    # These constants should be refined once you have real data.
    norm = {}

    # Laplacian variance / Tenengrad – higher is sharper
    # Typical good US images often sit in the 50–400 range
    # depending on resolution and preprocessing.
    sharp = metrics.get("sharpness_laplacian", 0.0)
    norm["sharpness"] = np.clip(sharp / 300.0, 0.0, 1.0)

    # RMS contrast or Michelson – higher is better
    contrast = metrics.get("contrast_rms", 0.0)
    norm["contrast"] = np.clip(contrast / 0.25, 0.0, 1.0)

    # SNR – higher is better.  Good homogeneous tissue often > 5–10
    snr = metrics.get("snr", 0.0)
    norm["snr"] = np.clip(snr / 15.0, 0.0, 1.0)

    # CNR – higher is better.  Clinically useful lesions often > 1–3
    cnr = metrics.get("cnr", 0.0)
    norm["cnr"] = np.clip(cnr / 4.0, 0.0, 1.0)

    # Uniformity – we store coefficient of variation (lower is better)
    # so we invert it.
    unif_cov = metrics.get("uniformity_cov", 1.0)
    norm["uniformity"] = np.clip(1.0 - unif_cov / 0.30, 0.0, 1.0)

    # Weighted sum
    total_w = sum(weights.values())
    score = 0.0
    for key, w in weights.items():
        score += w * norm.get(key, 0.0)

    return float(100.0 * score / total_w)


def print_metrics(metrics: Dict[str, float], title: str = "Image Quality Metrics") -> None:
    """Pretty-print a metrics dictionary."""
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:10.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print("=" * 60 + "\n")