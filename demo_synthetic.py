#!/usr/bin/env python3
"""
Quick self-test / demonstration using a synthetic ultrasound-like image.
Generates a phantom-style image with a dark 'cyst' and brighter background,
then runs the full analysis pipeline without any user interaction.

This lets you verify that the installation works and see example numbers
before you start analysing real clinical scans.
"""

from pathlib import Path

import cv2
import numpy as np

from iq_metrics import compute_all_metrics
from utils import overall_quality_score, print_metrics
from visualization import plot_image_with_rois, plot_metric_bars, plot_histograms
from roi_utils import extract_roi_pixels


def make_synthetic_ultrasound(
    size: int = 400,
    cyst_center: tuple = (180, 160),
    cyst_radius: int = 45,
    noise_level: float = 18.0,
) -> np.ndarray:
    """
    Create a simple synthetic B-mode-like image:
    - brighter background tissue with speckle
    - darker circular 'cyst'
    - mild intensity gradient (simulates depth attenuation)
    """
    yy, xx = np.mgrid[0:size, 0:size]
    # Depth-dependent mean intensity (darker at bottom)
    base = 140 - 0.25 * yy

    # Speckle (multiplicative-ish)
    speckle = np.random.rayleigh(noise_level, size=(size, size))
    img = base + speckle

    # Dark cyst
    dist = np.sqrt((xx - cyst_center[0]) ** 2 + (yy - cyst_center[1]) ** 2)
    cyst_mask = dist < cyst_radius
    img[cyst_mask] = img[cyst_mask] * 0.25 + 25

    # Mild Gaussian blur to mimic limited resolution
    img = cv2.GaussianBlur(img.astype(np.float32), (5, 5), 0.8)

    # Clip to 8-bit range
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def main():
    print("Generating synthetic ultrasound-like image …")
    img = make_synthetic_ultrasound()

    # Fixed ROIs (no interaction needed)
    # Target = inside the cyst, Background = nearby tissue
    target_roi = (145, 125, 70, 70)       # x, y, w, h
    background_roi = (260, 100, 80, 80)

    print("Computing metrics …")
    metrics = compute_all_metrics(img, target_roi, background_roi)
    metrics["overall_quality_score"] = overall_quality_score(metrics)
    print_metrics(metrics, title="Synthetic Image – Quality Metrics")

    out_dir = Path("demo_output")
    out_dir.mkdir(exist_ok=True)

    plot_image_with_rois(
        img,
        [target_roi, background_roi],
        labels=["Cyst (Target)", "Tissue (Background)"],
        title="Synthetic US – Selected ROIs",
        save_path=out_dir / "synthetic_rois.png",
        show=False,
    )

    t_pix = extract_roi_pixels(img, target_roi)
    b_pix = extract_roi_pixels(img, background_roi)
    plot_histograms(
        t_pix,
        b_pix,
        title="Synthetic US – Intensity Histograms",
        save_path=out_dir / "synthetic_hist.png",
        show=False,
    )

    plot_metric_bars(
        metrics,
        title="Synthetic US – Key Metrics",
        save_path=out_dir / "synthetic_bars.png",
        show=False,
    )

    print(f"\nDemo figures saved in: {out_dir.resolve()}")
    print("You can now open those PNGs or run the same pipeline on real clinical images.")


if __name__ == "__main__":
    main()