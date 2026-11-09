"""
Plotting and simple report helpers for ultrasound image quality results.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from roi_utils import draw_rois_on_image


def plot_image_with_rois(
    img: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    title: str = "Selected ROIs",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """Display (and optionally save) the image with drawn ROIs."""
    vis = draw_rois_on_image(img, rois, labels=labels)
    # Convert BGR → RGB for matplotlib
    vis_rgb = vis[:, :, ::-1]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(vis_rgb)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved ROI visualisation → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_metric_bars(
    metrics: Dict[str, float],
    keys: Optional[List[str]] = None,
    title: str = "Key Image Quality Metrics",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """Simple bar chart of selected metrics (useful for the thesis)."""
    if keys is None:
        keys = [
            "sharpness_laplacian",
            "contrast_rms",
            "snr",
            "cnr",
            "gcnr",
            "uniformity_cov",
        ]

    present = [k for k in keys if k in metrics]
    values = [metrics[k] for k in present]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(present)), values, color="steelblue", edgecolor="k")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels(present, rotation=30, ha="right")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved metric bar chart → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_histograms(
    target: np.ndarray,
    background: np.ndarray,
    title: str = "Intensity Histograms – Target vs Background",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> None:
    """Side-by-side histograms of the two ROIs (helps interpret CNR / gCNR)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(target.ravel(), bins=64, alpha=0.6, label="Target", density=True, color="C0")
    ax.hist(
        background.ravel(),
        bins=64,
        alpha=0.6,
        label="Background",
        density=True,
        color="C1",
    )
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved histogram plot → {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def save_results_csv(
    metrics: Dict[str, float],
    image_name: str,
    csv_path: Union[str, Path],
) -> None:
    """Append a single image’s results to a CSV file (creates header if needed)."""
    import pandas as pd

    csv_path = Path(csv_path)
    row = {"image": image_name, **metrics}
    df = pd.DataFrame([row])

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)
    print(f"Results written → {csv_path}")