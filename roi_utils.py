"""
Interactive Region-of-Interest (ROI) selection utilities.
Uses matplotlib for a simple, reliable interface that works
in notebooks and plain Python scripts.
"""

from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import RectangleSelector


def select_rois_interactive(
    img: np.ndarray,
    n_rois: int = 2,
    titles: Optional[List[str]] = None,
    cmap: str = "gray",
) -> List[Tuple[int, int, int, int]]:
    """
    Interactively select one or more rectangular ROIs on an image.

    Parameters
    ----------
    img : np.ndarray
        Grayscale image.
    n_rois : int
        Number of ROIs the user should draw.
    titles : list of str, optional
        Labels shown for each ROI (e.g. ["Target / Lesion", "Background"]).
    cmap : str
        Matplotlib colormap.

    Returns
    -------
    rois : list of (x, y, w, h)
        OpenCV-style rectangles.
    """
    if titles is None:
        titles = [f"ROI {i+1}" for i in range(n_rois)]

    rois: List[Tuple[int, int, int, int]] = []

    for i in range(n_rois):
        print(f"\n>>> Draw {titles[i]} (click-drag a rectangle, then press Enter)")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img, cmap=cmap)
        ax.set_title(f"Select: {titles[i]}\n(Drag rectangle → press Enter when done)")
        ax.axis("off")

        # Keep a reference so the callback can modify it
        selected = {"roi": None}

        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            selected["roi"] = (x, y, w, h)

        selector = RectangleSelector(
            ax,
            onselect,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
        )

        plt.show(block=True)

        if selected["roi"] is None:
            raise RuntimeError(f"No ROI selected for '{titles[i]}'")
        rois.append(selected["roi"])
        print(f"   Selected {titles[i]}: {selected['roi']}")

    return rois


def select_rois_opencv(
    img: np.ndarray,
    n_rois: int = 2,
    window_name: str = "Select ROI – SPACE/ENTER to confirm, C to cancel",
) -> List[Tuple[int, int, int, int]]:
    """
    Alternative ROI selector using OpenCV's built-in selectROI.
    Useful when matplotlib interactive mode is problematic.
    """
    # OpenCV expects BGR or gray; we work with gray
    display = img.copy()
    if display.dtype != np.uint8:
        display = cv2.normalize(display, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    rois = []
    for i in range(n_rois):
        print(f"Select ROI {i+1}/{n_rois} …")
        r = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow(window_name)
        if r[2] == 0 or r[3] == 0:
            raise RuntimeError("ROI selection cancelled or empty")
        rois.append(tuple(int(v) for v in r))
        # Draw the accepted ROI so the user can see previous selections
        x, y, w, h = r
        cv2.rectangle(display, (x, y), (x + w, y + h), 255, 2)

    return rois


def draw_rois_on_image(
    img: np.ndarray,
    rois: List[Tuple[int, int, int, int]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
) -> np.ndarray:
    """
    Return a colour copy of the image with ROIs drawn on it
    (useful for saving a visual record of the analysis).
    """
    if img.ndim == 2:
        vis = cv2.cvtColor(
            cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            cv2.COLOR_GRAY2BGR,
        )
    else:
        vis = img.copy()

    if labels is None:
        labels = [f"ROI{i+1}" for i in range(len(rois))]
    if colors is None:
        # BGR colours
        colors = [(0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

    for i, (x, y, w, h) in enumerate(rois):
        c = colors[i % len(colors)]
        cv2.rectangle(vis, (x, y), (x + w, y + h), c, 2)
        cv2.putText(
            vis,
            labels[i],
            (x, max(y - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            c,
            2,
            cv2.LINE_AA,
        )
    return vis


def extract_roi_pixels(img: np.ndarray, roi: Tuple[int, int, int, int]) -> np.ndarray:
    """Return a flattened 1-D array of pixel values inside the ROI."""
    x, y, w, h = roi
    return img[y : y + h, x : x + w].ravel().astype(np.float64)