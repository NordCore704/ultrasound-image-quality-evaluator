"""
Core image-quality metrics for diagnostic ultrasound.

All functions accept a 2-D grayscale image (or an already-extracted
ROI) and return a scalar or a small dictionary of related values.
They are deliberately kept independent so each can be unit-tested
and explained in the Methods chapter.
"""

from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.measure import shannon_entropy


# ---------------------------------------------------------------------------
# 1. Sharpness / Spatial-resolution proxies
# ---------------------------------------------------------------------------

def sharpness_laplacian(img: np.ndarray) -> float:
    """
    Variance of the Laplacian (classic focus / sharpness measure).

    Higher values → sharper image.
    Very robust and commonly used as a no-reference sharpness metric.
    """
    img = img.astype(np.float64)
    lap = cv2.Laplacian(img, cv2.CV_64F, ksize=3)
    return float(lap.var())


def sharpness_tenengrad(img: np.ndarray, ksize: int = 3) -> float:
    """
    Tenengrad (sum of squared Sobel gradient magnitudes).

    Higher values → sharper image.
    Slightly more sensitive to directional edges than the Laplacian Variance.
    """
    img = img.astype(np.float64)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    return float(np.sum(gx ** 2 + gy ** 2))


def sharpness_brenner(img: np.ndarray) -> float:
    """
    Brenner gradient (difference of pixels two steps apart).

    """
    img = img.astype(np.float64)
    diff = img[:, 2:] - img[:, :-2]
    return float(np.sum(diff ** 2))


# ---------------------------------------------------------------------------
# 2. Contrast
# ---------------------------------------------------------------------------

def contrast_rms(img: np.ndarray) -> float:
    """
    Root-mean-square (RMS) contrast.

    Standard deviation of the normalised intensities.
    Requires the image (or ROI) to be scaled to [0, 1] for
    comparability across different bit-depths; we do that internally.
    """
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-12)
    return float(np.std(img))


def contrast_michelson(img: np.ndarray) -> float:
    """
    Michelson contrast: (Imax – Imin) / (Imax + Imin).

    Useful for high-contrast structures; and less informative on
    low-contrast tissue.
    """
    img = img.astype(np.float64)
    i_max, i_min = img.max(), img.min()
    denom = i_max + i_min
    if denom < 1e-12:
        return 0.0
    return float((i_max - i_min) / denom)


def contrast_weber(target: np.ndarray, background: np.ndarray) -> float:
    """
    Weber contrast: (μ_target – μ_background) / μ_background.
    """
    mu_t = float(np.mean(target))
    mu_b = float(np.mean(background))
    if abs(mu_b) < 1e-12:
        return 0.0
    return (mu_t - mu_b) / mu_b


# ---------------------------------------------------------------------------
# 3. Noise metrics – SNR & CNR
# ---------------------------------------------------------------------------

def snr_simple(roi: np.ndarray) -> float:
    """
    Simple SNR = mean / standard deviation inside a homogeneous ROI.

    Commonly used for tissue or background regions.
    """
    roi = roi.astype(np.float64)
    mu = np.mean(roi)
    sigma = np.std(roi)
    if sigma < 1e-12:
        return float("inf")
    return float(mu / sigma)


def cnr(
    target: np.ndarray,
    background: np.ndarray,
    method: str = "classic",
) -> float:
    """
    Contrast-to-Noise Ratio.

    Parameters
    ----------
    target, background : np.ndarray
        Pixel values of the two ROIs.
    method : {"classic", "alternative"}
        - "classic"      : |μt – μb| / sqrt(σt² + σb²)
        - "alternative"  : |μt – μb| / σb        

    Returns
    -------
    cnr : float
    """
    target = target.astype(np.float64)
    background = background.astype(np.float64)

    mu_t = np.mean(target)
    mu_b = np.mean(background)
    sig_t = np.std(target)
    sig_b = np.std(background)

    num = abs(mu_t - mu_b)

    if method == "classic":
        denom = np.sqrt(sig_t ** 2 + sig_b ** 2)
    elif method == "alternative":
        denom = sig_b
    else:
        raise ValueError("method must be 'classic' or 'alternative'")

    if denom < 1e-12:
        return float("inf")
    return float(num / denom)


def gcnr(
    target: np.ndarray,
    background: np.ndarray,
    bins: int = 256,
) -> float:
    """
    Generalized Contrast-to-Noise Ratio (Rodriguez-Molares et al.).

    gCNR = 1 – overlap of the two intensity histograms.
    Robust to dynamic-range transformations and preferred in
    modern ultrasound literature.
    """
    target = target.astype(np.float64).ravel()
    background = background.astype(np.float64).ravel()

    # Common bin edges
    all_vals = np.concatenate([target, background])
    hist_range = (all_vals.min(), all_vals.max())
    if hist_range[1] <= hist_range[0]:
        return 0.0

    hist_t, _ = np.histogram(target, bins=bins, range=hist_range, density=True)
    hist_b, _ = np.histogram(background, bins=bins, range=hist_range, density=True)

    # Normalise to probability mass
    hist_t = hist_t / (hist_t.sum() + 1e-12)
    hist_b = hist_b / (hist_b.sum() + 1e-12)

    overlap = np.minimum(hist_t, hist_b).sum()
    return float(1.0 - overlap)


# ---------------------------------------------------------------------------
# 4. Uniformity / brightness balance
# ---------------------------------------------------------------------------

def uniformity_coefficient_of_variation(
    img: np.ndarray,
    n_blocks: int = 4,
) -> Dict[str, float]:
    """
    Split the image (or a large ROI) into an n_blocks × n_blocks grid
    and compute the coefficient of variation of the block means.

    Lower CoV → more uniform brightness.
    Also returns the min/max block mean for extra insight.
    """
    img = img.astype(np.float64)
    h, w = img.shape
    bh, bw = h // n_blocks, w // n_blocks

    means = []
    for i in range(n_blocks):
        for j in range(n_blocks):
            block = img[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            if block.size > 0:
                means.append(np.mean(block))

    means = np.array(means)
    global_mean = np.mean(means)
    if abs(global_mean) < 1e-12:
        cov = 0.0
    else:
        cov = float(np.std(means) / global_mean)

    return {
        "uniformity_cov": cov,
        "uniformity_block_means_std": float(np.std(means)),
        "uniformity_min_block_mean": float(means.min()),
        "uniformity_max_block_mean": float(means.max()),
    }


def horizontal_profile_uniformity(img: np.ndarray) -> Dict[str, float]:
    """
    Compute mean intensity along vertical lines (horizontal profile)
    and derive simple uniformity descriptors.
    Useful for detecting drop-out or element failure artefacts.
    """
    img = img.astype(np.float64)
    profile = np.mean(img, axis=0)  # average over rows → 1-D horizontal profile
    mu = np.mean(profile)
    if abs(mu) < 1e-12:
        cov = 0.0
    else:
        cov = float(np.std(profile) / mu)

    return {
        "horiz_profile_cov": cov,
        "horiz_profile_range": float(profile.max() - profile.min()),
    }


# ---------------------------------------------------------------------------
# 5. Simple artefact / noise indicators
# ---------------------------------------------------------------------------

def local_variance_map(
    img: np.ndarray,
    win_size: int = 15,
) -> np.ndarray:
    """
    Compute a sliding-window local variance map.
    High local variance can indicate strong speckle or certain artefacts.
    """
    img = img.astype(np.float64)
    mu = ndimage.uniform_filter(img, size=win_size)
    mu_sq = ndimage.uniform_filter(img ** 2, size=win_size)
    return mu_sq - mu ** 2


def artefact_indicator_high_variance(
    img: np.ndarray,
    win_size: int = 15,
    percentile: float = 99.0,
) -> Dict[str, float]:
    """
    Simple numeric flag for unusually high local variance regions
    (possible strong artefacts or extreme speckle).

    Returns the chosen high percentile of the local-variance map
    and the fraction of pixels that exceed a robust threshold.
    """
    var_map = local_variance_map(img, win_size=win_size)
    high_val = float(np.percentile(var_map, percentile))
    # Robust threshold: median + 3 * MAD
    med = np.median(var_map)
    mad = np.median(np.abs(var_map - med))
    thresh = med + 3.0 * 1.4826 * mad
    fraction = float(np.mean(var_map > thresh))

    return {
        "local_var_high_percentile": high_val,
        "local_var_outlier_fraction": fraction,
    }


def entropy(img: np.ndarray) -> float:
    """Shannon entropy of the intensity histogram (information content)."""
    return float(shannon_entropy(img))


# ---------------------------------------------------------------------------
# Convenience wrapper that computes a full set of metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    img: np.ndarray,
    target_roi: Optional[Tuple[int, int, int, int]] = None,
    background_roi: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of quality metrics.

    If target_roi and background_roi are supplied, SNR/CNR/gCNR
    are calculated from those regions; otherwise only global
    metrics are returned.
    """
    from utils import crop_to_roi  # local import to avoid circularity

    results: Dict[str, float] = {}

    # Global sharpness
    results["sharpness_laplacian"] = sharpness_laplacian(img)
    results["sharpness_tenengrad"] = sharpness_tenengrad(img)
    results["sharpness_brenner"] = sharpness_brenner(img)

    # Global contrast & entropy
    results["contrast_rms"] = contrast_rms(img)
    results["contrast_michelson"] = contrast_michelson(img)
    results["entropy"] = entropy(img)

    # Uniformity
    results.update(uniformity_coefficient_of_variation(img, n_blocks=4))
    results.update(horizontal_profile_uniformity(img))

    # Simple artefact indicators
    results.update(artefact_indicator_high_variance(img))

    # ROI-based metrics
    if target_roi is not None and background_roi is not None:
        target = crop_to_roi(img, target_roi)
        background = crop_to_roi(img, background_roi)

        results["snr_target"] = snr_simple(target)
        results["snr_background"] = snr_simple(background)
        results["snr"] = results["snr_background"]  # conventional choice

        results["cnr"] = cnr(target, background, method="classic")
        results["cnr_alt"] = cnr(target, background, method="alternative")
        results["gcnr"] = gcnr(target, background)
        results["contrast_weber"] = contrast_weber(target, background)

        results["mean_target"] = float(np.mean(target))
        results["mean_background"] = float(np.mean(background))
        results["std_target"] = float(np.std(target))
        results["std_background"] = float(np.std(background))

    return results