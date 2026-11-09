#!/usr/bin/env python3
"""
Ultrasound Image Quality Assessment – main entry point.

Usage examples
--------------
# Interactive analysis of a single image
python main.py --image /path/to/scan.png --interactive

# Non-interactive (supply ROIs as x,y,w,h)
python main.py --image /path/to/scan.png \
               --target 120,80,60,50 \
               --background 300,200,80,60

# Batch mode on a folder (ROIs must be supplied via a CSV or run interactively per image)
python main.py --folder /path/to/images/ --interactive --output results.csv

The script produces:
  - printed metrics table
  - optional ROI visualisation PNG
  - optional bar-chart PNG
  - optional CSV row for later statistical analysis
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Local imports
from iq_metrics import compute_all_metrics
from roi_utils import (
    select_rois_interactive,
    select_rois_opencv,
    extract_roi_pixels,
)
from utils import load_image, overall_quality_score, print_metrics
from visualization import (
    plot_image_with_rois,
    plot_metric_bars,
    plot_histograms,
    save_results_csv,
)


def parse_roi(s: str) -> Tuple[int, int, int, int]:
    """Parse 'x,y,w,h' string into a tuple of ints."""
    parts = [int(v.strip()) for v in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("ROI must be x,y,w,h")
    return tuple(parts)  # type: ignore


def analyse_one_image(
    image_path: Path,
    target_roi: Optional[Tuple[int, int, int, int]] = None,
    background_roi: Optional[Tuple[int, int, int, int]] = None,
    interactive: bool = False,
    use_opencv_roi: bool = False,
    save_dir: Optional[Path] = None,
    show_plots: bool = True,
) -> dict:
    """
    Full analysis pipeline for a single ultrasound image.
    """
    print(f"\n{'='*60}")
    print(f"Analysing: {image_path.name}")
    print(f"{'='*60}")

    img = load_image(image_path, as_gray=True, normalize=False)
    print(f"Image shape: {img.shape}, dtype: {img.dtype}, "
          f"range: [{img.min():.1f}, {img.max():.1f}]")

    # ------------------------------------------------------------------
    # ROI selection
    # ------------------------------------------------------------------
    if interactive and (target_roi is None or background_roi is None):
        print("\nInteractive ROI selection required.")
        print("  1st ROI → Target / structure of interest")
        print("  2nd ROI → Background / surrounding tissue")
        if use_opencv_roi:
            rois = select_rois_opencv(img, n_rois=2)
        else:
            rois = select_rois_interactive(
                img,
                n_rois=2,
                titles=["Target / Lesion / Structure", "Background tissue"],
            )
        target_roi, background_roi = rois[0], rois[1]

    if target_roi is None or background_roi is None:
        print("WARNING: No ROIs supplied – only global metrics will be computed.")
        metrics = compute_all_metrics(img)
    else:
        metrics = compute_all_metrics(img, target_roi, background_roi)

    # Overall score
    metrics["overall_quality_score"] = overall_quality_score(metrics)

    print_metrics(metrics, title=f"Results for {image_path.name}")

    # ------------------------------------------------------------------
    # Visualisations
    # ------------------------------------------------------------------
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        stem = image_path.stem

        if target_roi is not None and background_roi is not None:
            plot_image_with_rois(
                img,
                [target_roi, background_roi],
                labels=["Target", "Background"],
                title=f"ROIs – {image_path.name}",
                save_path=save_dir / f"{stem}_rois.png",
                show=show_plots,
            )

            # Histograms
            t_pix = extract_roi_pixels(img, target_roi)
            b_pix = extract_roi_pixels(img, background_roi)
            plot_histograms(
                t_pix,
                b_pix,
                title=f"Histograms – {image_path.name}",
                save_path=save_dir / f"{stem}_hist.png",
                show=show_plots,
            )

        plot_metric_bars(
            metrics,
            title=f"Key Metrics – {image_path.name}",
            save_path=save_dir / f"{stem}_bars.png",
            show=show_plots,
        )

    return metrics


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Objective ultrasound image quality assessment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Folder containing images")
    parser.add_argument(
        "--target",
        type=parse_roi,
        help="Target ROI as x,y,w,h (pixels)",
    )
    parser.add_argument(
        "--background",
        type=parse_roi,
        help="Background ROI as x,y,w,h (pixels)",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Select ROIs interactively with the mouse",
    )
    parser.add_argument(
        "--opencv-roi",
        action="store_true",
        help="Use OpenCV selectROI instead of matplotlib",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ultrasound_iq_results.csv",
        help="CSV file to append results to",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="iq_output",
        help="Directory for PNG visualisations",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots (only save them)",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default="png,jpg,jpeg,tif,tiff,bmp",
        help="Comma-separated list of image extensions for folder mode",
    )

    args = parser.parse_args(argv)

    if not args.image and not args.folder:
        parser.error("Provide either --image or --folder")

    save_dir = Path(args.save_dir)
    show_plots = not args.no_show

    image_list: List[Path] = []
    if args.image:
        image_list.append(Path(args.image))
    if args.folder:
        folder = Path(args.folder)
        exts = {e.strip().lower() for e in args.extensions.split(",")}
        for p in sorted(folder.iterdir()):
            if p.suffix.lower().lstrip(".") in exts:
                image_list.append(p)

    if not image_list:
        print("No images found.")
        sys.exit(1)

    print(f"Found {len(image_list)} image(s) to analyse.")

    for img_path in image_list:
        metrics = analyse_one_image(
            img_path,
            target_roi=args.target,
            background_roi=args.background,
            interactive=args.interactive,
            use_opencv_roi=args.opencv_roi,
            save_dir=save_dir,
            show_plots=show_plots,
        )
        save_results_csv(metrics, img_path.name, args.output)

    print("\nDone. Results saved to:", args.output)
    print("Visualisations saved under:", save_dir)


if __name__ == "__main__":
    main()