#!/usr/bin/env python3
"""Build the transparent portrait-frame overlay used by Poster Studio.

The source poster is a flattened RGB image.  This script isolates only the
golden circular rim and the two laurel branches that need to sit in front of a
circle-cropped portrait.  The mask is intentionally constrained by geometry;
global colour-keying would also pick up the many gold rays in the background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


CANVAS_WIDTH = 1229
CANVAS_HEIGHT = 1536

# Geometry audited against assets/unite-bg-clean.png.
RING_CENTER = (608.0, 779.0)
RING_RADIUS = 365.0


def smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    values = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return values * values * (3.0 - 2.0 * values)


def polygon_mask(shape: tuple[int, int], points: list[tuple[int, int]], feather: float) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(points, dtype=np.int32)], 255)
    if feather > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), feather)
    return mask.astype(np.float32) / 255.0


def build_geometry_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    distance = np.hypot(xx - RING_CENTER[0], yy - RING_CENTER[1])

    # The rim itself is about 8-14px wide.  A soft 30px band also preserves its
    # small star glows without carrying a dark circular strip into the portrait.
    ring = smoothstep(RING_RADIUS - 22.0, RING_RADIUS - 9.0, distance)
    ring *= 1.0 - smoothstep(RING_RADIUS + 10.0, RING_RADIUS + 27.0, distance)
    ring *= 1.0 - smoothstep(1090.0, 1124.0, yy)

    # Tight silhouettes around the two laurel branches.  They deliberately
    # exclude the faceted monument at left and the background light rays at
    # right, which share almost the same gold colour.
    left_branch = polygon_mask(
        (height, width),
        [
            (207, 692), (242, 710), (277, 764), (310, 826),
            (348, 910), (389, 1003), (405, 1097), (319, 1102),
            (260, 1052), (207, 979), (165, 904), (176, 833),
            (190, 775),
        ],
        feather=2.0,
    )
    right_branch = polygon_mask(
        (height, width),
        [
            (972, 694), (941, 714), (910, 771), (884, 837),
            (850, 919), (813, 1007), (797, 1098), (887, 1103),
            (946, 1053), (1000, 981), (1040, 906), (1034, 837),
            (1018, 774),
        ],
        feather=2.0,
    )

    branches = np.maximum(left_branch, right_branch)
    return ring, branches


def build_colour_alpha(rgb: np.ndarray) -> np.ndarray:
    pixels = rgb.astype(np.float32)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    value = np.max(pixels, axis=2)
    minimum = np.min(pixels, axis=2)
    luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue

    # Golden material is warm (R/G above B); the rim also contains near-white
    # specular highlights, so high neutral luminance gets a second path.
    warmth = smoothstep(14.0, 68.0, red - blue)
    gold_luminance = smoothstep(62.0, 142.0, luma)
    highlight = smoothstep(148.0, 235.0, value) * smoothstep(105.0, 205.0, minimum)
    alpha = np.maximum(gold_luminance * warmth, highlight)

    # Dark source pixels must never become even a faint black veil over a light
    # portrait.  Keep a hard floor as well as the soft luminance ramp: the
    # source contains dark brown anti-aliasing around the plaque/rim which is
    # acceptable on its black background but creates visible jagged seams on a
    # white crop.
    visible_gold = (luma >= 62.0) & (
        ((red - blue) >= 14.0) | ((luma >= 168.0) & (minimum >= 104.0))
    )
    alpha *= visible_gold.astype(np.float32)
    alpha[alpha < 0.12] = 0.0
    return alpha


def build_overlay(source_path: Path, output_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read source image: {source_path}")
    height, width = bgr.shape[:2]
    if (width, height) != (CANVAS_WIDTH, CANVAS_HEIGHT):
        raise ValueError(
            f"Unexpected source size {width}x{height}; expected "
            f"{CANVAS_WIDTH}x{CANVAS_HEIGHT}"
        )

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ring_geometry, branch_geometry = build_geometry_masks(height, width)
    colour_alpha = build_colour_alpha(rgb)
    pixels = rgb.astype(np.float32)
    luma = 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    # The right/top portion of the rim is a very thin, nearly neutral highlight
    # rather than saturated gold.  Keep that highlight through a ring-only
    # luminance path; the tight annulus prevents unrelated pale pixels from
    # leaking into the overlay.
    rim_alpha = smoothstep(50.0, 138.0, luma)
    combined_alpha = np.maximum(
        branch_geometry * colour_alpha,
        ring_geometry * np.maximum(colour_alpha, rim_alpha),
    )
    combined_alpha[combined_alpha < 0.12] = 0.0
    alpha = np.clip(combined_alpha * 255.0, 0, 255).astype(np.uint8)

    # Remove isolated background sparkles while retaining the long, thin rim.
    supported = cv2.morphologyEx(
        (alpha > 20).astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)),
    )
    alpha = np.where(supported > 0, alpha, 0).astype(np.uint8)

    # Clear hidden RGB outside the mask.  Besides making the transparency
    # unambiguous, this shrinks the full-canvas PNG dramatically.
    rgb_out = rgb.copy().astype(np.float32)
    # Lift the remaining low mids to a readable gold floor.  This affects only
    # extracted overlay pixels and prevents source-brown edge pixels from
    # drawing a dark contour over pale portraits.
    luma = (
        0.2126 * rgb_out[..., 0]
        + 0.7152 * rgb_out[..., 1]
        + 0.0722 * rgb_out[..., 2]
    )
    lift = np.where((alpha > 0) & (luma < 92.0), 92.0 / np.maximum(luma, 1.0), 1.0)
    rgb_out = np.clip(rgb_out * lift[..., None], 0, 255).astype(np.uint8)
    rgb_out[alpha == 0] = 0
    rgba = np.dstack([rgb_out, alpha])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(
        str(output_path),
        cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA),
        [cv2.IMWRITE_PNG_COMPRESSION, 9],
    ):
        raise OSError(f"Cannot write overlay: {output_path}")
    return rgba


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("assets/unite-bg-clean.png"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("assets/unite-portrait-frame-overlay.png"),
    )
    args = parser.parse_args()

    rgba = build_overlay(args.source, args.output)
    alpha = rgba[..., 3]
    ys, xs = np.nonzero(alpha)
    bbox = None if len(xs) == 0 else (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    opaque_pixels = int(np.count_nonzero(alpha))
    semi_opaque_pixels = int(np.count_nonzero(alpha >= 128))
    print(f"Wrote {args.output}")
    print(f"Canvas: {rgba.shape[1]}x{rgba.shape[0]}")
    print(f"Alpha bbox: {bbox}")
    print(f"Alpha pixels: {opaque_pixels}; alpha>=128: {semi_opaque_pixels}")


if __name__ == "__main__":
    main()
