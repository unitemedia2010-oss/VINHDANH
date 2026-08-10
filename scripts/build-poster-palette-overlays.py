#!/usr/bin/env python3
"""Create colour-matched Poster Studio overlay assets for all palettes.

The source artwork is already separated into two transparent layers:

* ``unite-foreground.png``: nameplate, laurels and podium.
* ``unite-portrait-frame-overlay.png``: circular rim and portrait laurels.

This script recolours RGB only.  It copies the source alpha channel byte for
byte, so output geometry and antialiasing remain identical across palettes.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Palette:
    slug: str
    label: str
    base: tuple[str, str, str]
    metal: tuple[str, str, str, str]


PALETTES = (
    Palette(
        "red-gold",
        "Red Gold",
        ("#080204", "#5b0810", "#bd2520"),
        ("#351300", "#a96005", "#f0bc35", "#fff1a1"),
    ),
    Palette(
        "blue-silver",
        "Blue Silver",
        ("#020711", "#10315c", "#3375ad"),
        ("#1d2733", "#77899d", "#d5e1ed", "#ffffff"),
    ),
    Palette(
        "champagne-gold",
        "Champagne Gold",
        ("#1b140d", "#8c7353", "#ead8b9"),
        ("#594018", "#c19539", "#f1d47f", "#fff6cf"),
    ),
    Palette(
        "amber-orange",
        "Amber Orange",
        ("#150501", "#792507", "#e86315"),
        ("#471500", "#bd4f00", "#f59f1b", "#ffe5a1"),
    ),
    Palette(
        "black-gold",
        "Black Gold",
        ("#010101", "#111214", "#35373a"),
        ("#351300", "#a96005", "#f0bc35", "#fff1a1"),
    ),
    Palette(
        "platinum-silver",
        "Platinum Silver",
        ("#080a0d", "#2c3138", "#69737f"),
        ("#222b35", "#8795a5", "#dce5ee", "#ffffff"),
    ),
    Palette(
        "crimson-silver",
        "Crimson Silver",
        ("#100206", "#600817", "#c5233e"),
        ("#202833", "#7f8c9c", "#d9e2ec", "#ffffff"),
    ),
    Palette(
        "midnight-blue-gold",
        "Midnight Blue Gold",
        ("#01040d", "#081a38", "#174977"),
        ("#351300", "#a96005", "#f0bc35", "#fff1a1"),
    ),
)


def hex_rgb(value: str) -> np.ndarray:
    value = value.lstrip("#")
    return np.asarray([int(value[index : index + 2], 16) for index in (0, 2, 4)], dtype=np.float32)


def ramp(values: np.ndarray, colours: tuple[str, ...]) -> np.ndarray:
    """Map normalized luminance through an evenly spaced RGB colour ramp."""
    stops = np.stack([hex_rgb(colour) for colour in colours])
    scaled = np.clip(values, 0.0, 1.0) * (len(stops) - 1)
    lower = np.floor(scaled).astype(np.int16)
    upper = np.minimum(lower + 1, len(stops) - 1)
    weight = (scaled - lower)[..., None]
    return stops[lower] * (1.0 - weight) + stops[upper] * weight


def foreground_base_mask(height: int, width: int, rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Identify the nameplate's coloured body while leaving metal ornaments out."""
    mask_image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask_image).polygon(
        [
            (218, 1118), (1010, 1118), (1028, 1170), (973, 1298),
            (255, 1298), (198, 1170),
        ],
        fill=255,
    )
    panel = np.asarray(mask_image, dtype=np.float32) / 255.0
    pixels = rgb.astype(np.float32)
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    luma = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
    minimum = np.min(pixels, axis=2)

    # Laurels, border lines and white glints inside the panel stay metallic.
    ornament = ((red - blue) > 34.0) & (luma > 0.39)
    ornament |= (minimum > 112.0) & (luma > 0.58)
    return (panel > 0.5) & (alpha > 0) & ~ornament


def recolour(source: Image.Image, palette: Palette, layer: str) -> Image.Image:
    source_rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    rgb = source_rgba[..., :3]
    alpha = source_rgba[..., 3]
    pixels = rgb.astype(np.float32)
    luma = (0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]) / 255.0

    # Metal benefits from stronger lifted mids while the panel keeps a deeper,
    # cinematic contrast.  Luminance drives the mapping and preserves every
    # highlight/shadow edge from the source artwork.
    metal_t = np.clip(np.power(luma, 0.72), 0.0, 1.0)
    base_t = np.clip(np.power(luma, 0.82), 0.0, 1.0)
    metal_rgb = ramp(metal_t, palette.metal)
    base_rgb = ramp(base_t, palette.base)

    if layer == "foreground":
        panel = foreground_base_mask(source.height, source.width, rgb, alpha)
        output_rgb = np.where(panel[..., None], base_rgb, metal_rgb)
    elif layer == "portrait-frame":
        output_rgb = metal_rgb
    else:
        raise ValueError(f"Unsupported layer: {layer}")

    output_rgb = np.clip(np.rint(output_rgb), 0, 255).astype(np.uint8)
    output_rgb[alpha == 0] = 0
    output = np.dstack([output_rgb, alpha])
    return Image.fromarray(output)


def make_preview(
    generated: list[tuple[Palette, Image.Image, Image.Image]],
    destination: Path,
) -> None:
    tile_width, tile_height = 360, 430
    sheet = Image.new("RGB", (tile_width * 4, tile_height * 2), "#080a10")
    font = ImageFont.load_default(size=18)
    crop_box = (115, 360, 1114, 1410)

    for index, (palette, foreground, portrait_frame) in enumerate(generated):
        preview = Image.new("RGBA", (1229, 1536), "#080a10")
        draw = ImageDraw.Draw(preview)
        draw.ellipse((272, 443, 944, 1115), fill=palette.base[1])
        preview.alpha_composite(portrait_frame)
        preview.alpha_composite(foreground)
        preview = preview.crop(crop_box)
        preview.thumbnail((tile_width, tile_height - 34), Image.Resampling.LANCZOS)

        tile = Image.new("RGB", (tile_width, tile_height), "#080a10")
        x = (tile_width - preview.width) // 2
        tile.paste(preview.convert("RGB"), (x, 0))
        ImageDraw.Draw(tile).text((14, tile_height - 28), palette.label, fill="#ffffff", font=font)
        sheet.paste(tile, ((index % 4) * tile_width, (index // 4) * tile_height))

    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG", compress_level=9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=Path("assets"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets/poster-palettes"))
    args = parser.parse_args()

    foreground_path = args.source_dir / "unite-foreground.png"
    portrait_frame_path = args.source_dir / "unite-portrait-frame-overlay.png"
    foreground_source = Image.open(foreground_path).convert("RGBA")
    portrait_frame_source = Image.open(portrait_frame_path).convert("RGBA")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[tuple[Palette, Image.Image, Image.Image]] = []
    manifest: dict[str, object] = {
        "canvas": {"width": foreground_source.width, "height": foreground_source.height},
        "palettes": [],
    }

    for palette in PALETTES:
        foreground = recolour(foreground_source, palette, "foreground")
        portrait_frame = recolour(portrait_frame_source, palette, "portrait-frame")
        foreground_output = args.output_dir / f"{palette.slug}-foreground.png"
        portrait_frame_output = args.output_dir / f"{palette.slug}-portrait-frame-overlay.png"
        foreground.save(foreground_output, format="PNG", compress_level=9)
        portrait_frame.save(portrait_frame_output, format="PNG", compress_level=9)

        if foreground.getchannel("A").tobytes() != foreground_source.getchannel("A").tobytes():
            raise AssertionError(f"Foreground alpha changed for {palette.slug}")
        if portrait_frame.getchannel("A").tobytes() != portrait_frame_source.getchannel("A").tobytes():
            raise AssertionError(f"Portrait-frame alpha changed for {palette.slug}")

        generated.append((palette, foreground, portrait_frame))
        manifest["palettes"].append(
            {
                "id": palette.slug,
                "label": palette.label,
                "foreground": foreground_output.name,
                "portraitFrameOverlay": portrait_frame_output.name,
                "baseRamp": list(palette.base),
                "metalRamp": list(palette.metal),
            }
        )

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    preview_path = args.output_dir / "palette-preview.png"
    make_preview(generated, preview_path)
    print(f"Generated {len(generated)} palettes in {args.output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()
