#!/usr/bin/env python3
"""Build lightweight runtime backgrounds and a 10-theme QA contact sheet.

Nine AI-generated PNG sources in ``assets/poster-themes`` are preserved as
masters.  Runtime assets are deterministic 1229x1536 sRGB WebP files.  The QA
sheet composites each background with its matching portrait-frame and
foreground layers; no synthetic name, title or logo is drawn on the posters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageCms, ImageDraw, ImageFont, ImageOps


CANVAS_SIZE = (1229, 1536)
WEBP_QUALITY = 88
WEBP_METHOD = 6
BRAND_LOGO_BOX = (414, 38, 400)


@dataclass(frozen=True)
class Theme:
    slug: str
    label: str
    palette: str

    @property
    def source_name(self) -> str:
        return f"background-{self.slug}-source.png"

    @property
    def runtime_name(self) -> str:
        return f"background-{self.slug}.webp"


THEMES = (
    Theme("ruby-gold", "Ruby Gold", "red-gold"),
    Theme("sapphire-silver", "Sapphire Silver", "blue-silver"),
    Theme("champagne-gold", "Champagne Gold", "champagne-gold"),
    Theme("amber-orange", "Amber Orange", "amber-orange"),
    Theme("obsidian-gold", "Obsidian Gold", "black-gold"),
    Theme("platinum-silver", "Platinum Silver", "platinum-silver"),
    Theme("crimson-silver", "Crimson Silver", "crimson-silver"),
    Theme("midnight-blue-gold", "Midnight Blue Gold", "midnight-blue-gold"),
    Theme("solar-gold", "Solar Gold", "black-gold"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def srgb_profile() -> bytes:
    # LittleCMS stamps newly created profiles with the current second, which
    # would make otherwise identical WebP/PNG outputs hash differently.  ICC
    # bytes 24..35 are only the profile creation date; pinning a valid date
    # keeps the standard sRGB transform while making builds reproducible.
    profile = bytearray(ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes())
    profile[24:36] = struct.pack(">6H", 2000, 1, 1, 0, 0, 0)
    return bytes(profile)


def normalize_to_srgb(source: Image.Image) -> Image.Image:
    source = ImageOps.exif_transpose(source)
    embedded_profile = source.info.get("icc_profile")
    if embedded_profile:
        try:
            source_profile = ImageCms.ImageCmsProfile(__import__("io").BytesIO(embedded_profile))
            source = ImageCms.profileToProfile(
                source,
                source_profile,
                ImageCms.createProfile("sRGB"),
                outputMode="RGB",
            )
        except (OSError, ValueError):
            # AI exports occasionally carry a malformed profile.  Their RGB
            # pixels are already intended for screens, so conversion is safe.
            source = source.convert("RGB")
    else:
        source = source.convert("RGB")
    return source


def build_runtime(source_path: Path, output_path: Path, profile: bytes) -> None:
    with Image.open(source_path) as source:
        image = normalize_to_srgb(source)
        # All supplied masters are the same 4:5 composition.  Direct resize
        # keeps the full vertical design and changes aspect by less than 0.03%.
        image = image.resize(CANVAS_SIZE, Image.Resampling.LANCZOS)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(
            output_path,
            format="WEBP",
            quality=WEBP_QUALITY,
            method=WEBP_METHOD,
            lossless=False,
            icc_profile=profile,
        )


def composite_theme(
    background: Image.Image,
    portrait_frame: Image.Image,
    foreground: Image.Image,
    brand_logo: Image.Image | None = None,
) -> Image.Image:
    poster = background.convert("RGBA")
    if brand_logo is not None:
        logo = brand_logo.convert("RGBA")
        logo_width = BRAND_LOGO_BOX[2]
        logo_height = round(logo.height * logo_width / logo.width)
        logo = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
        poster.alpha_composite(logo, (BRAND_LOGO_BOX[0], BRAND_LOGO_BOX[1]))
    poster.alpha_composite(portrait_frame.convert("RGBA"))
    poster.alpha_composite(foreground.convert("RGBA"))
    return poster.convert("RGB")


def make_contact_sheet(
    themes_dir: Path,
    palettes_dir: Path,
    assets_dir: Path,
    destination: Path,
) -> None:
    entries: list[tuple[str, Image.Image]] = []
    brand_logo = Image.open(assets_dir / "unite-group-logo.png").convert("RGBA")

    # Existing Tinh Hoa theme keeps its original, hand-approved asset layers.
    tinhhoa = composite_theme(
        Image.open(assets_dir / "unite-bg-clean.png"),
        Image.open(assets_dir / "unite-portrait-frame-overlay.png"),
        Image.open(assets_dir / "unite-foreground.png"),
    )
    entries.append(("Tinh Hoa", tinhhoa))

    for theme in THEMES:
        poster = composite_theme(
            Image.open(themes_dir / theme.runtime_name),
            Image.open(palettes_dir / f"{theme.palette}-portrait-frame-overlay.png"),
            Image.open(palettes_dir / f"{theme.palette}-foreground.png"),
            brand_logo,
        )
        entries.append((theme.label, poster))

    tile_width, poster_height, label_height = 238, 298, 28
    gap = 8
    sheet_width = gap + 5 * (tile_width + gap)
    sheet_height = gap + 2 * (poster_height + label_height + gap)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "#07090e")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)

    for index, (label, poster) in enumerate(entries):
        thumbnail = poster.resize((tile_width, poster_height), Image.Resampling.LANCZOS)
        column, row = index % 5, index // 5
        x = gap + column * (tile_width + gap)
        y = gap + row * (poster_height + label_height + gap)
        sheet.paste(thumbnail, (x, y))
        draw.text((x + 6, y + poster_height + 5), label, fill="#f4f4f6", font=font)

    sheet.save(destination, format="PNG", compress_level=9, icc_profile=srgb_profile())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--themes-dir", type=Path, default=Path("assets/poster-themes"))
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument("--palettes-dir", type=Path, default=Path("assets/poster-palettes"))
    args = parser.parse_args()

    profile = srgb_profile()
    manifest_entries: list[dict[str, object]] = []
    with Image.open(args.assets_dir / "unite-group-logo.png") as logo_source:
        brand_logo_height = round(
            logo_source.height * BRAND_LOGO_BOX[2] / logo_source.width
        )

    for theme in THEMES:
        source_path = args.themes_dir / theme.source_name
        output_path = args.themes_dir / theme.runtime_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source background: {source_path}")
        build_runtime(source_path, output_path, profile)
        with Image.open(output_path) as runtime:
            if runtime.size != CANVAS_SIZE or runtime.mode != "RGB":
                raise AssertionError(f"Invalid runtime image: {output_path}")
            if not runtime.info.get("icc_profile"):
                raise AssertionError(f"Missing sRGB profile: {output_path}")
        manifest_entries.append(
            {
                "id": theme.slug,
                "label": theme.label,
                "source": theme.source_name,
                "runtime": theme.runtime_name,
                "palette": theme.palette,
                "foreground": f"../poster-palettes/{theme.palette}-foreground.png",
                "portraitFrameOverlay": (
                    f"../poster-palettes/{theme.palette}-portrait-frame-overlay.png"
                ),
                "brandLogoSrc": "../unite-group-logo.png",
                "sourceSha256": sha256(source_path),
                "runtimeSha256": sha256(output_path),
                "runtimeBytes": output_path.stat().st_size,
            }
        )

    manifest = {
        "canvas": {"width": CANVAS_SIZE[0], "height": CANVAS_SIZE[1]},
        "runtime": {
            "format": "webp",
            "colourSpace": "sRGB",
            "quality": WEBP_QUALITY,
            "method": WEBP_METHOD,
            "resize": "direct-lanczos",
        },
        "existingTheme": {
            "id": "tinhhoa",
            "label": "Tinh Hoa",
            "background": "../unite-bg-clean.png",
            "foreground": "../unite-foreground.png",
            "portraitFrameOverlay": "../unite-portrait-frame-overlay.png",
            "brandLogoBaked": True,
            "brandLogoSrc": None,
        },
        "brandLogoPlacement": {
            "x": BRAND_LOGO_BOX[0],
            "y": BRAND_LOGO_BOX[1],
            "width": BRAND_LOGO_BOX[2],
            "height": brand_logo_height,
            "fit": "contain",
        },
        "themes": manifest_entries,
        "preview": "theme-contact-sheet.png",
    }
    manifest_path = args.themes_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    preview_path = args.themes_dir / "theme-contact-sheet.png"
    make_contact_sheet(args.themes_dir, args.palettes_dir, args.assets_dir, preview_path)
    print(f"Generated {len(THEMES)} runtime WebP backgrounds in {args.themes_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Contact sheet: {preview_path}")


if __name__ == "__main__":
    main()
