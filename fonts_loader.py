"""Fonts with full Turkish glyph coverage (ş, ğ, ı, İ, ç, ö, ü) plus optional color emoji."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BASE_DIR = Path(__file__).resolve().parent
FONTS_DIR = BASE_DIR / "fonts"
TURKISH_PROBE = "şğıİçöüŞĞÇÖÜ"

if sys.platform == "darwin":
    BOLD_CANDIDATES = [
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/Library/Fonts/Arial Bold.ttf"),
    ]
    REGULAR_CANDIDATES = [
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    EMOJI_CANDIDATES = [Path("/System/Library/Fonts/Apple Color Emoji.ttc")]
elif sys.platform.startswith("linux"):
    BOLD_CANDIDATES = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
    ]
    REGULAR_CANDIDATES = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    EMOJI_CANDIDATES = [
        Path("/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"),
        Path("/usr/share/fonts/noto/NotoColorEmoji.ttf"),
    ]
else:
    BOLD_CANDIDATES = [Path("C:/Windows/Fonts/arialbd.ttf"), Path("C:/Windows/Fonts/segoeuib.ttf")]
    REGULAR_CANDIDATES = [Path("C:/Windows/Fonts/arial.ttf"), Path("C:/Windows/Fonts/segoeui.ttf")]
    EMOJI_CANDIDATES = [Path("C:/Windows/Fonts/seguiemj.ttf")]

# Bitmap emoji fonts only load at their native strike sizes.
EMOJI_NATIVE_SIZES = (160, 137, 109, 96, 64)


def _bundled(kind: str) -> list[Path]:
    if not FONTS_DIR.exists():
        return []
    files = sorted(p for p in FONTS_DIR.iterdir() if p.suffix.lower() in {".ttf", ".otf"})
    if kind == "bold":
        return [p for p in files if "bold" in p.name.lower()]
    return [p for p in files if "bold" not in p.name.lower()]


def _covers_turkish(font: ImageFont.FreeTypeFont) -> bool:
    """Reject fonts that render Turkish letters as the same .notdef box."""
    def render(char: str) -> bytes:
        image = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(image).text((8, 8), char, font=font, fill=255)
        return image.tobytes()

    notdef = render("\uffff")
    return all(render(char) != notdef for char in TURKISH_PROBE)


@lru_cache(maxsize=None)
def _resolve(kind: str) -> Path | None:
    candidates = _bundled(kind) + (BOLD_CANDIDATES if kind == "bold" else REGULAR_CANDIDATES)
    for path in candidates:
        if not path.exists():
            continue
        try:
            font = ImageFont.truetype(str(path), size=40)
        except OSError:
            continue
        if _covers_turkish(font):
            return path
    print(f"Warning: no Turkish-capable {kind} font found — install fonts-dejavu-core or fonts-liberation.")
    return None


def _load(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = _resolve(kind)
    if path is not None:
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


def load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return _load("bold", size)


def load_regular_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    return _load("regular", size)


@lru_cache(maxsize=1)
def load_emoji_font() -> ImageFont.FreeTypeFont | None:
    """Return a color emoji font at a native size, or None when unavailable."""
    for path in EMOJI_CANDIDATES:
        if not path.exists():
            continue
        for size in EMOJI_NATIVE_SIZES:
            try:
                return ImageFont.truetype(str(path), size=size)
            except OSError:
                continue
    return None
