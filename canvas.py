"""Compose slides: crop the photo, add a bottom gradient, headline, logos, and location line."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageStat

import config
from fonts_loader import load_bold_font, load_regular_font

Role = Literal["cover", "content", "cta", "story"]
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

WHITE = (255, 255, 255)
LOGO_CONTRAST_THRESHOLD = 90
HALO_STRENGTH = 2.2
_warned_missing: set[str] = set()


@dataclass
class SlideText:
    headline: str
    subline: str = ""
    role: Role = "content"
    show_location: bool = False
    contact_lines: list[str] = field(default_factory=list)


def cover_crop(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.45))


def apply_bottom_gradient(image: Image.Image, start_y: int, max_alpha: int = 215) -> Image.Image:
    width, height = image.size
    start_y = max(0, min(start_y, height - 1))
    mask = Image.new("L", (1, height), 0)
    span = max(1, height - start_y)
    for y in range(start_y, height):
        t = (y - start_y) / span
        mask.putpixel((0, y), int(max_alpha * min(1.0, t * 1.6) ** 1.2))
    mask = mask.resize((width, height))
    shade = Image.new("RGB", (width, height), config.BRAND_NAVY_DARK)
    return Image.composite(shade, image, mask)


def _text_width(font: Font, text: str) -> int:
    left, _, right, _ = font.getbbox(text)
    return int(right - left)


def _line_height(font: Font) -> int:
    _, top, _, bottom = font.getbbox("ÇĞİŞÖÜgjy")
    return int(bottom - top)


def wrap_text(text: str, font: Font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if current and _text_width(font, candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def fit_headline(text: str, height: int, max_width: int) -> tuple[Font, list[str]]:
    """Largest font near 6% of height that fits in HEADLINE_MAX_LINES lines."""
    size = int(height * config.HEADLINE_FONT_RATIO)
    minimum = int(size * 0.6)
    while True:
        font = load_bold_font(size)
        lines = wrap_text(text, font, max_width)
        fits = len(lines) <= config.HEADLINE_MAX_LINES and all(_text_width(font, l) <= max_width for l in lines)
        if fits or size <= minimum:
            return font, lines[: config.HEADLINE_MAX_LINES]
        size = int(size * 0.93)


@lru_cache(maxsize=None)
def _load_logo(path: Path) -> Image.Image | None:
    if not path.exists():
        if path.name not in _warned_missing:
            print(f"Warning: logo missing — add {path.relative_to(config.BASE_DIR)} (skipping it for now).")
            _warned_missing.add(path.name)
        return None
    return Image.open(path).convert("RGBA")


def _scaled(logo: Image.Image, target_height: int) -> Image.Image:
    ratio = target_height / logo.height
    return logo.resize((max(1, round(logo.width * ratio)), target_height), Image.Resampling.LANCZOS)


def _logo_luminance(logo: Image.Image) -> float:
    gray = logo.convert("L")
    mask = logo.getchannel("A").point(lambda a: 255 if a > 128 else 0)
    stat = ImageStat.Stat(gray, mask)
    return stat.mean[0] if stat.count[0] else 128.0


def _corner_score(image: Image.Image, region: tuple[int, int, int, int], logo_lum: float) -> tuple[float, float]:
    """(score, contrast): prefer light, calm areas behind the dark logo."""
    stat = ImageStat.Stat(image.crop(region).convert("L"))
    contrast = stat.mean[0] - logo_lum
    return contrast - 0.6 * stat.stddev[0], contrast


def _soft_halo(logo: Image.Image, height: int) -> Image.Image:
    """Feathered light glow that follows the logo's own outline (no box or pill)."""
    pad = height // 2
    canvas = Image.new("L", (logo.width + 2 * pad, logo.height + 2 * pad), 0)
    canvas.paste(logo.getchannel("A"), (pad, pad))
    spread = max(3, (int(height * 0.12) // 2) * 2 + 1)
    for _ in range(3):
        canvas = canvas.filter(ImageFilter.MaxFilter(spread))
    canvas = canvas.filter(ImageFilter.GaussianBlur(height * 0.12)).point(lambda a: min(235, int(a * HALO_STRENGTH)))
    halo = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    halo.putalpha(canvas)
    return halo


def place_logos(image: Image.Image, top: int, margin: int) -> Image.Image:
    """Paste logos unmodified in the top corner whose background suits them best."""
    height = int(image.height * config.LOGO_HEIGHT_RATIO)
    logos = [_scaled(logo, height) for logo in (_load_logo(p) for p in config.LOGO_FILES) if logo]
    if not logos:
        return image

    gap = int(height * 0.35)
    total_width = sum(l.width for l in logos) + gap * (len(logos) - 1)
    pad = int(height * 0.3)
    logo_lum = sum(_logo_luminance(l) for l in logos) / len(logos)

    candidates = {
        "left": margin,
        "right": image.width - margin - total_width,
    }
    scored = {
        side: _corner_score(image, (x - pad, top - pad, x + total_width + pad, top + height + pad), logo_lum)
        for side, x in candidates.items()
    }
    side = max(scored, key=lambda s: scored[s][0])
    contrast = scored[side][1]

    base = image.convert("RGBA")
    x = candidates[side]
    for logo in logos:
        if contrast < LOGO_CONTRAST_THRESHOLD:
            halo = _soft_halo(logo, height)
            offset = (halo.width - logo.width) // 2
            base.alpha_composite(halo, (x - offset, top - offset))
        base.alpha_composite(logo, (x, top))
        x += logo.width + gap
    return base.convert("RGB")


def draw_location(image: Image.Image, x: int, y: int, font: Font) -> int:
    """Draw LOCATION_LABEL as plain text at (x, y); returns the line height."""
    _draw_text(ImageDraw.Draw(image), (x, y), config.location_label(), font)
    return _line_height(font)


def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: Font, fill=WHITE) -> None:
    _, top, _, _ = font.getbbox(text)
    x, y = xy[0], xy[1] - int(top)
    shadow = max(2, int(getattr(font, "size", 40) * 0.04))
    draw.text((x + shadow, y + shadow), text, font=font, fill=(0, 0, 0))
    draw.text((x, y), text, font=font, fill=fill)


def render_slide(background: Image.Image, size: tuple[int, int], text: SlideText) -> Image.Image:
    width, height = size
    margin = int(width * config.EDGE_MARGIN_RATIO)
    if text.role == "story":
        safe_top = int(height * config.STORY_SAFE_ZONE_RATIO)
        safe_bottom = height - int(height * config.STORY_SAFE_ZONE_RATIO)
    else:
        safe_top = margin
        safe_bottom = height - margin
    max_width = width - 2 * margin

    head_font, head_lines = fit_headline(text.headline, height, max_width)
    sub_font = load_regular_font(int(height * config.SUBLINE_FONT_RATIO))
    sub_lines = wrap_text(text.subline, sub_font, max_width)[:2] if text.subline else []
    contact_font = load_regular_font(int(height * config.SUBLINE_FONT_RATIO * 0.9))
    loc_font = load_bold_font(int(height * config.LOCATION_FONT_RATIO))

    head_h = _line_height(head_font)
    head_gap = int(head_h * 0.28)
    sub_h = _line_height(sub_font)
    contact_h = _line_height(contact_font)
    loc_h = _line_height(loc_font)
    block_gap = int(height * 0.022)

    blocks: list[tuple[str, int]] = [("headline", len(head_lines) * head_h + (len(head_lines) - 1) * head_gap)]
    if sub_lines:
        blocks.append(("subline", len(sub_lines) * sub_h + (len(sub_lines) - 1) * int(sub_h * 0.4)))
    if text.contact_lines:
        blocks.append(("contact", len(text.contact_lines) * contact_h + (len(text.contact_lines) - 1) * int(contact_h * 0.6)))
    if text.show_location:
        blocks.append(("location", loc_h))
    total = sum(h for _, h in blocks) + block_gap * (len(blocks) - 1)
    block_top = safe_bottom - total

    image = cover_crop(background, size)
    image = apply_bottom_gradient(image, block_top - int(height * 0.25))
    image = place_logos(image, safe_top, margin)

    draw = ImageDraw.Draw(image)
    y = block_top
    for name, block_h in blocks:
        if name == "headline":
            for i, line in enumerate(head_lines):
                _draw_text(draw, (margin, y + i * (head_h + head_gap)), line, head_font)
        elif name == "subline":
            for i, line in enumerate(sub_lines):
                _draw_text(draw, (margin, y + i * int(sub_h * 1.4)), line, sub_font, fill=(235, 238, 245))
        elif name == "contact":
            for i, line in enumerate(text.contact_lines):
                _draw_text(draw, (margin, y + i * int(contact_h * 1.6)), line, contact_font, fill=(235, 238, 245))
        elif name == "location":
            draw_location(image, margin, y, loc_font)
            draw = ImageDraw.Draw(image)
        y += block_h + block_gap
    return image


def save_image(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=True)
    return path
