"""Compose slides: crop the photo, add a bottom gradient, headline, logos, and location line."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat

import config
from fonts_loader import load_bold_font, load_emoji_font, load_regular_font
from image_gen import draw_pin

Role = Literal["cover", "content", "cta", "story"]
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

WHITE = (255, 255, 255)
PILL_ALPHA = 225
PILL_CONTRAST_THRESHOLD = 90
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


def place_logos(image: Image.Image, top: int, margin: int) -> Image.Image:
    """Paste logos unmodified side by side; add a white pill when the background hides them."""
    height = int(image.height * config.LOGO_HEIGHT_RATIO)
    logos = [_scaled(logo, height) for logo in (_load_logo(p) for p in config.LOGO_FILES) if logo]
    if not logos:
        return image

    gap = int(height * 0.35)
    total_width = sum(l.width for l in logos) + gap * (len(logos) - 1)
    pad = int(height * 0.25)
    region = (margin - pad, top - pad, margin + total_width + pad, top + height + pad)

    background_lum = ImageStat.Stat(image.crop(region).convert("L")).mean[0]
    logo_lum = sum(_logo_luminance(l) for l in logos) / len(logos)
    base = image.convert("RGBA")
    if abs(background_lum - logo_lum) < PILL_CONTRAST_THRESHOLD:
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        ImageDraw.Draw(overlay).rounded_rectangle(
            region, radius=int((height + 2 * pad) / 2), fill=(255, 255, 255, PILL_ALPHA)
        )
        base = Image.alpha_composite(base, overlay)

    x = margin
    for logo in logos:
        base.alpha_composite(logo, (x, top))
        x += logo.width + gap
    return base.convert("RGB")


def _emoji_pin(target_height: int) -> Image.Image | None:
    font = load_emoji_font()
    if font is None:
        return None
    try:
        left, top, right, bottom = font.getbbox("📍")
        canvas = Image.new("RGBA", (int(right - left) + 4, int(bottom - top) + 4), (0, 0, 0, 0))
        ImageDraw.Draw(canvas).text((-left + 2, -top + 2), "📍", font=font, embedded_color=True)
        if canvas.getbbox() is None:
            return None
        return _scaled(canvas.crop(canvas.getbbox()), target_height)
    except (OSError, ValueError):
        return None


def draw_location(image: Image.Image, x: int, y: int, font: Font) -> int:
    """Draw the pin + LOCATION_LABEL at (x, y); returns the line height."""
    line_h = _line_height(font)
    icon_h = int(line_h * 1.25)
    icon = _emoji_pin(icon_h)
    if icon is not None:
        image.paste(icon, (x, y + (line_h - icon.height) // 2), icon)
        icon_w = icon.width
    else:
        draw = ImageDraw.Draw(image)
        icon_w = int(icon_h * 0.7)
        draw_pin(draw, x + icon_w / 2, y + line_h * 0.25, icon_h * 0.85, WHITE)
    text_x = x + icon_w + int(line_h * 0.45)
    _draw_text(ImageDraw.Draw(image), (text_x, y), config.location_label(), font)
    return line_h


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
