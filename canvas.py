"""Compose slides: crop the photo, add a bottom gradient, then the headline and text lines."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from PIL import Image, ImageDraw, ImageFont, ImageOps

import config
from fonts_loader import load_bold_font, load_regular_font

Role = Literal["cover", "content", "cta", "story"]
Font = ImageFont.FreeTypeFont | ImageFont.ImageFont

WHITE = (255, 255, 255)


@dataclass
class SlideText:
    headline: str
    subline: str = ""
    role: Role = "content"
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
        safe_bottom = height - int(height * config.STORY_SAFE_ZONE_RATIO)
    else:
        safe_bottom = height - margin
    max_width = width - 2 * margin

    head_font, head_lines = fit_headline(text.headline, height, max_width)
    sub_font = load_regular_font(int(height * config.SUBLINE_FONT_RATIO))
    sub_lines = wrap_text(text.subline, sub_font, max_width)[:2] if text.subline else []
    contact_font = load_regular_font(int(height * config.SUBLINE_FONT_RATIO * 0.9))

    head_h = _line_height(head_font)
    head_gap = int(head_h * 0.28)
    sub_h = _line_height(sub_font)
    contact_h = _line_height(contact_font)
    block_gap = int(height * 0.022)

    blocks: list[tuple[str, int]] = [("headline", len(head_lines) * head_h + (len(head_lines) - 1) * head_gap)]
    if sub_lines:
        blocks.append(("subline", len(sub_lines) * sub_h + (len(sub_lines) - 1) * int(sub_h * 0.4)))
    if text.contact_lines:
        blocks.append(("contact", len(text.contact_lines) * contact_h + (len(text.contact_lines) - 1) * int(contact_h * 0.6)))
    total = sum(h for _, h in blocks) + block_gap * (len(blocks) - 1)
    block_top = safe_bottom - total

    image = cover_crop(background, size)
    image = apply_bottom_gradient(image, block_top - int(height * 0.25))

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
        y += block_h + block_gap
    return image


def save_image(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path, format="PNG", optimize=True)
    return path
