"""Slide backgrounds: real photo library first, then Gemini image model, then a branded fallback."""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter

import config

PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

IMAGE_PROMPT_STYLE = "photorealistic, natural light, modern Istanbul office / professional context"
IMAGE_PROMPT_BANS = (
    "no text, no letters, no logos, no watermarks, no ID cards, "
    "no documents with readable content, no screens with readable text"
)
PEOPLE_RULE = "anonymous, face not prominent"
PEOPLE_WORDS = ("person", "people", "man", "woman", "hand", "hands", "team", "employee", "customer", "colleague")


@dataclass
class SlideImage:
    image: Image.Image
    source: str
    photo: str | None = None


def build_image_prompt(scene: str) -> str:
    scene = scene.strip() or "a calm, modern office desk with a laptop and a coffee cup"
    parts = [scene, IMAGE_PROMPT_STYLE, IMAGE_PROMPT_BANS]
    if any(word in scene.lower() for word in PEOPLE_WORDS):
        parts.append(PEOPLE_RULE)
    return ", ".join(parts)


def _photo_library() -> list[tuple[Path, set[str]]]:
    if not config.PHOTOS_DIR.exists():
        return []
    tags_by_file: dict[str, set[str]] = {}
    if config.PHOTOS_MANIFEST_PATH.exists():
        try:
            manifest = json.loads(config.PHOTOS_MANIFEST_PATH.read_text(encoding="utf-8"))
            for item in manifest.get("photos", []):
                tags_by_file[str(item.get("file", ""))] = {str(t).lower() for t in item.get("tags", [])}
        except (OSError, json.JSONDecodeError, AttributeError):
            print("Warning: assets/photos/photos.json is invalid — using file names as tags.")

    library: list[tuple[Path, set[str]]] = []
    for path in sorted(config.PHOTOS_DIR.iterdir()):
        if path.suffix.lower() not in PHOTO_EXTENSIONS:
            continue
        name_tags = {part for part in path.stem.lower().replace("_", "-").split("-") if part and not part.isdigit()}
        library.append((path, tags_by_file.get(path.name, set()) | name_tags))
    return library


def select_photo(tags: list[str], usage: dict[str, int], exclude: set[str] | None = None) -> Path | None:
    """Least-used library photo whose tags overlap the topic tags."""
    wanted = {t.lower() for t in tags}
    skip = exclude or set()
    matches = [
        path
        for path, photo_tags in _photo_library()
        if wanted & photo_tags and _rel(path) not in skip
    ]
    if not matches:
        return None
    return min(matches, key=lambda p: (usage.get(_rel(p), 0), p.name))


def _rel(path: Path) -> str:
    try:
        return path.relative_to(config.BASE_DIR).as_posix()
    except ValueError:
        return path.as_posix()


@lru_cache(maxsize=1)
def _available_image_model() -> str | None:
    name = config.env("GEMINI_IMAGE_MODEL")
    key = config.env("GEMINI_API_KEY")
    if not name or not key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=key)
        available = {m.name.removeprefix("models/") for m in client.models.list()}
    except Exception as exc:
        print(f"Warning: could not list Gemini models ({type(exc).__name__}) — skipping AI images.")
        return None
    target = name.removeprefix("models/")
    if target not in available:
        print(f"Warning: GEMINI_IMAGE_MODEL '{target}' is not offered by the API — skipping AI images.")
        return None
    return target


def _aspect_ratio(size: tuple[int, int]) -> str:
    return "9:16" if size[1] / size[0] > 1.5 else "3:4"


def generate_ai_image(scene: str, size: tuple[int, int]) -> Image.Image | None:
    model = _available_image_model()
    if model is None:
        return None
    prompt = build_image_prompt(scene)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=config.env("GEMINI_API_KEY"))
        if model.startswith("imagen"):
            result = client.models.generate_images(
                model=model,
                prompt=prompt,
                config=types.GenerateImagesConfig(number_of_images=1, aspect_ratio=_aspect_ratio(size)),
            )
            data = result.generated_images[0].image.image_bytes
        else:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
            data = _first_inline_image(response)
        if not data:
            print("Warning: image model returned no image — using fallback background.")
            return None
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        print(f"Warning: AI image generation failed ({type(exc).__name__}) — using fallback background.")
        return None


def _first_inline_image(response: Any) -> bytes | None:
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is not None and getattr(inline, "data", None):
                return inline.data
    return None


def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def fallback_background(size: tuple[int, int], pillar: str = "") -> Image.Image:
    """Plain branded background with a simple line icon for the pillar."""
    width, height = size
    image = Image.new("RGB", size, config.BRAND_NAVY_DARK)
    draw = ImageDraw.Draw(image)
    for y in range(height):
        draw.line([(0, y), (width, y)], fill=_lerp(config.BRAND_NAVY, config.BRAND_NAVY_DARK, y / height))

    glow = Image.new("L", size, 0)
    ImageDraw.Draw(glow).ellipse(
        [width * 0.15, height * 0.12, width * 1.1, height * 0.62], fill=110
    )
    glow = glow.filter(ImageFilter.GaussianBlur(width * 0.12))
    image = Image.composite(Image.new("RGB", size, config.BRAND_BLUE), image, glow)

    draw = ImageDraw.Draw(image)
    cx, cy = width / 2, height * 0.38
    radius = width * 0.2
    stroke = max(6, int(width * 0.012))
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=config.BRAND_WHITE, width=stroke)
    _draw_icon(draw, pillar, cx, cy, radius * 0.55, stroke)
    draw.arc(
        [cx - radius * 1.25, cy - radius * 1.25, cx + radius * 1.25, cy + radius * 1.25],
        start=110, end=200, fill=config.BRAND_RED, width=stroke,
    )
    return image


def _draw_icon(draw: ImageDraw.ImageDraw, pillar: str, cx: float, cy: float, s: float, w: int) -> None:
    color = config.BRAND_WHITE
    if pillar in ("kep_bilgi", "cta_kampanya"):
        box = [cx - s, cy - s * 0.65, cx + s, cy + s * 0.65]
        draw.rectangle(box, outline=color, width=w)
        draw.line([(box[0], box[1]), (cx, cy + s * 0.1), (box[2], box[1])], fill=color, width=w, joint="curve")
    elif pillar == "eimza_bilgi":
        draw.line([(cx - s, cy + s * 0.7), (cx + s * 0.8, cy - s * 0.9)], fill=color, width=w * 2)
        draw.line([(cx - s, cy + s), (cx + s, cy + s)], fill=color, width=w)
    elif pillar == "basvuru_sureci":
        for i in range(3):
            y = cy - s * 0.6 + i * s * 0.6
            draw.line([(cx - s, y), (cx - s * 0.75, y + s * 0.2), (cx - s * 0.4, y - s * 0.2)], fill=color, width=w)
            draw.line([(cx - s * 0.15, y), (cx + s, y)], fill=color, width=w)
    elif pillar == "kurumsal":
        draw.rectangle([cx - s * 0.7, cy - s, cx + s * 0.7, cy + s], outline=color, width=w)
        for row in range(3):
            for col in range(2):
                x = cx - s * 0.35 + col * s * 0.5
                y = cy - s * 0.65 + row * s * 0.5
                draw.rectangle([x - s * 0.1, y, x + s * 0.1, y + s * 0.22], fill=color)
    elif pillar == "konum_guven":
        draw_pin(draw, cx, cy - s * 0.2, s * 1.6, color)
    else:
        draw.arc([cx - s * 0.6, cy - s, cx + s * 0.6, cy + s * 0.2], start=180, end=60, fill=color, width=w)
        draw.line([(cx + s * 0.3, cy - s * 0.1), (cx, cy + s * 0.35)], fill=color, width=w)
        draw.ellipse([cx - w, cy + s * 0.7 - w, cx + w, cy + s * 0.7 + w], fill=color)


def draw_pin(draw: ImageDraw.ImageDraw, cx: float, cy: float, height: float, color) -> None:
    """Map pin: circle head with a pointed tail; (cx, cy) is the head centre."""
    r = height * 0.32
    tip = (cx, cy + height * 0.68)
    angle = math.radians(35)
    left = (cx - r * math.cos(angle), cy + r * math.sin(angle))
    right = (cx + r * math.cos(angle), cy + r * math.sin(angle))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
    draw.polygon([left, right, tip], fill=color)
    hole = r * 0.42
    draw.ellipse([cx - hole, cy - hole, cx + hole, cy + hole], fill=config.BRAND_RED)


def get_slide_image(
    *,
    scene: str,
    photo_tags: list[str],
    pillar: str,
    size: tuple[int, int],
    usage: dict[str, int],
    exclude: set[str] | None = None,
) -> SlideImage:
    photo = select_photo(photo_tags, usage, exclude)
    if photo is not None:
        try:
            return SlideImage(Image.open(photo).convert("RGB"), "photo", _rel(photo))
        except OSError:
            print(f"Warning: could not open {photo.name} — trying the next source.")

    generated = generate_ai_image(scene, size)
    if generated is not None:
        return SlideImage(generated, "gemini")

    return SlideImage(fallback_background(size, pillar), "fallback")
