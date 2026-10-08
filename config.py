"""Layout constants and environment-backed settings for DKR TÜRKKEP automation."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"
LOGOS_DIR = ASSETS_DIR / "logos"
PHOTOS_DIR = ASSETS_DIR / "photos"
OUTPUT_DIR = BASE_DIR / "output"

BUSINESS_FACTS_PATH = DATA_DIR / "business_facts.json"
TOPICS_PATH = DATA_DIR / "topics.json"
PHOTOS_MANIFEST_PATH = PHOTOS_DIR / "photos.json"

LOGO_FILES = (LOGOS_DIR / "dkr.png", LOGOS_DIR / "turkkep.png")

POST_SIZE = (1080, 1350)
STORY_SIZE = (1080, 1920)

LOGO_HEIGHT_RATIO = 0.055
EDGE_MARGIN_RATIO = 0.04
HEADLINE_FONT_RATIO = 0.06
SUBLINE_FONT_RATIO = 0.03
LOCATION_FONT_RATIO = 0.024
STORY_SAFE_ZONE_RATIO = 0.12
HEADLINE_MAX_LINES = 2
HEADLINE_MAX_WORDS = 8

# Used only for the plain branded fallback background (from the DKR website).
BRAND_NAVY_DARK = (1, 24, 53)
BRAND_NAVY = (4, 37, 86)
BRAND_BLUE = (8, 73, 166)
BRAND_RED = (193, 18, 30)
BRAND_WHITE = (245, 246, 248)

LOCAL_HASHTAGS = ("#üsküdar", "#acıbadem", "#kadıköy", "#anadoluyakası")
SERVICE_HASHTAGS = ("#kep", "#eimza", "#türkkep", "#kayıtlıelektronikposta", "#elektronikimza")


def env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def env_int(name: str, default: int) -> int:
    raw = env(name)
    try:
        return int(raw) if raw else default
    except ValueError:
        print(f"Warning: {name} is not an integer — using default {default}.")
        return default


def brand_name() -> str:
    return env("BRAND_NAME", "DKR TÜRKKEP Başvuru Merkezi")


def location_label() -> str:
    label = env("LOCATION_LABEL", "Acıbadem, Üsküdar")
    # The on-image location is plain text: emoji and pin symbols are dropped.
    return "".join(ch for ch in label if ord(ch) < 0x2190 or ch == "·").strip()


def history_size() -> int:
    return env_int("CONTENT_HISTORY_SIZE", 30)


def pillar_cooldown() -> int:
    return env_int("PILLAR_COOLDOWN", 2)


def duplicate_retries() -> int:
    return max(1, env_int("DUPLICATE_CONTENT_RETRIES", 5))
