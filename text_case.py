"""Turkish sentence case for on-image text: no ALL CAPS, no Title Case, correct İ/ı handling."""

from __future__ import annotations

import re

CANONICAL_WORDS = {
    word.replace("I", "ı").replace("İ", "i").lower(): word
    for word in (
        "KEP", "DKR", "TÜRKKEP", "SSS", "BTK", "İK", "DM", "WhatsApp", "e-Devlet",
        "Acıbadem", "Üsküdar", "Kadıköy", "Ataşehir", "İstanbul", "Türkiye", "Anadolu", "Yakası",
    )
}
CANONICAL_PHRASES = ("DKR TÜRKKEP Başvuru Merkezi", "TÜRKKEP Başvuru Merkezi")
WORD_PATTERN = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
SENTENCE_START = re.compile(r"(^|[.!?…]\s+)(\W*)(\w)")


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def turkish_upper(text: str) -> str:
    return text.replace("i", "İ").replace("ı", "I").upper()


def _restore(match: re.Match[str]) -> str:
    word = match.group(0)
    return CANONICAL_WORDS.get(word, word)


def sentence_case(text: str) -> str:
    """Lowercase everything, restore brand/place names, capitalise each sentence start."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return text
    lowered = WORD_PATTERN.sub(_restore, turkish_lower(text))
    for phrase in CANONICAL_PHRASES:
        lowered = re.sub(re.escape(WORD_PATTERN.sub(_restore, turkish_lower(phrase))), phrase, lowered)
    return SENTENCE_START.sub(lambda m: m.group(1) + m.group(2) + turkish_upper(m.group(3)), lowered)
