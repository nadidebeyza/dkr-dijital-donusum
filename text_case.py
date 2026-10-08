"""Turkish casing for on-image text with correct İ/ı handling.

Headlines use title case ("KEP ile Standart E-Posta Arasındaki Fark"); sublines use sentence case.
"""

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

# Conjunctions and the question particle stay lowercase inside Turkish titles.
TITLE_LOWERCASE = {
    "ve", "ile", "veya", "ya", "da", "de", "ki",
    "mi", "mı", "mu", "mü",
    "misin", "mısın", "musun", "müsün",
    "miyiz", "mıyız", "muyuz", "müyüz",
    "misiniz", "mısınız", "musunuz", "müsünüz",
    "midir", "mıdır", "mudur", "müdür",
}

WORD_PATTERN = re.compile(r"([^\W\d_]+(?:-[^\W\d_]+)*)(['’][^\W\d_]+)?")
SENTENCE_START = re.compile(r"(^|[.!?…:]\s+)(\W*)(\w)")


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def turkish_upper(text: str) -> str:
    return text.replace("i", "İ").replace("ı", "I").upper()


def _capitalize(word: str) -> str:
    return turkish_upper(word[:1]) + word[1:] if word else word


def _canonical(base: str) -> str | None:
    return CANONICAL_WORDS.get(turkish_lower(base))


def _prepare(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sentence_case(text: str) -> str:
    """Lowercase everything, restore brand/place names, capitalise each sentence start."""
    text = _prepare(text)
    if not text:
        return text

    def word(match: re.Match[str]) -> str:
        base, suffix = match.group(1), match.group(2) or ""
        return (_canonical(base) or turkish_lower(base)) + turkish_lower(suffix)

    lowered = WORD_PATTERN.sub(word, text)
    lowered = SENTENCE_START.sub(lambda m: m.group(1) + m.group(2) + turkish_upper(m.group(3)), lowered)
    return _restore_phrases_case_sensitive(lowered)


def title_case(text: str) -> str:
    """Capitalise every word (and each hyphen part) except Turkish conjunctions and the mi particle."""
    text = _prepare(text)
    if not text:
        return text
    first = True

    def word(match: re.Match[str]) -> str:
        nonlocal first
        base, suffix = match.group(1), match.group(2) or ""
        canonical = _canonical(base)
        lower = turkish_lower(base)
        if canonical:
            out = canonical
            if first:
                out = _capitalize(out)
        elif lower in TITLE_LOWERCASE and not first:
            out = lower
        else:
            out = "-".join(_capitalize(part) for part in lower.split("-"))
        first = False
        return out + turkish_lower(suffix)

    return _restore_phrases_case_sensitive(WORD_PATTERN.sub(word, text))


def _restore_phrases_case_sensitive(text: str) -> str:
    lowered = turkish_lower(text)
    for phrase in CANONICAL_PHRASES:
        target = turkish_lower(phrase)
        start = lowered.find(target)
        while start != -1:
            text = text[:start] + phrase + text[start + len(phrase):]
            lowered = turkish_lower(text)
            start = lowered.find(target, start + len(phrase))
    return text
