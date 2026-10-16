"""Content history: persistence, duplicate checks, and fabricated-claim detection."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import config

Kind = Literal["story", "post"]

HISTORY_FILES: dict[str, Path] = {
    "story": config.BASE_DIR / "story_history.json",
    "post": config.BASE_DIR / "post_history.json",
}

NUMBER_PATTERN = re.compile(r"\d+(?:[.,:/]\d+)*")
CURRENCY_PATTERN = re.compile(r"(?<!\w)tl(?!\w)|₺", re.IGNORECASE)
PERCENT_PATTERN = re.compile(r"%|\byüzde\b", re.IGNORECASE)
PRICE_WORDS_PATTERN = re.compile(r"\b(ücretsiz|bedava|indirim\w*|kampanya\w*|fiyat\w*|ücret\w*)\b", re.IGNORECASE)
UNIT_PATTERN = re.compile(r"(saniye|dakika|saat|gün|hafta|ay|yıl|adım|kişi|personel|belge|evrak|avantaj)")
HASHTAG_PATTERN = re.compile(r"#[\wçğıöşüÇĞİÖŞÜ]+")


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def normalize_text(text: str) -> str:
    """Lowercase (Turkish-aware), drop punctuation, keep Turkish letters, collapse spaces."""
    lowered = turkish_lower(text)
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def plan_headlines(plan: Any) -> list[str]:
    slides = getattr(plan, "slides", None) or []
    return [slide.headline for slide in slides if getattr(slide, "headline", "")]


def plan_headline(plan: Any) -> str:
    headlines = plan_headlines(plan)
    return headlines[0] if headlines else ""


def fingerprint(topic_id: str, headline: str) -> str:
    return f"{topic_id}::{normalize_text(headline)}"


def plan_fingerprint(plan: Any) -> str:
    return fingerprint(plan.topic_id, plan_headline(plan))


def other_kind(kind: Kind) -> Kind:
    return "post" if kind == "story" else "story"


def _history_path(kind: Kind) -> Path:
    return HISTORY_FILES[kind]


def load_history(kind: Kind, path: Path | None = None) -> list[dict[str, Any]]:
    target = path or _history_path(kind)
    if not target.exists():
        return []
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print(f"Warning: could not read {target.name} — starting with empty history.")
        return []
    entries = payload.get("entries", []) if isinstance(payload, dict) else []
    return [entry for entry in entries if isinstance(entry, dict)][-config.history_size():]


def save_history(kind: Kind, entries: list[dict[str, Any]], path: Path | None = None) -> None:
    target = path or _history_path(kind)
    trimmed = entries[-config.history_size():]
    target.write_text(
        json.dumps({"entries": trimmed}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_entry(plan: Any, photos: Iterable[str] = ()) -> dict[str, Any]:
    headlines = plan_headlines(plan)
    return {
        "topic_id": plan.topic_id,
        "pillar": plan.pillar,
        "format": plan.format,
        "headline": headlines[0] if headlines else "",
        "headlines": headlines,
        "fingerprint": plan_fingerprint(plan),
        "photos": list(photos),
        "published_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def record_published(kind: Kind, plan: Any, photos: Iterable[str] = (), path: Path | None = None) -> None:
    entries = load_history(kind, path)
    entries.append(build_entry(plan, photos))
    save_history(kind, entries, path)
    print(f"Recorded {kind} history: {plan.topic_id}")


def photo_usage_counts(kind: Kind | None = None) -> dict[str, int]:
    """How often each library photo was used. Pass a kind to count only that format."""
    counts: dict[str, int] = {}
    kinds: tuple[str, ...] = (kind,) if kind else tuple(HISTORY_FILES)
    for item in kinds:
        for entry in load_history(item):  # type: ignore[arg-type]
            for photo in entry.get("photos") or []:
                counts[photo] = counts.get(photo, 0) + 1
    return counts


def photos_used_by(kind: Kind) -> set[str]:
    used: set[str] = set()
    for entry in load_history(kind):
        used.update(str(photo) for photo in entry.get("photos") or [])
    return used


def _number_claims(text: str) -> list[tuple[str, str]]:
    """(number, unit) pairs; unit is the matched UNIT_PATTERN stem of the next word or ''."""
    claims: list[tuple[str, str]] = []
    for match in NUMBER_PATTERN.finditer(text):
        following = turkish_lower(text[match.end():match.end() + 20]).lstrip(" -'’")
        unit = UNIT_PATTERN.match(following)
        claims.append((match.group(0), unit.group(1) if unit else ""))
    return claims


def find_fabricated_claims(caption: str, facts_text: str) -> list[str]:
    """Return numbers, currency, percent, or price terms in the caption that are not backed by facts."""
    body = HASHTAG_PATTERN.sub(" ", caption)
    fact_claims = _number_claims(facts_text)
    fact_numbers = {number for number, _ in fact_claims}
    fact_pairs = set(fact_claims)
    problems: list[str] = []

    for number, unit in _number_claims(body):
        backed = (number, unit) in fact_pairs if unit else number in fact_numbers
        if not backed:
            problems.append(f"{number} {unit}".strip())

    for pattern, label in ((CURRENCY_PATTERN, "TL"), (PERCENT_PATTERN, "%")):
        match = pattern.search(body)
        if match and not pattern.search(facts_text):
            problems.append(match.group(0) or label)

    # Prices and campaigns are never part of the facts, so these words are always rejected.
    problems.extend(match.group(0) for match in PRICE_WORDS_PATTERN.finditer(body))

    return list(dict.fromkeys(problems))


@dataclass
class Verdict:
    exact_repeat: bool = False
    pillar_repeat: bool = False
    fabricated: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not (self.exact_repeat or self.pillar_repeat or self.fabricated)

    @property
    def only_pillar_violation(self) -> bool:
        return self.pillar_repeat and not self.exact_repeat and not self.fabricated


_HEADLINE_STOPWORDS = {
    "ile", "ve", "için", "bir", "bu", "şu", "o", "da", "de", "mi", "mı", "mu", "mü",
    "veya", "ya", "ki", "daha", "çok", "en", "her", "gibi",
}


def headline_tokens(text: str) -> set[str]:
    return {word for word in normalize_text(text).split() if len(word) > 2 and word not in _HEADLINE_STOPWORDS}


def headlines_too_similar(left: str, right: str) -> bool:
    """True when two headlines are the same message with small wording changes."""
    left_tokens = headline_tokens(left)
    right_tokens = headline_tokens(right)
    if len(left_tokens) < 3 or len(right_tokens) < 3:
        return False
    overlap = len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))
    return overlap >= 0.75


def _stored_headlines(entry: dict[str, Any]) -> list[str]:
    stored = [str(entry.get("headline", ""))]
    stored.extend(str(item) for item in entry.get("headlines") or [])
    return [item for item in stored if item.strip()]


def _headlines_conflict(plan: Any, entry: dict[str, Any], *, similar: bool) -> bool:
    for headline in plan_headlines(plan):
        normalized = normalize_text(headline)
        for stored in _stored_headlines(entry):
            if normalized and normalize_text(stored) == normalized:
                return True
            if similar and headlines_too_similar(headline, stored):
                return True
    return False


def is_exact_repeat(plan: Any, history: list[dict[str, Any]], other_history: list[dict[str, Any]] | None = None) -> bool:
    """Same topic or headline inside this format, or the same message on the other format."""
    fp = plan_fingerprint(plan)
    for entry in history[-config.history_size():]:
        if entry.get("topic_id") == plan.topic_id or entry.get("fingerprint") == fp:
            return True
        if _headlines_conflict(plan, entry, similar=False):
            return True
    for entry in (other_history or [])[-config.cross_channel_lookback():]:
        if entry.get("topic_id") == plan.topic_id or _headlines_conflict(plan, entry, similar=True):
            return True
    return False


def is_pillar_repeat(plan: Any, history: list[dict[str, Any]], cooldown: int | None = None) -> bool:
    window = config.pillar_cooldown() if cooldown is None else cooldown
    if window <= 0:
        return False
    return any(entry.get("pillar") == plan.pillar for entry in history[-window:])


def check_plan(
    plan: Any,
    history: list[dict[str, Any]],
    facts_text: str,
    *,
    cooldown: int | None = None,
    other_history: list[dict[str, Any]] | None = None,
) -> Verdict:
    verdict = Verdict()
    if is_exact_repeat(plan, history, other_history):
        verdict.exact_repeat = True
        verdict.reasons.append(
            f"topic '{plan.topic_id}' or its headline was already used as a story or a post"
        )
    if is_pillar_repeat(plan, history, cooldown):
        verdict.pillar_repeat = True
        verdict.reasons.append(f"pillar '{plan.pillar}' is on cooldown")
    verdict.fabricated = find_fabricated_claims(plan.caption, facts_text)
    if verdict.fabricated:
        verdict.reasons.append("caption contains unsupported claims: " + ", ".join(verdict.fabricated))
    return verdict


def _history_lines(entries: list[dict[str, Any]]) -> list[str]:
    if not entries:
        return ["(none yet)"]
    return [
        f"- {entry.get('topic_id')} | {entry.get('pillar')} | {entry.get('headline')}"
        for entry in entries[-config.history_size():]
    ]


def format_history_for_prompt(
    history: list[dict[str, Any]],
    other_history: list[dict[str, Any]] | None = None,
    *,
    kind: str = "post",
) -> str:
    own_label = "stories" if kind == "story" else "feed posts"
    own_singular = "story" if kind == "story" else "feed post"
    other_label = "feed posts" if kind == "story" else "stories"
    lines = [
        "",
        f"## Recently published {own_label} — DO NOT REUSE these topic_ids or headlines",
        *_history_lines(history),
    ]
    window = config.pillar_cooldown()
    if window > 0 and history:
        blocked = [str(entry.get("pillar")) for entry in history[-window:]]
        lines.append(f"Pillars on cooldown for this format only (do not use): {', '.join(dict.fromkeys(blocked))}")
    lines.append("")
    lines.append(
        f"## Recently published {other_label} — a new {own_singular} must NOT repeat these topics, headlines, or messages"
    )
    lines.extend(_history_lines((other_history or [])[-config.cross_channel_lookback():]))
    return "\n".join(lines) + "\n"
