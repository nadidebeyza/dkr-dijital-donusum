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


def plan_headline(plan: Any) -> str:
    slides = getattr(plan, "slides", None) or []
    return slides[0].headline if slides else ""


def fingerprint(topic_id: str, headline: str) -> str:
    return f"{topic_id}::{normalize_text(headline)}"


def plan_fingerprint(plan: Any) -> str:
    return fingerprint(plan.topic_id, plan_headline(plan))


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
    return {
        "topic_id": plan.topic_id,
        "pillar": plan.pillar,
        "format": plan.format,
        "headline": plan_headline(plan),
        "fingerprint": plan_fingerprint(plan),
        "photos": list(photos),
        "published_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }


def record_published(kind: Kind, plan: Any, photos: Iterable[str] = (), path: Path | None = None) -> None:
    entries = load_history(kind, path)
    entries.append(build_entry(plan, photos))
    save_history(kind, entries, path)
    print(f"Recorded {kind} history: {plan.topic_id}")


def photo_usage_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for kind in HISTORY_FILES:
        for entry in load_history(kind):  # type: ignore[arg-type]
            for photo in entry.get("photos") or []:
                counts[photo] = counts.get(photo, 0) + 1
    return counts


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


def is_exact_repeat(plan: Any, history: list[dict[str, Any]]) -> bool:
    headline = normalize_text(plan_headline(plan))
    fp = plan_fingerprint(plan)
    for entry in history[-config.history_size():]:
        if entry.get("topic_id") == plan.topic_id:
            return True
        if entry.get("fingerprint") == fp:
            return True
        if headline and normalize_text(str(entry.get("headline", ""))) == headline:
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
) -> Verdict:
    verdict = Verdict()
    if is_exact_repeat(plan, history):
        verdict.exact_repeat = True
        verdict.reasons.append(f"topic '{plan.topic_id}' or its headline was used recently")
    if is_pillar_repeat(plan, history, cooldown):
        verdict.pillar_repeat = True
        verdict.reasons.append(f"pillar '{plan.pillar}' is on cooldown")
    verdict.fabricated = find_fabricated_claims(plan.caption, facts_text)
    if verdict.fabricated:
        verdict.reasons.append("caption contains unsupported claims: " + ", ".join(verdict.fabricated))
    return verdict


def format_history_for_prompt(history: list[dict[str, Any]]) -> str:
    if not history:
        return "\n## Recently published\n(none yet)\n"
    lines = ["", "## Recently published — DO NOT REUSE these topic_ids or headlines"]
    for entry in history[-config.history_size():]:
        lines.append(f"- {entry.get('topic_id')} | {entry.get('pillar')} | {entry.get('headline')}")
    window = config.pillar_cooldown()
    if window > 0:
        blocked = [str(e.get("pillar")) for e in history[-window:]]
        lines.append(f"Pillars on cooldown (do not use): {', '.join(dict.fromkeys(blocked))}")
    return "\n".join(lines) + "\n"
