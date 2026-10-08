"""Gemini text generation: content plan JSON, model fallback chain, duplicate regeneration."""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any, Literal

import config
from content_history import (
    check_plan,
    format_history_for_prompt,
    load_history,
)

Kind = Literal["story", "post"]
PostFormat = Literal["single", "carousel", "story"]

MODEL_FALLBACKS = (
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-flash-latest",
    "gemini-3.6-flash",
)

CAROUSEL_MIN = 2
CAROUSEL_MAX = 10
CTA_HEADLINE = "Başvuru için bize ulaşın"


@dataclass
class Slide:
    headline: str
    subline: str = ""
    image_prompt: str = ""


@dataclass
class ContentPlan:
    topic_id: str
    pillar: str
    format: PostFormat
    slides: list[Slide]
    caption: str
    hashtags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@lru_cache(maxsize=1)
def load_facts() -> dict[str, Any]:
    return json.loads(config.BUSINESS_FACTS_PATH.read_text(encoding="utf-8"))


def _string_values(node: Any) -> list[str]:
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict):
        return [s for value in node.values() for s in _string_values(value)]
    if isinstance(node, list):
        return [s for value in node for s in _string_values(value)]
    return []


def facts_text() -> str:
    """Prose of the facts file only; JSON keys and step numbers are not citable facts."""
    return "\n".join(_string_values(load_facts()))


@lru_cache(maxsize=1)
def load_topics() -> tuple[dict[str, Any], ...]:
    payload = json.loads(config.TOPICS_PATH.read_text(encoding="utf-8"))
    return tuple(payload.get("topics", []))


def topics_by_id() -> dict[str, dict[str, Any]]:
    return {topic["topic_id"]: topic for topic in load_topics()}


PROMPT_TEMPLATE = """
You write daily Instagram content for "{brand}", an authorized TÜRKKEP application center in Acıbadem, Üsküdar (Istanbul, Anatolian side).

Goals: attract customers (drive calls, DMs, applications) and inform them about KEP, e-imza, the application process, renewal and corporate solutions.
Audience: company owners, accountants, lawyers, sole proprietors and individuals who need e-imza in Üsküdar, Acıbadem, Kadıköy, Ataşehir.
Tone: trustworthy, plain, professional, warm. Visuals are photo-led with very little text.

## Business facts (the ONLY source of truth)
{facts}

## Topic pool (pick exactly one topic_id from here)
{topics}
{history}
## Task
Create one {kind_label}. Required format: {format_rule}

Return ONLY a JSON object:
{{
  "topic_id": "<topic_id from the pool>",
  "pillar": "<that topic's pillar>",
  "format": "{format_value}",
  "slides": [
    {{"headline": "Türkçe başlık", "subline": "", "image_prompt": "English, photorealistic scene description"}}
  ],
  "caption": "Türkçe caption",
  "hashtags": ["#kep", "#eimza"]
}}

Rules:
1. headline: Turkish, at most {max_words} words, correct Turkish characters (ş, ğ, ı, İ, ç, ö, ü). subline: optional, one short Turkish line or "".
2. caption: Turkish, 2-5 short paragraphs, informative, ends with a call to action (call, DM, or visit). Do NOT include address, phone, hours or website — they are appended automatically.
3. NEVER invent prices, fees, durations, discounts, campaigns, percentages, legal claims or guarantees. Use numbers ONLY if they appear in the business facts. Never write "TL", "%", "ücretsiz", "indirim", "kampanya" or "fiyat".
4. Do not speak on behalf of TÜRKKEP; no official announcements.
5. Only mention the services listed in the facts.
6. image_prompt: English, a real-world photographic scene that fits the slide (modern Istanbul office, desk, laptop, hands signing, city view). No text, logos, documents, ID cards or screens in the scene. People only anonymous, faces not prominent.
7. hashtags: 3-8 lowercase Turkish service hashtags.
8. Do NOT reuse any topic_id or headline from "Recently published" and do not use pillars on cooldown.
{carousel_rules}
"""

CAROUSEL_RULES = (
    "9. Carousel: {min}-{max} slides. Slide 1 is an eye-catching cover. Middle slides each carry one idea. "
    "The LAST slide is a call to action (its headline will be replaced with \"{cta}\")."
)


def _format_rule(kind: Kind, requested: PostFormat | None) -> tuple[str, str, str]:
    if kind == "story":
        return "a single 9:16 story (exactly 1 slide).", "story", ""
    if requested == "single":
        return "a single 4:5 image post (exactly 1 slide).", "single", ""
    carousel = CAROUSEL_RULES.format(min=CAROUSEL_MIN, max=CAROUSEL_MAX, cta=CTA_HEADLINE)
    if requested == "carousel":
        return f"a 4:5 carousel post with {CAROUSEL_MIN}-{CAROUSEL_MAX} slides.", "carousel", carousel
    return (
        "either a single 4:5 image post (\"single\", 1 slide) or a 4:5 carousel (\"carousel\", "
        f"{CAROUSEL_MIN}-{CAROUSEL_MAX} slides). Prefer the topic's suggested format.",
        "single|carousel",
        carousel,
    )


def build_prompt(
    kind: Kind,
    history: list[dict[str, Any]],
    *,
    requested_format: PostFormat | None = None,
    rejection_note: str = "",
) -> str:
    topics = "\n".join(
        f"- {t['topic_id']} | {t['pillar']} | {t['format']} | {t['description']}" for t in load_topics()
    )
    format_rule, format_value, carousel_rules = _format_rule(kind, requested_format)
    prompt = PROMPT_TEMPLATE.format(
        brand=config.brand_name(),
        facts=json.dumps(load_facts(), ensure_ascii=False, indent=1),
        topics=topics,
        history=format_history_for_prompt(history),
        kind_label="Instagram story" if kind == "story" else "Instagram feed post",
        format_rule=format_rule,
        format_value=format_value,
        max_words=config.HEADLINE_MAX_WORDS,
        carousel_rules=carousel_rules,
    )
    if rejection_note:
        prompt += (
            "\nYour previous answer repeated a banned topic or broke a rule "
            f"({rejection_note}). Choose a different topic_id and pillar and follow every rule.\n"
        )
    return prompt


def _extract_json(raw: str) -> dict[str, Any]:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Gemini response is not a JSON object.")
    return data


def _clip_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit])


def _normalize_hashtags(raw: Any) -> list[str]:
    tags: list[str] = []
    for item in raw if isinstance(raw, list) else str(raw or "").split():
        tag = str(item).strip()
        if not tag:
            continue
        tag = "#" + tag.lstrip("#").replace(" ", "")
        tags.append(tag.replace("I", "ı").replace("İ", "i").lower())
    return list(dict.fromkeys(tags))


def normalize_plan(data: dict[str, Any], kind: Kind, requested_format: PostFormat | None = None) -> ContentPlan:
    topics = topics_by_id()
    topic_id = str(data.get("topic_id", "")).strip()
    if topic_id not in topics:
        raise ValueError(f"unknown topic_id '{topic_id}'")
    topic = topics[topic_id]

    raw_slides = data.get("slides") or []
    slides = [
        Slide(
            headline=_clip_words(str(s.get("headline", "")).strip(), config.HEADLINE_MAX_WORDS),
            subline=str(s.get("subline", "") or "").strip(),
            image_prompt=str(s.get("image_prompt", "") or "").strip(),
        )
        for s in raw_slides
        if isinstance(s, dict)
    ]
    slides = [s for s in slides if s.headline]
    if not slides:
        raise ValueError("plan has no slides with a headline")

    if kind == "story":
        fmt: PostFormat = "story"
    elif requested_format in ("single", "carousel"):
        fmt = requested_format
    else:
        fmt = "carousel" if str(data.get("format")) == "carousel" or len(slides) > 1 else "single"

    if fmt in ("story", "single"):
        slides = slides[:1]
    else:
        slides = slides[:CAROUSEL_MAX]
        if len(slides) < CAROUSEL_MIN:
            raise ValueError(f"carousel needs {CAROUSEL_MIN}-{CAROUSEL_MAX} slides, got {len(slides)}")
        slides[-1].headline = CTA_HEADLINE

    caption = str(data.get("caption", "")).strip()
    if not caption:
        raise ValueError("plan has an empty caption")

    return ContentPlan(
        topic_id=topic_id,
        pillar=topic["pillar"],
        format=fmt,
        slides=slides,
        caption=caption,
        hashtags=_normalize_hashtags(data.get("hashtags")),
    )


def _retry_delay_seconds(error: Exception) -> int | None:
    match = re.search(r"retry in (\d+(?:\.\d+)?)s", str(error), re.IGNORECASE)
    if match:
        return max(1, int(float(match.group(1))))
    return None


def _is_retryable(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "429",
            "503",
            "quota",
            "rate limit",
            "resource exhausted",
            "resource_exhausted",
            "unavailable",
            "high demand",
            "404",
            "not_found",
            "not found",
            "no longer available",
        )
    )


def _call_model(client, model: str, prompt: str) -> dict[str, Any]:
    from google.genai import types

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.9),
    )
    return _extract_json(response.text)


def _generate_once(client, models: list[str], prompt: str) -> dict[str, Any]:
    errors: list[str] = []
    for candidate in models:
        try:
            data = _call_model(client, candidate, prompt)
            if candidate != models[0]:
                print(f"Note: used fallback model '{candidate}'.")
            return data
        except json.JSONDecodeError as exc:
            errors.append(f"{candidate}: invalid JSON ({exc})")
            continue
        except Exception as exc:
            if not _is_retryable(exc):
                raise
            delay = _retry_delay_seconds(exc)
            if delay and delay <= 60:
                print(f"Rate limited on {candidate}, retrying in {delay}s...")
                time.sleep(delay)
                try:
                    return _call_model(client, candidate, prompt)
                except Exception as retry_exc:
                    errors.append(f"{candidate}: {type(retry_exc).__name__}")
                    continue
            errors.append(f"{candidate}: {type(exc).__name__}")
    raise RuntimeError("All Gemini text models failed. Details: " + " | ".join(errors))


def model_chain(model: str | None = None) -> list[str]:
    preferred = model or config.env("GEMINI_MODEL")
    models = [preferred] if preferred else []
    for fallback in MODEL_FALLBACKS:
        if fallback not in models:
            models.append(fallback)
    return models


def select_plan(
    kind: Kind,
    produce,
    history: list[dict[str, Any]],
    *,
    retries: int | None = None,
) -> ContentPlan:
    """
    Run produce(rejection_note) until a clean plan appears.

    Falls back to the first plan that only broke the pillar cooldown; never returns
    an exact repeat or a plan with fabricated claims.
    """
    attempts = retries or config.duplicate_retries()
    facts = facts_text()
    pillar_only: ContentPlan | None = None
    note = ""

    for attempt in range(1, attempts + 1):
        try:
            plan = produce(note)
        except ValueError as exc:
            note = f"invalid plan: {exc}"
            print(f"Plan rejected ({attempt}/{attempts}): {note}")
            continue

        verdict = check_plan(plan, history, facts)
        if verdict.clean:
            return plan
        if verdict.only_pillar_violation and pillar_only is None:
            pillar_only = plan
        note = "; ".join(verdict.reasons)
        print(f"Plan rejected ({attempt}/{attempts}): {plan.topic_id} — {note}")

    if pillar_only is not None:
        print(
            f"Warning: no fully clean {kind} plan after {attempts} attempts — "
            f"publishing '{pillar_only.topic_id}' despite pillar cooldown."
        )
        return pillar_only

    raise RuntimeError(
        f"Could not produce a valid {kind} plan after {attempts} attempts "
        "(duplicates or unsupported claims). Increase DUPLICATE_CONTENT_RETRIES or review history."
    )


def generate_plan(
    kind: Kind,
    *,
    model: str | None = None,
    requested_format: PostFormat | None = None,
    history: list[dict[str, Any]] | None = None,
) -> ContentPlan:
    key = config.env("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError("GEMINI_API_KEY is not set.")

    from google import genai

    recent = history if history is not None else load_history(kind)
    if recent:
        print(f"Avoiding {len(recent)} recent {kind} entries from history")

    client = genai.Client(api_key=key)
    models = model_chain(model)

    def produce(note: str) -> ContentPlan:
        prompt = build_prompt(kind, recent, requested_format=requested_format, rejection_note=note)
        return normalize_plan(_generate_once(client, models, prompt), kind, requested_format)

    return select_plan(kind, produce, recent)


def offline_plan(
    kind: Kind,
    *,
    requested_format: PostFormat | None = None,
    history: list[dict[str, Any]] | None = None,
) -> ContentPlan:
    """Rule-based plan from topics.json for local dry runs without GEMINI_API_KEY."""
    recent = history if history is not None else load_history(kind)
    facts = facts_text()
    topics = list(load_topics())
    if kind == "story":
        topics.sort(key=lambda t: t["format"] != "story")
    elif requested_format:
        topics.sort(key=lambda t: t["format"] != requested_format)

    for topic in topics:
        if kind == "story":
            fmt: PostFormat = "story"
        elif requested_format in ("single", "carousel"):
            fmt = requested_format
        else:
            fmt = "carousel" if topic["format"] == "carousel" else "single"

        cover = Slide(
            headline=_clip_words(topic["headline"], config.HEADLINE_MAX_WORDS),
            subline="KEP ve e-imza için yetkili başvuru merkezi",
            image_prompt=f"{topic['description']} — modern Istanbul office scene",
        )
        slides = [cover]
        if fmt == "carousel":
            slides.append(Slide(headline="Uzman ekip, hızlı süreç", subline="Yüz yüze veya uzaktan başvuru"))
            slides.append(Slide(headline=CTA_HEADLINE))

        plan = ContentPlan(
            topic_id=topic["topic_id"],
            pillar=topic["pillar"],
            format=fmt,
            slides=slides,
            caption=(
                f"{topic['description']}\n\n"
                "KEP ve e-imza başvurularınızda Acıbadem'deki merkezimizde yüz yüze ya da uzaktan yanınızdayız.\n\n"
                "Bilgi ve randevu için bize DM atın ya da arayın."
            ),
            hashtags=["#kep", "#eimza"],
        )
        if check_plan(plan, recent, facts).clean:
            return plan

    raise RuntimeError("No unused topic available for the offline plan.")


def plan_summary(plan: ContentPlan) -> str:
    lines = [f"Topic: {plan.topic_id} ({plan.pillar}) — format: {plan.format}"]
    for index, slide in enumerate(plan.slides, 1):
        sub = f" / {slide.subline}" if slide.subline else ""
        lines.append(f"  Slide {index}: {slide.headline}{sub}")
    return "\n".join(lines)


__all__ = [
    "CTA_HEADLINE",
    "ContentPlan",
    "Slide",
    "facts_text",
    "generate_plan",
    "load_facts",
    "load_topics",
    "offline_plan",
    "plan_summary",
    "select_plan",
    "topics_by_id",
]
