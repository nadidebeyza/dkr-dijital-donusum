#!/usr/bin/env python3
"""
DKR TÜRKKEP Başvuru Merkezi — daily Instagram feed post (single 4:5 image or carousel).

Never run without --dry-run locally: a normal run publishes to the real account.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

import canvas
import config
from content_history import load_history, photo_usage_counts, record_published
from gemini_client import ContentPlan, generate_plan, offline_plan, plan_summary, topics_by_id
from image_gen import get_slide_image

Kind = Literal["story", "post"]


def build_parser(description: str = "Generate and publish the daily DKR Instagram feed post.") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render images into output/ and print the plan without publishing",
    )
    parser.add_argument("--model", default=None, help="Gemini text model placed first in the fallback chain")
    parser.add_argument(
        "--format",
        choices=("single", "carousel"),
        default=None,
        help="Force the post format (default: chosen by the plan)",
    )
    return parser


def contact_lines() -> list[str]:
    return [value for value in (config.env("BUSINESS_PHONE"), config.env("BUSINESS_WEBSITE")) if value]


def compose_caption(plan: ContentPlan) -> str:
    footer = [
        ("📍", config.env("BUSINESS_ADDRESS")),
        ("📞", config.env("BUSINESS_PHONE")),
        ("🕘", config.env("BUSINESS_HOURS")),
        ("🌐", config.env("BUSINESS_WEBSITE")),
    ]
    missing = [name for name, (_, value) in zip(
        ("BUSINESS_ADDRESS", "BUSINESS_PHONE", "BUSINESS_HOURS", "BUSINESS_WEBSITE"), footer
    ) if not value]
    if missing:
        print("Warning: caption footer is missing " + ", ".join(missing))

    hashtags = list(dict.fromkeys([*config.LOCAL_HASHTAGS, *plan.hashtags, *config.SERVICE_HASHTAGS]))[:30]
    parts = [plan.caption.strip()]
    footer_text = "\n".join(f"{icon} {value}" for icon, value in footer if value)
    if footer_text:
        parts.append(footer_text)
    parts.append(" ".join(hashtags))
    return "\n\n".join(parts)


def print_plan(plan: ContentPlan, caption: str | None, paths: list[Path]) -> None:
    print(plan_summary(plan))
    for path in paths:
        print(f"  Image: {path.relative_to(config.BASE_DIR)}")
    if caption is not None:
        print(f"\nCaption:\n{caption}\n")


def obtain_plan(kind: Kind, *, dry_run: bool, model: str | None, requested_format=None) -> ContentPlan:
    history = load_history(kind)
    if dry_run and not config.env("GEMINI_API_KEY"):
        print("Note: GEMINI_API_KEY is not set — dry run uses an offline plan from topics.json.")
        return offline_plan(kind, requested_format=requested_format, history=history)
    return generate_plan(kind, model=model, requested_format=requested_format, history=history)


def preflight(dry_run: bool) -> None:
    from instagram_setup import verify_instagram_setup

    if not dry_run:
        verify_instagram_setup()
        return
    if config.env("INSTAGRAM_ACCESS_TOKEN"):
        try:
            verify_instagram_setup()
        except Exception as exc:
            print(f"Warning (dry run): Instagram check failed — {exc}")


def render_plan(plan: ContentPlan, kind: Kind, out_dir: Path) -> tuple[list[Path], list[str]]:
    size = config.STORY_SIZE if kind == "story" else config.POST_SIZE
    topic = topics_by_id().get(plan.topic_id, {})
    tags = list(topic.get("photo_tags", []))
    usage = photo_usage_counts()
    used: set[str] = set()
    paths: list[Path] = []
    last = len(plan.slides) - 1

    for index, slide in enumerate(plan.slides):
        if kind == "story":
            text = canvas.SlideText(slide.headline, slide.subline, role="story", show_location=True)
        elif plan.format == "carousel" and index == last:
            text = canvas.SlideText(slide.headline, "", role="cta", show_location=True, contact_lines=contact_lines())
        else:
            text = canvas.SlideText(
                slide.headline, slide.subline, role="cover" if index == 0 else "content", show_location=index == 0
            )

        source = get_slide_image(
            scene=slide.image_prompt or topic.get("description", ""),
            photo_tags=tags,
            pillar=plan.pillar,
            size=size,
            usage=usage,
            exclude=used,
        )
        if source.photo:
            used.add(source.photo)
        print(f"Slide {index + 1}: background from {source.source}")

        name = "final_story.png" if kind == "story" else f"final_post_{index + 1}.png"
        paths.append(canvas.save_image(canvas.render_slide(source.image, size, text), out_dir / name))
    return paths, sorted(used)


def run_post_pipeline(*, dry_run: bool = False, model: str | None = None, requested_format=None) -> None:
    preflight(dry_run)
    plan = obtain_plan("post", dry_run=dry_run, model=model, requested_format=requested_format)
    caption = compose_caption(plan)
    out_dir = config.OUTPUT_DIR if dry_run else config.BASE_DIR
    paths, photos = render_plan(plan, "post", out_dir)
    print_plan(plan, caption, paths)

    if dry_run:
        print("Dry run — nothing was published and history is unchanged.")
        return

    from image_host import get_public_image_urls
    from instagram_client import InstagramClient

    urls = get_public_image_urls(paths)
    client = InstagramClient()
    if plan.format == "carousel":
        media_id = client.publish_carousel(urls, caption)
    else:
        media_id = client.publish_post(urls[0], caption)
    print(f"Published post: media_id={media_id}")
    record_published("post", plan, photos)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    try:
        run_post_pipeline(dry_run=args.dry_run, model=args.model, requested_format=args.format)
        return 0
    except EnvironmentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Post pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
