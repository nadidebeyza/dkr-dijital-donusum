#!/usr/bin/env python3
"""
DKR TÜRKKEP Başvuru Merkezi — daily 9:16 Instagram story.

Stories cannot carry a location tag via the API, so the location is drawn on the image.
Never run without --dry-run locally: a normal run publishes to the real account.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

import config
from content_history import record_published
from main import build_parser, obtain_plan, preflight, print_plan, render_plan


def run_story_pipeline(*, dry_run: bool = False, model: str | None = None) -> None:
    preflight(dry_run)
    plan = obtain_plan("story", dry_run=dry_run, model=model)
    out_dir = config.OUTPUT_DIR if dry_run else config.BASE_DIR
    paths, photos = render_plan(plan, "story", out_dir)
    print_plan(plan, None, paths)

    if dry_run:
        print("Dry run — nothing was published and history is unchanged.")
        return

    from image_host import get_public_image_url
    from instagram_client import InstagramClient

    url = get_public_image_url(paths[0])
    media_id = InstagramClient().publish_story(url)
    print(f"Published story: media_id={media_id}")
    record_published("story", plan, photos)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_parser("Generate and publish the daily DKR Instagram story.")
    args = parser.parse_args(argv)
    if args.format:
        print("Note: --format is ignored for stories (always 9:16 story).")
    try:
        run_story_pipeline(dry_run=args.dry_run, model=args.model)
        return 0
    except EnvironmentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Story pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
