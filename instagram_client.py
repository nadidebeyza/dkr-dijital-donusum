"""Instagram Graph API publishing: single image, carousel, and story."""

from __future__ import annotations

import os
import time
from typing import Any

import requests

from instagram_setup import instagram_api_base, resolve_instagram_account_id

CONTAINER_POLL_SECONDS = 2
CONTAINER_TIMEOUT_SECONDS = 90
CAROUSEL_MIN = 2
CAROUSEL_MAX = 10


def location_id() -> str | None:
    value = (os.getenv("INSTAGRAM_LOCATION_ID") or "").strip()
    if not value:
        print("Warning: INSTAGRAM_LOCATION_ID is empty — publishing without a location tag.")
        return None
    return value


class InstagramClient:
    def __init__(self, account_id: str | None = None, access_token: str | None = None) -> None:
        self.access_token = (access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN", "")).strip()
        if not self.access_token:
            raise EnvironmentError("INSTAGRAM_ACCESS_TOKEN must be set.")
        self.base_url = instagram_api_base(self.access_token)
        self.account_id = resolve_instagram_account_id(account_id, self.access_token)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        query = {**(params or {}), "access_token": self.access_token}
        response = requests.request(method, url, params=query, data=data, timeout=120)
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not response.ok or "error" in payload:
            error = payload.get("error", {}) if isinstance(payload, dict) else {}
            message = error.get("message") or f"HTTP {response.status_code}"
            raise RuntimeError(f"Instagram API error ({response.status_code}): {message}")
        return payload

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", path, data=data)

    def _get(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        return self._request("GET", path, params=params)

    def _wait_for_container(self, creation_id: str) -> None:
        deadline = time.time() + CONTAINER_TIMEOUT_SECONDS
        while time.time() < deadline:
            payload = self._get(creation_id, {"fields": "status_code,status"})
            status = payload.get("status_code")
            if status in {"FINISHED", "PUBLISHED"}:
                return
            if status in {"ERROR", "EXPIRED"}:
                raise RuntimeError(f"Media container {status}: {payload.get('status', '')}")
            time.sleep(CONTAINER_POLL_SECONDS)
        raise TimeoutError("Timed out waiting for Instagram media container.")

    def _create_container(self, fields: dict[str, Any]) -> str:
        container = self._post(f"{self.account_id}/media", fields)
        creation_id = container["id"]
        self._wait_for_container(creation_id)
        return creation_id

    def _publish(self, creation_id: str) -> str:
        published = self._post(f"{self.account_id}/media_publish", {"creation_id": creation_id})
        return published["id"]

    def publish_post(self, image_url: str, caption: str) -> str:
        fields: dict[str, Any] = {"image_url": image_url, "caption": caption}
        loc = location_id()
        if loc:
            fields["location_id"] = loc
        return self._publish(self._create_container(fields))

    def publish_carousel(self, image_urls: list[str], caption: str) -> str:
        if not CAROUSEL_MIN <= len(image_urls) <= CAROUSEL_MAX:
            raise ValueError(f"Carousel needs {CAROUSEL_MIN}-{CAROUSEL_MAX} images, got {len(image_urls)}.")
        children = [
            self._create_container({"image_url": url, "is_carousel_item": "true"})
            for url in image_urls
        ]
        fields: dict[str, Any] = {
            "media_type": "CAROUSEL",
            "children": ",".join(children),
            "caption": caption,
        }
        loc = location_id()
        if loc:
            fields["location_id"] = loc
        return self._publish(self._create_container(fields))

    def publish_story(self, image_url: str) -> str:
        return self._publish(self._create_container({"image_url": image_url, "media_type": "STORIES"}))
