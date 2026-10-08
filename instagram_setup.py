"""Pre-flight checks for Instagram Graph API credentials."""

from __future__ import annotations

import os

import requests

GRAPH_VERSION = "v21.0"
FACEBOOK_GRAPH_BASE = f"https://graph.facebook.com/{GRAPH_VERSION}"
INSTAGRAM_LOGIN_API_VERSION = os.getenv("INSTAGRAM_LOGIN_API_VERSION", "v23.0")
INSTAGRAM_LOGIN_API_BASE = f"https://graph.instagram.com/{INSTAGRAM_LOGIN_API_VERSION}"


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _uses_instagram_login_api(token: str) -> bool:
    return token.startswith(("IGAA", "IGQ", "IGQV"))


def _graph_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    error = payload.get("error") or {}
    message = error.get("message") or response.text or response.reason
    code = error.get("code")
    subcode = error.get("error_subcode")
    parts = [message]
    if code is not None:
        parts.append(f"code={code}")
    if subcode is not None:
        parts.append(f"subcode={subcode}")
    return " ".join(parts)


def _verify_facebook_login_token(token: str, ig_id: str) -> str:
    if not ig_id.isdigit():
        raise RuntimeError(
            "INSTAGRAM_ACCOUNT_ID must be a numeric Instagram Business Account ID."
        )

    response = requests.get(
        f"{FACEBOOK_GRAPH_BASE}/{ig_id}",
        params={"fields": "username,name", "access_token": token},
        timeout=30,
    )
    if response.ok:
        username = response.json().get("username", "?")
        return username

    detail = _graph_error_message(response)
    if response.status_code in {400, 401, 403}:
        raise RuntimeError(
            "Instagram access token is invalid, expired, or missing publish permissions.\n"
            f"API response: {detail}\n"
            "Fix in Meta for Developers → Graph API Explorer:\n"
            "  1. Select your App and Facebook Page\n"
            "  2. Add permissions: instagram_basic, instagram_content_publish, "
            "pages_read_engagement, pages_show_list\n"
            "  3. Generate a Page Access Token (not a User token)\n"
            "  4. Update INSTAGRAM_ACCESS_TOKEN in GitHub Secrets / .env\n"
            "     (no spaces or trailing newlines)"
        )
    response.raise_for_status()
    return "?"


def _verify_instagram_login_token(token: str) -> tuple[str, str]:
    response = requests.get(
        f"{INSTAGRAM_LOGIN_API_BASE}/me",
        params={"fields": "id,user_id,username", "access_token": token},
        timeout=30,
    )
    if not response.ok:
        detail = _graph_error_message(response)
        raise RuntimeError(
            "Instagram Login access token is invalid or expired.\n"
            f"API response: {detail}\n"
            "Generate a fresh token in Meta for Developers and update "
            "INSTAGRAM_ACCESS_TOKEN in GitHub Secrets / .env."
        )

    data = response.json()
    account_id = str(data.get("user_id") or data.get("id") or "")
    username = str(data.get("username") or "?")
    if not account_id:
        raise RuntimeError("Could not resolve Instagram account id from /me.")
    return account_id, username


def resolve_instagram_account_id(
    account_id: str | None = None,
    access_token: str | None = None,
) -> str:
    """Return the Instagram account id used for publishing endpoints."""
    token = _clean(access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN"))
    ig_id = _clean(account_id or os.getenv("INSTAGRAM_ACCOUNT_ID"))
    if _uses_instagram_login_api(token):
        resolved_id, _ = _verify_instagram_login_token(token)
        return resolved_id
    return ig_id or _account_id_from_page_token(token)


def _account_id_from_page_token(token: str) -> str:
    """Look up the Instagram Business Account linked to the Page that owns this token."""
    owner = requests.get(
        f"{FACEBOOK_GRAPH_BASE}/me",
        params={"fields": "id,name", "access_token": token},
        timeout=30,
    )
    if not owner.ok:
        raise RuntimeError(
            "INSTAGRAM_ACCESS_TOKEN was rejected by the Graph API: "
            f"{_graph_error_message(owner)}"
        )
    owner_name = owner.json().get("name", "?")

    response = requests.get(
        f"{FACEBOOK_GRAPH_BASE}/me",
        params={"fields": "instagram_business_account", "access_token": token},
        timeout=30,
    )
    if response.ok:
        linked = (response.json().get("instagram_business_account") or {}).get("id")
        if linked:
            print(f"Note: INSTAGRAM_ACCOUNT_ID is empty — using the Instagram account linked to Page '{owner_name}'.")
            return str(linked)
        raise RuntimeError(
            f"INSTAGRAM_ACCOUNT_ID is empty and Page '{owner_name}' has no linked Instagram business "
            "account. Link the Instagram account to this Page (Page settings → Linked accounts → "
            "Instagram) or set INSTAGRAM_ACCOUNT_ID."
        )
    raise RuntimeError(
        f"INSTAGRAM_ACCOUNT_ID is empty and INSTAGRAM_ACCESS_TOKEN belongs to '{owner_name}', which "
        "looks like a User token, not a Page token. Run me/accounts in Graph API Explorer and use "
        "the DKR Page's access_token. "
        f"API response: {_graph_error_message(response)}"
    )


def instagram_api_base(access_token: str | None = None) -> str:
    token = _clean(access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN"))
    if _uses_instagram_login_api(token):
        return INSTAGRAM_LOGIN_API_BASE
    return FACEBOOK_GRAPH_BASE


def uses_instagram_login_api(access_token: str | None = None) -> bool:
    token = _clean(access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN"))
    return _uses_instagram_login_api(token)


def verify_instagram_setup(
    account_id: str | None = None,
    access_token: str | None = None,
) -> None:
    """
    Validate Instagram credentials before publishing.
    Uses a direct Graph API call instead of debug_token, which often returns 400
    unless an app access token is supplied.
    """
    token = _clean(access_token or os.getenv("INSTAGRAM_ACCESS_TOKEN"))
    ig_id = _clean(account_id or os.getenv("INSTAGRAM_ACCOUNT_ID"))
    if not token:
        raise RuntimeError("Missing INSTAGRAM_ACCESS_TOKEN in .env or GitHub Secrets.")

    if _uses_instagram_login_api(token):
        resolved_id, username = _verify_instagram_login_token(token)
        if ig_id and ig_id != resolved_id:
            print(
                "Warning: INSTAGRAM_ACCOUNT_ID does not match the token's account; "
                "using the token account."
            )
        print(f"Instagram ready: @{username}")
        return

    username = _verify_facebook_login_token(token, ig_id or _account_id_from_page_token(token))
    print(f"Instagram ready: @{username}")
