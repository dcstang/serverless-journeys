"""
Google Custom Search JSON API client, used to research diagnostic/procedure
codes that aren't in a CodeSystem's curated dictionary and whose meaning
isn't obvious from the bare code alone (see src/codes/research.py, which
consumes this module's results).

Requires:
  GOOGLE_SEARCH_API_KEY - Google Cloud API key with the Custom Search API enabled
  GOOGLE_SEARCH_CSE_ID  - Programmable Search Engine (CSE) ID, configured to
                          search the entire web. Create one at
                          https://programmablesearchengine.google.com/

Env var GOOGLE_SEARCH_ENDPOINT overrides the API base URL (used by tests to
point at a local fake server instead of the real Google endpoint).
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

GOOGLE_SEARCH_ENDPOINT = "https://www.googleapis.com/customsearch/v1"


class SearchUnavailableError(RuntimeError):
    """Raised when search credentials are missing or the search request fails."""


def is_configured() -> bool:
    """Return True if Google Custom Search credentials are present."""
    return bool(os.environ.get("GOOGLE_SEARCH_API_KEY")) and bool(os.environ.get("GOOGLE_SEARCH_CSE_ID"))


def search(query: str, num_results: int = 5, timeout: float = 10.0) -> list[dict[str, str]]:
    """Run a Google Custom Search query and return simplified results.

    Args:
        query: Search query string.
        num_results: Max number of results to request (clamped to 1-10,
            the Custom Search API's per-request limit).
        timeout: HTTP request timeout in seconds.

    Returns:
        List of dicts with keys 'title', 'snippet', 'link'. Empty list if
        the search returns no results.

    Raises:
        SearchUnavailableError: If credentials are missing or the request fails.
    """
    import httpx  # noqa: PLC0415

    api_key = os.environ.get("GOOGLE_SEARCH_API_KEY")
    cse_id = os.environ.get("GOOGLE_SEARCH_CSE_ID")
    if not api_key or not cse_id:
        raise SearchUnavailableError(
            "GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CSE_ID must both be set "
            "to use code research (see src/codes/search.py docstring)."
        )

    endpoint = os.environ.get("GOOGLE_SEARCH_ENDPOINT", GOOGLE_SEARCH_ENDPOINT)
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": max(1, min(num_results, 10)),
    }

    try:
        response = httpx.get(endpoint, params=params, timeout=timeout)
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        # ValueError covers response.json() failing to decode a malformed
        # body even on a 200 status (json.JSONDecodeError is a ValueError
        # subclass, not an httpx.HTTPError).
        raise SearchUnavailableError(f"Google Custom Search request failed: {exc}") from exc

    items = data.get("items") if isinstance(data, dict) else None
    if not isinstance(items, list):
        items = []

    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("snippet", ""),
            "link": item.get("link", ""),
        }
        for item in items
        if isinstance(item, dict)
    ]
