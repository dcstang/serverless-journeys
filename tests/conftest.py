"""Shared pytest fixtures for the serverless-journeys test suite."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest


@pytest.fixture
def sample_patient() -> dict:
    """A minimal patient dict sufficient for prompt-building functions."""
    return {
        "person_id": "test-person-1",
        "full_name": "Jordan Smith",
        "age": 65,
        "sex": "Male",
        "past_medical_history": ["Hypertension"],
    }


class _FakeGoogleSearchHandler(BaseHTTPRequestHandler):
    response_items: list[dict] = []
    status_code = 200
    requests_received: list[dict] = []

    def do_GET(self):  # noqa: N802
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(self.path)
        query_params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
        type(self).requests_received.append(query_params)

        payload = {"items": type(self).response_items} if type(self).response_items else {}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(type(self).status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):  # noqa: A002
        pass  # silence default request logging


@pytest.fixture
def fake_search_endpoint(monkeypatch):
    """Start a local fake Google Custom Search server and point src.codes.search at it."""
    handler_cls = type(
        "FakeGoogleSearchHandler",
        (_FakeGoogleSearchHandler,),
        {"response_items": [], "status_code": 200, "requests_received": []},
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    base_url = f"http://127.0.0.1:{port}/customsearch/v1"

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "fake-search-key")
    monkeypatch.setenv("GOOGLE_SEARCH_CSE_ID", "fake-cse-id")
    monkeypatch.setenv("GOOGLE_SEARCH_ENDPOINT", base_url)

    def set_items(items: list[dict]) -> None:
        handler_cls.response_items = items

    def set_status(code: int) -> None:
        handler_cls.status_code = code

    ns = SimpleNamespace(
        base_url=base_url,
        set_items=set_items,
        set_status=set_status,
        requests_received=handler_cls.requests_received,
    )
    try:
        yield ns
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
