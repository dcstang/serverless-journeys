"""Tests for src/codes/search.py: the Google Custom Search JSON API client.

Uses the fake_search_endpoint fixture (tests/conftest.py) - a local HTTP
server speaking the Custom Search JSON API response shape - so these run
without real Google credentials or network access.
"""

from __future__ import annotations

import pytest

from src.codes import search


class TestIsConfigured:
    def test_true_when_both_env_vars_set(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "key")
        monkeypatch.setenv("GOOGLE_SEARCH_CSE_ID", "cse")
        assert search.is_configured() is True

    def test_false_when_api_key_missing(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_SEARCH_CSE_ID", "cse")
        assert search.is_configured() is False

    def test_false_when_cse_id_missing(self, monkeypatch):
        monkeypatch.setenv("GOOGLE_SEARCH_API_KEY", "key")
        monkeypatch.delenv("GOOGLE_SEARCH_CSE_ID", raising=False)
        assert search.is_configured() is False


class TestSearchWithoutCredentials:
    def test_raises_when_credentials_missing(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_SEARCH_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_SEARCH_CSE_ID", raising=False)
        with pytest.raises(search.SearchUnavailableError):
            search.search("test query")


class TestSearchAgainstFakeEndpoint:
    def test_returns_simplified_results(self, fake_search_endpoint):
        fake_search_endpoint.set_items(
            [
                {
                    "title": "ICD-10 I21.0",
                    "snippet": "Acute STEMI anterior wall",
                    "link": "https://example.com/1",
                },
                {"title": "Another result", "snippet": "More info", "link": "https://example.com/2"},
            ]
        )

        results = search.search("I21.0 clinical meaning")

        assert len(results) == 2
        assert results[0] == {
            "title": "ICD-10 I21.0",
            "snippet": "Acute STEMI anterior wall",
            "link": "https://example.com/1",
        }

    def test_sends_query_and_credentials_as_request_params(self, fake_search_endpoint):
        fake_search_endpoint.set_items([])
        search.search("some diagnostic code query")

        sent = fake_search_endpoint.requests_received[0]
        assert sent["q"] == "some diagnostic code query"
        assert sent["key"] == "fake-search-key"
        assert sent["cx"] == "fake-cse-id"

    def test_no_items_returns_empty_list(self, fake_search_endpoint):
        fake_search_endpoint.set_items([])
        assert search.search("no results query") == []

    def test_num_results_is_clamped_to_api_limit(self, fake_search_endpoint):
        fake_search_endpoint.set_items([])
        search.search("query", num_results=50)
        assert fake_search_endpoint.requests_received[0]["num"] == "10"

    def test_http_error_raises_search_unavailable(self, fake_search_endpoint):
        fake_search_endpoint.set_status(500)
        with pytest.raises(search.SearchUnavailableError):
            search.search("boom")
