"""Tests for src/codes/research.py: web-search-grounded code research.

Mocks src.codes.search.search and src.processing.call_llm directly (unit
level) - see tests/test_pipeline_correction.py for the full pipeline-level
integration through a fake search endpoint + fake LLM.
"""

from __future__ import annotations

import json

import pytest

from src import processing
from src.codes import registry, research, search

_TEST_SYSTEM = registry.CodeSystem(
    key="test-research-system",
    name="Test Research System",
    kind="diagnostic",
    codes={},
    specialty_field="specialty",
    type_field="admission_type",
    chapter_map={},
    default_specialty="General Medicine",
)


@pytest.fixture(autouse=True)
def _clear_research_cache():
    research.clear_cache()
    yield
    research.clear_cache()


def _fake_search_result():
    return [{"title": "M75.100", "snippet": "Rotator cuff syndrome, unspecified shoulder", "link": "https://x"}]


def _fake_synthesis_response(**overrides):
    payload = {
        "description": "Rotator cuff syndrome",
        "specialty": "Orthopaedics",
        "type": "elective",
        "typical_los_days_min": 0,
        "typical_los_days_max": 1,
        "confidence": "high",
    }
    payload.update(overrides)
    return json.dumps(payload)


class TestResearchCodeSuccess:
    def test_synthesizes_metadata_shaped_like_a_curated_entry(self, monkeypatch):
        monkeypatch.setattr(search, "search", lambda query, num_results=5: _fake_search_result())
        monkeypatch.setattr(
            processing, "call_llm", lambda prompt, model=None, temp=0.7, **kw: _fake_synthesis_response()
        )

        info = research.research_code(_TEST_SYSTEM, "M75.100")

        assert info["description"] == "Rotator cuff syndrome"
        assert info["specialty"] == "Orthopaedics"
        assert info["admission_type"] == "elective"
        assert info["typical_los_days"] == (0, 1)

    def test_search_query_includes_code_and_system_name(self, monkeypatch):
        captured = {}

        def fake_search(query, num_results=5):
            captured["query"] = query
            return _fake_search_result()

        monkeypatch.setattr(search, "search", fake_search)
        monkeypatch.setattr(
            processing, "call_llm", lambda prompt, model=None, temp=0.7, **kw: _fake_synthesis_response()
        )

        research.research_code(_TEST_SYSTEM, "M75.100")

        assert "M75.100" in captured["query"]
        assert _TEST_SYSTEM.name in captured["query"]

    def test_synthesis_prompt_includes_search_snippets(self, monkeypatch):
        captured = {}

        monkeypatch.setattr(search, "search", lambda query, num_results=5: _fake_search_result())

        def fake_call_llm(prompt, model=None, temp=0.7, **kw):
            captured["prompt"] = prompt
            return _fake_synthesis_response()

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        research.research_code(_TEST_SYSTEM, "M75.100")

        assert "Rotator cuff syndrome, unspecified shoulder" in captured["prompt"]

    def test_result_is_cached_per_system_and_code(self, monkeypatch):
        call_count = {"search": 0, "llm": 0}

        def fake_search(query, num_results=5):
            call_count["search"] += 1
            return _fake_search_result()

        def fake_call_llm(prompt, model=None, temp=0.7, **kw):
            call_count["llm"] += 1
            return _fake_synthesis_response()

        monkeypatch.setattr(search, "search", fake_search)
        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        research.research_code(_TEST_SYSTEM, "M75.100")
        research.research_code(_TEST_SYSTEM, "M75.100")
        research.research_code(_TEST_SYSTEM, "m75.100")  # case-insensitive same cache key

        assert call_count["search"] == 1
        assert call_count["llm"] == 1

    def test_clear_cache_forces_re_research(self, monkeypatch):
        call_count = {"search": 0}

        def fake_search(query, num_results=5):
            call_count["search"] += 1
            return _fake_search_result()

        monkeypatch.setattr(search, "search", fake_search)
        monkeypatch.setattr(
            processing, "call_llm", lambda prompt, model=None, temp=0.7, **kw: _fake_synthesis_response()
        )

        research.research_code(_TEST_SYSTEM, "M75.100")
        research.clear_cache()
        research.research_code(_TEST_SYSTEM, "M75.100")

        assert call_count["search"] == 2


class TestResearchCodeFailureModes:
    def test_search_unavailable_returns_none(self, monkeypatch):
        def raise_unavailable(query, num_results=5):
            raise search.SearchUnavailableError("no credentials")

        monkeypatch.setattr(search, "search", raise_unavailable)
        assert research.research_code(_TEST_SYSTEM, "X99.9") is None

    def test_no_search_results_returns_none(self, monkeypatch):
        monkeypatch.setattr(search, "search", lambda query, num_results=5: [])
        assert research.research_code(_TEST_SYSTEM, "X99.9") is None

    def test_low_confidence_synthesis_returns_none(self, monkeypatch):
        monkeypatch.setattr(search, "search", lambda query, num_results=5: _fake_search_result())
        monkeypatch.setattr(
            processing,
            "call_llm",
            lambda prompt, model=None, temp=0.7, **kw: _fake_synthesis_response(confidence="low"),
        )
        assert research.research_code(_TEST_SYSTEM, "X99.9") is None

    def test_unparseable_llm_response_returns_none(self, monkeypatch):
        monkeypatch.setattr(search, "search", lambda query, num_results=5: _fake_search_result())
        monkeypatch.setattr(processing, "call_llm", lambda prompt, model=None, temp=0.7, **kw: "not json")
        assert research.research_code(_TEST_SYSTEM, "X99.9") is None

    def test_missing_required_field_returns_none(self, monkeypatch):
        monkeypatch.setattr(search, "search", lambda query, num_results=5: _fake_search_result())
        monkeypatch.setattr(
            processing,
            "call_llm",
            lambda prompt, model=None, temp=0.7, **kw: json.dumps({"description": "d", "confidence": "high"}),
        )
        assert research.research_code(_TEST_SYSTEM, "X99.9") is None


class TestGetResearchedClinicalContext:
    def test_formats_researched_info_with_disclosure_prefix(self, monkeypatch):
        monkeypatch.setattr(search, "search", lambda query, num_results=5: _fake_search_result())
        monkeypatch.setattr(
            processing, "call_llm", lambda prompt, model=None, temp=0.7, **kw: _fake_synthesis_response()
        )

        context = research.get_researched_clinical_context(_TEST_SYSTEM, "M75.100")

        assert context is not None
        assert "Web-researched" in context
        assert "M75.100" in context
        assert "Rotator cuff syndrome" in context
        assert "Orthopaedics" in context

    def test_returns_none_when_research_fails(self, monkeypatch):
        monkeypatch.setattr(search, "search", lambda query, num_results=5: [])
        assert research.get_researched_clinical_context(_TEST_SYSTEM, "M75.100") is None
