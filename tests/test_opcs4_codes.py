"""Unit tests for src/codes/opcs4.py: parsing, lookup, and context generation.

Mirrors tests/test_icd10_codes.py for the OPCS-4 procedure code module.

OPCS-4's curated dictionary (code_systems/opcs4.json) is intentionally
empty - an audit against the NHS's official OPCS-4 tabular list found most
of the original curated entries pointed at the wrong procedure entirely
(e.g. codes labelled as thyroid or breast surgery that are officially
something else), so the bad data was removed pending proper curation
rather than left in place. Every OPCS-4 code is therefore "uncurated" for
now - these tests reflect that, exercising the same fallback path any
uncurated standard uses (see src/codes/registry.py).
"""

from __future__ import annotations

from src.codes import opcs4


class TestParseCodes:
    def test_parses_comma_separated_codes(self):
        assert opcs4.parse_codes("K40.1, W37.1,H01.1") == ["K40.1", "W37.1", "H01.1"]

    def test_empty_string_returns_empty_list(self):
        assert opcs4.parse_codes("") == []
        assert opcs4.parse_codes("   ") == []

    def test_uppercases_and_strips_codes(self):
        assert opcs4.parse_codes(" k40.1 , w37.1 ") == ["K40.1", "W37.1"]

    def test_filters_empty_entries(self):
        assert opcs4.parse_codes("K40.1,,W37.1,") == ["K40.1", "W37.1"]


class TestLookupCode:
    def test_uncurated_code_returns_none(self):
        assert opcs4.lookup_code("K40.1") is None

    def test_case_insensitive_lookup(self):
        assert opcs4.lookup_code("k40.1") == opcs4.lookup_code("K40.1")

    def test_unknown_code_returns_none(self):
        assert opcs4.lookup_code("ZZ99.9") is None

    def test_empty_code_returns_none(self):
        assert opcs4.lookup_code("") is None


class TestGetClinicalContext:
    def test_uncurated_code_context_contains_code_and_fallback_guidance(self):
        context = opcs4.get_clinical_context("K40.1")
        assert "K40.1" in context
        assert "not found" in context.lower()
        # Procedure-kind systems get perioperative fallback guidance (see
        # registry.get_clinical_context) even with no curated entry.
        assert "pre-operative" in context.lower()

    def test_unknown_code_context_still_contains_code(self):
        context = opcs4.get_clinical_context("ZZ99.9")
        assert "ZZ99.9" in context
        assert "not found" in context.lower()
        assert "procedure code" in context.lower()


class TestInferSpecialty:
    def test_uncurated_code_falls_back_to_chapter_heuristic(self):
        # K has no curated entry, so this falls through to the chapter map.
        assert opcs4.infer_specialty("K40.1") == "Cardiac Surgery"

    def test_unknown_chapter_falls_back_to_chapter_heuristic(self):
        assert opcs4.infer_specialty("D99.9") == "Ear, Nose and Throat Surgery"

    def test_empty_code_falls_back_to_general_surgery(self):
        assert opcs4.infer_specialty("") == "General Surgery"
