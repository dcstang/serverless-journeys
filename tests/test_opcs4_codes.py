"""Unit tests for src/codes/opcs4.py: parsing, lookup, and context generation.

Mirrors tests/test_icd10_codes.py for the OPCS-4 procedure code module.
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
    def test_known_code_returns_metadata(self):
        info = opcs4.lookup_code("K40.1")
        assert info is not None
        assert info["description"] == "Coronary artery bypass grafting using saphenous vein graft"
        assert info["surgical_specialty"] == "Cardiac Surgery"

    def test_case_insensitive_lookup(self):
        assert opcs4.lookup_code("k40.1") == opcs4.lookup_code("K40.1")

    def test_unknown_code_returns_none(self):
        assert opcs4.lookup_code("ZZ99.9") is None

    def test_empty_code_returns_none(self):
        assert opcs4.lookup_code("") is None


class TestGetClinicalContext:
    def test_known_code_context_contains_code_and_description(self):
        context = opcs4.get_clinical_context("K40.1")
        assert "K40.1" in context
        assert "Coronary artery bypass grafting" in context
        assert "Cardiac Surgery" in context

    def test_day_case_procedure_notes_no_overnight_stay(self):
        context = opcs4.get_clinical_context("H41.1")  # 0-1 day typical LOS
        assert "day case" in context.lower()

    def test_unknown_code_context_still_contains_code(self):
        context = opcs4.get_clinical_context("ZZ99.9")
        assert "ZZ99.9" in context
        assert "OPCS-4 surgical/interventional procedure code" in context


class TestInferSpecialty:
    def test_known_code_uses_curated_specialty(self):
        assert opcs4.infer_specialty("K40.1") == "Cardiac Surgery"

    def test_unknown_code_falls_back_to_chapter_heuristic(self):
        assert opcs4.infer_specialty("D99.9") == "Ear, Nose and Throat Surgery"

    def test_empty_code_falls_back_to_general_surgery(self):
        assert opcs4.infer_specialty("") == "General Surgery"
