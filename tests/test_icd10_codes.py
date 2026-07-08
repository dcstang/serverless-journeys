"""Unit tests for src/codes/icd10.py: parsing, lookup, and context generation."""

from __future__ import annotations

from src.codes import icd10


class TestParseCodes:
    def test_parses_comma_separated_codes(self):
        assert icd10.parse_codes("I21.0, J18.1,K35.2") == ["I21.0", "J18.1", "K35.2"]

    def test_empty_string_returns_empty_list(self):
        assert icd10.parse_codes("") == []
        assert icd10.parse_codes("   ") == []

    def test_uppercases_and_strips_codes(self):
        assert icd10.parse_codes(" i21.0 , j18.1 ") == ["I21.0", "J18.1"]

    def test_filters_empty_entries(self):
        assert icd10.parse_codes("I21.0,,J18.1,") == ["I21.0", "J18.1"]


class TestLookupCode:
    def test_known_code_returns_metadata(self):
        info = icd10.lookup_code("I21.0")
        assert info is not None
        assert info["description"] == (
            "Acute transmural myocardial infarction of anterior wall (STEMI)"
        )
        assert info["specialty"] == "Cardiology"

    def test_case_insensitive_lookup(self):
        assert icd10.lookup_code("i21.0") == icd10.lookup_code("I21.0")

    def test_unknown_code_returns_none(self):
        assert icd10.lookup_code("ZZ99.9") is None

    def test_empty_code_returns_none(self):
        assert icd10.lookup_code("") is None


class TestGetClinicalContext:
    def test_known_code_context_contains_code_and_description(self):
        context = icd10.get_clinical_context("I21.0")
        assert "I21.0" in context
        assert "Acute transmural myocardial infarction" in context
        assert "Cardiology" in context

    def test_unknown_code_context_still_contains_code(self):
        context = icd10.get_clinical_context("ZZ99.9")
        assert "ZZ99.9" in context
        assert "not found" in context.lower()


class TestInferSpecialty:
    def test_known_code_uses_curated_specialty(self):
        assert icd10.infer_specialty("I21.0") == "Cardiology"

    def test_unknown_code_falls_back_to_chapter_heuristic(self):
        assert icd10.infer_specialty("J99.9") == "Respiratory Medicine"

    def test_empty_code_falls_back_to_general_medicine(self):
        assert icd10.infer_specialty("") == "General Medicine"
