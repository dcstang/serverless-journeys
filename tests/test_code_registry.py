"""Tests for src/codes/registry.py: the generic diagnostic/procedure code
system abstraction.

ICD-10 and OPCS-4 are just the two built-in registered systems (see
src/codes/icd10.py / src/codes/opcs4.py). These tests exercise the
registry directly and prove genuine pluggability by registering a third,
made-up code system and running it through the same generic functions the
built-in systems use - nothing pipeline code does is ICD-10/OPCS-4-specific.
"""

from __future__ import annotations

import pytest

from src.codes import icd10, opcs4, registry


class TestBuiltInSystemsAreRegistered:
    def test_icd10_is_registered(self):
        system = registry.get_code_system("icd10")
        assert system.name == "ICD-10"
        assert system.kind == "diagnostic"

    def test_opcs4_is_registered(self):
        system = registry.get_code_system("opcs4")
        assert system.name == "OPCS-4"
        assert system.kind == "procedure"

    def test_lookup_is_case_insensitive_on_key(self):
        assert registry.get_code_system("ICD10") is registry.get_code_system("icd10")

    def test_unknown_system_raises_key_error(self):
        with pytest.raises(KeyError):
            registry.get_code_system("not-a-real-system")

    def test_list_code_systems_filters_by_kind(self):
        assert "icd10" in registry.list_code_systems(kind="diagnostic")
        assert "opcs4" not in registry.list_code_systems(kind="diagnostic")
        assert "opcs4" in registry.list_code_systems(kind="procedure")
        assert "icd10" not in registry.list_code_systems(kind="procedure")


class TestGenericFunctionsMatchModuleWrappers:
    """icd10.py/opcs4.py's public functions should be thin wrappers - prove
    they agree with calling the registry directly."""

    def test_icd10_lookup_matches_registry(self):
        system = registry.get_code_system("icd10")
        assert icd10.lookup_code("I21.0") == registry.lookup_code(system, "I21.0")

    def test_opcs4_context_matches_registry(self):
        system = registry.get_code_system("opcs4")
        assert opcs4.get_clinical_context("K40.1") == registry.get_clinical_context(system, "K40.1")


class TestThirdPartyCodeSystemPluggability:
    """Register a made-up third code system to prove the pipeline isn't
    hardwired to ICD-10/OPCS-4 - any standard that fits the CodeSystem shape
    works identically."""

    @pytest.fixture(autouse=True)
    def _register_demo_system(self):
        demo_system = registry.CodeSystem(
            key="demo-snomed",
            name="Demo SNOMED CT",
            kind="diagnostic",
            codes={
                "22298006": {
                    "description": "Myocardial infarction",
                    "specialty": "Cardiology",
                    "admission_type": "emergency",
                    "typical_los_days": (4, 7),
                },
            },
            specialty_field="specialty",
            type_field="admission_type",
            chapter_map={},
            default_specialty="General Medicine",
        )
        registry.register_code_system(demo_system)
        yield
        # Registry is a process-wide singleton; leave it as we found it so
        # other tests don't see this demo system.
        registry._REGISTRY.pop("demo-snomed", None)

    def test_demo_system_is_retrievable(self):
        system = registry.get_code_system("demo-snomed")
        assert system.name == "Demo SNOMED CT"
        assert system.kind == "diagnostic"

    def test_demo_system_curated_lookup_works(self):
        system = registry.get_code_system("demo-snomed")
        info = registry.lookup_code(system, "22298006")
        assert info["description"] == "Myocardial infarction"
        assert registry.infer_specialty(system, "22298006") == "Cardiology"

    def test_demo_system_context_contains_code_and_description(self):
        system = registry.get_code_system("demo-snomed")
        context = registry.get_clinical_context(system, "22298006")
        assert "22298006" in context
        assert "Myocardial infarction" in context
        assert "Demo SNOMED CT" in context

    def test_demo_system_handles_entirely_uncurated_code(self):
        """A code not in the curated dict must still resolve gracefully -
        this is what makes an uncurated/unfamiliar standard usable at all."""
        system = registry.get_code_system("demo-snomed")
        assert registry.lookup_code(system, "99999999") is None
        context = registry.get_clinical_context(system, "99999999")
        assert "99999999" in context
        assert "not found" in context.lower()
        assert registry.infer_specialty(system, "99999999") == "General Medicine"

    def test_generate_from_codes_works_with_demo_system(self, monkeypatch, sample_patient):
        """The forward pass (generate_from_codes) must work for a code
        system it has never seen before, purely via the registry."""
        from src import processing

        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            captured["prompt"] = prompt
            return '{"admission_type": "emergency", "estimated_los_days": 5}'

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        admission = processing.generate_from_codes(
            diagnostic_codes=["22298006"],
            procedure_codes=[],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
            diagnostic_code_system="demo-snomed",
        )

        assert "22298006" in captured["prompt"]
        assert "Myocardial infarction" in captured["prompt"]
        assert "Demo SNOMED CT" in captured["prompt"]
        assert admission["diagnostic_codes"] == ["22298006"]
        assert admission["diagnostic_code_system"] == "demo-snomed"
