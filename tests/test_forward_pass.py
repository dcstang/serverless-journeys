"""Forward-pass tests: diagnostic/procedure codes are looked up and pushed
into the generation prompts (admission -> journey -> notes) before any LLM
call.

Every test here mocks src.processing.call_llm and asserts on the *prompt
text actually sent*, not just the function's return value - this is what
proves the codes are "sent through" rather than silently dropped. Uses the
built-in ICD-10 ('icd10') and OPCS-4 ('opcs4') registered code systems, but
generate_from_codes itself is system-agnostic - see test_code_registry.py
for coverage of a third, made-up code system.
"""

from __future__ import annotations

import json

from src import processing
from src.codes import icd10, opcs4


class TestForwardPassSingleDiagnosticCode:
    def test_code_and_description_are_injected_into_prompt(self, monkeypatch, sample_patient):
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"admission_type": "emergency", "estimated_los_days": 5})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        processing.generate_from_codes(
            diagnostic_codes=["I21.0"],
            procedure_codes=[],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
        )

        prompt = captured["prompt"]
        assert "I21.0" in prompt
        assert "Acute transmural myocardial infarction of anterior wall (STEMI)" in prompt

    def test_admission_dict_carries_input_codes_and_system_forward(self, monkeypatch, sample_patient):
        monkeypatch.setattr(
            processing,
            "call_llm",
            lambda prompt, model=None, temp=0.7, **kw: json.dumps({"estimated_los_days": 5}),
        )

        admission = processing.generate_from_codes(
            diagnostic_codes=["I21.0"],
            procedure_codes=[],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
        )

        assert admission["diagnostic_codes"] == ["I21.0"]
        assert admission["diagnostic_code_system"] == "icd10"
        assert admission["procedure_codes"] == []
        assert admission["procedure_code_system"] is None

    def test_unparseable_llm_response_still_carries_codes_forward(self, monkeypatch, sample_patient):
        monkeypatch.setattr(
            processing,
            "call_llm",
            lambda prompt, model=None, temp=0.7, **kw: "not valid json",
        )

        admission = processing.generate_from_codes(
            diagnostic_codes=["I21.0", "J18.1"],
            procedure_codes=[],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
        )

        assert admission["diagnostic_codes"] == ["I21.0", "J18.1"]
        assert "parse_error" in admission


class TestForwardPassMultiCode:
    def test_all_diagnostic_and_procedure_codes_appear_in_prompt(self, monkeypatch, sample_patient):
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"admission_type": "emergency", "estimated_los_days": 5})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        processing.generate_from_codes(
            diagnostic_codes=["I21.0", "J18.1"],
            procedure_codes=["K40.1"],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
        )

        prompt = captured["prompt"]
        assert "I21.0" in prompt
        assert "J18.1" in prompt
        assert "K40.1" in prompt
        # System names must be identified explicitly, not assumed.
        assert "ICD-10" in prompt
        assert "OPCS-4" in prompt

    def test_unknown_code_is_still_sent_through(self, monkeypatch, sample_patient):
        """Even codes absent from the curated dictionary must reach the prompt."""
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"admission_type": "emergency", "estimated_los_days": 5})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        assert icd10.lookup_code("ZZ99.9") is None  # sanity check: genuinely unknown

        admission = processing.generate_from_codes(
            diagnostic_codes=["ZZ99.9"],
            procedure_codes=[],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
        )

        assert "ZZ99.9" in captured["prompt"]
        assert admission["diagnostic_codes"] == ["ZZ99.9"]


class TestForwardPassProcedureOnly:
    """The procedure-only path exercises generate_from_codes with an empty
    diagnostic_codes list, so it needs its own forward-pass coverage rather
    than piggybacking on the diagnostic-code tests."""

    def test_code_and_description_are_injected_into_prompt(self, monkeypatch, sample_patient):
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"admission_type": "elective", "estimated_los_days": 8})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        processing.generate_from_codes(
            diagnostic_codes=[],
            procedure_codes=["K40.1"],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
        )

        prompt = captured["prompt"]
        assert "K40.1" in prompt
        assert "Coronary artery bypass grafting using saphenous vein graft" in prompt

    def test_admission_dict_carries_input_codes_and_system_forward(self, monkeypatch, sample_patient):
        monkeypatch.setattr(
            processing,
            "call_llm",
            lambda prompt, model=None, temp=0.7, **kw: json.dumps({"estimated_los_days": 8}),
        )

        admission = processing.generate_from_codes(
            diagnostic_codes=[],
            procedure_codes=["K40.1"],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
        )

        assert admission["procedure_codes"] == ["K40.1"]
        assert admission["procedure_code_system"] == "opcs4"
        assert admission["diagnostic_codes"] == []
        assert admission["diagnostic_code_system"] is None

    def test_unknown_procedure_code_is_still_sent_through(self, monkeypatch, sample_patient):
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            captured["prompt"] = prompt
            return json.dumps({"estimated_los_days": 3})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        assert opcs4.lookup_code("ZZ99.9") is None  # sanity check: genuinely unknown

        admission = processing.generate_from_codes(
            diagnostic_codes=[],
            procedure_codes=["ZZ99.9"],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="09:00",
        )

        assert "ZZ99.9" in captured["prompt"]
        assert admission["procedure_codes"] == ["ZZ99.9"]


class TestForwardPassPropagatesIntoJourneyAndNotes:
    """The diagnosis must keep flowing forward past the admission step."""

    def test_admission_diagnosis_appears_in_journey_prompt(self, monkeypatch, sample_patient):
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            captured["prompt"] = prompt
            return json.dumps([])

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        admission = {"diagnostic_codes": ["I21.0"], "working_diagnosis": "I21.0 - STEMI"}

        processing.generate_journey(
            patient=sample_patient,
            admission=admission,
            admission_date="2026-07-08",
            discharge_date="2026-07-12",
            possible_event_types=["ED event", "general ward round"],
        )

        assert "I21.0" in captured["prompt"]

    def test_admission_diagnosis_appears_in_note_prompt(self, monkeypatch, sample_patient):
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            captured["prompt"] = prompt
            return "Generated note text"

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        admission = {"diagnostic_codes": ["I21.0"], "working_diagnosis": "I21.0 - STEMI"}
        event = {"event_type": "ED event", "event_date": "2026-07-08", "event_time": "09:00"}

        processing.generate_clinical_note(
            patient=sample_patient,
            admission=admission,
            event=event,
            previous_events=[],
            note_template_str="[Section]\nguidance",
            style_instructions="style",
        )

        assert "I21.0" in captured["prompt"]
