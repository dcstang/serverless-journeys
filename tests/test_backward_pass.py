"""Backward-pass tests: once text has been generated, re-check all four
generated parts (patient, admission, journey, notes) to confirm the ICD-10
diagnosis is actually contained in them.

Uses src.processing.check_diagnosis_reflected, the counterpart to the
forward-pass injection tested in test_forward_pass.py.
"""

from __future__ import annotations

import json

from src import processing
from src.codes import icd10, opcs4


class TestCheckDiagnosisReflected:
    def test_flags_all_four_parts_as_reflected_when_present(self):
        patient = {"past_medical_history": ["Prior STEMI in 2020"]}
        admission = {
            "working_diagnosis": (
                "I21.0 - Acute transmural myocardial infarction of anterior wall (STEMI)"
            )
        }
        journey = [{"brief_description": "ED assessment for suspected STEMI, I21.0 confirmed on ECG"}]
        notes = [{"clean_note_text": "Patient admitted with STEMI (I21.0), started on primary PCI pathway."}]

        result = processing.check_diagnosis_reflected("I21.0", patient, admission, journey, notes)

        assert result == {
            "patient": True,
            "admission": True,
            "journey": True,
            "notes": True,
        }

    def test_flags_missing_parts_as_not_reflected(self):
        patient = {"past_medical_history": ["Hypertension"]}
        admission = {"working_diagnosis": "I21.0 - STEMI"}
        journey = [{"brief_description": "Routine ward round"}]
        notes = [{"clean_note_text": "Patient stable overnight."}]

        result = processing.check_diagnosis_reflected("I21.0", patient, admission, journey, notes)

        assert result["patient"] is False
        assert result["admission"] is True
        assert result["journey"] is False
        assert result["notes"] is False

    def test_matches_on_bare_code_without_description_keywords(self):
        admission = {"working_diagnosis": "Confirmed I21.0"}
        result = processing.check_diagnosis_reflected("I21.0", {}, admission, [], [])
        assert result["admission"] is True

    def test_unknown_code_still_matches_on_raw_code(self):
        admission = {"notes": "Diagnosis code ZZ99.9 assigned pending review"}
        result = processing.check_diagnosis_reflected("ZZ99.9", {}, admission, [], [])
        assert result["admission"] is True

    def test_unknown_code_with_no_match_anywhere_is_all_false(self):
        result = processing.check_diagnosis_reflected(
            "ZZ99.9", {"a": "b"}, {"c": "d"}, [{"e": "f"}], [{"g": "h"}]
        )
        assert all(v is False for v in result.values())


class TestCheckProcedureReflected:
    """Mirrors TestCheckDiagnosisReflected for OPCS-4 procedure codes."""

    def test_flags_all_four_parts_as_reflected_when_present(self):
        patient = {"past_medical_and_surgical_history": ["Previous coronary artery bypass grafting in 2019"]}
        admission = {
            "planned_procedure": "K40.1 - Coronary artery bypass grafting using saphenous vein graft"
        }
        journey = [{"brief_description": "Patient taken to theatre for coronary artery bypass grafting, K40.1"}]
        notes = [{"clean_note_text": "Procedure K40.1 (CABG using saphenous vein graft) completed uneventfully."}]

        result = processing.check_procedure_reflected("K40.1", patient, admission, journey, notes)

        assert result == {
            "patient": True,
            "admission": True,
            "journey": True,
            "notes": True,
        }

    def test_flags_missing_parts_as_not_reflected(self):
        patient = {"past_medical_and_surgical_history": ["Hypertension"]}
        admission = {"planned_procedure": "K40.1 - CABG"}
        journey = [{"brief_description": "Routine post-op ward round"}]
        notes = [{"clean_note_text": "Patient recovering well."}]

        result = processing.check_procedure_reflected("K40.1", patient, admission, journey, notes)

        assert result["patient"] is False
        assert result["admission"] is True
        assert result["journey"] is False
        assert result["notes"] is False

    def test_unknown_opcs4_code_still_matches_on_raw_code(self):
        admission = {"notes": "Procedure code ZZ99.9 assigned pending review"}
        result = processing.check_procedure_reflected("ZZ99.9", {}, admission, [], [])
        assert result["admission"] is True


class TestForwardAndBackwardPassIntegration:
    """Full round trip: code -> prompts (forward) -> generated content
    (backward) -> reflection check, all through mocked LLM calls."""

    def test_icd10_code_flows_from_generation_through_to_final_notes(self, monkeypatch, sample_patient):
        code = "J18.1"
        description = icd10.lookup_code(code)["description"]

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            if "Return ONLY a valid JSON array" in prompt:
                return json.dumps(
                    [
                        {
                            "event_type": "ED event",
                            "event_date": "2026-07-08",
                            "event_time": "10:00",
                            "event_order": 1,
                            "brief_description": f"ED assessment for suspected {description}",
                            "clinician_type": "Emergency Physician",
                        }
                    ]
                )
            if "Write the note text only" in prompt:
                return f"Clinical note: patient managed for {code} ({description})."
            return json.dumps(
                {
                    "admission_type": "emergency",
                    "chief_complaint": f"Presents with {description}",
                    "working_diagnosis": f"{code} - {description}",
                    "estimated_los_days": 5,
                }
            )

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        admission = processing.generate_from_codes(
            icd10_codes=[code],
            opcs4_codes=[],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="10:00",
        )
        journey = processing.generate_journey(
            patient=sample_patient,
            admission=admission,
            admission_date="2026-07-08",
            discharge_date="2026-07-12",
            possible_event_types=["ED event", "general ward round"],
        )
        note_text = processing.generate_clinical_note(
            patient=sample_patient,
            admission=admission,
            event=journey[0],
            previous_events=[],
            note_template_str="[Section]\nguidance",
            style_instructions="style",
        )
        notes = [{"clean_note_text": note_text}]

        result = processing.check_diagnosis_reflected(code, sample_patient, admission, journey, notes)

        # Admission, journey, and notes must all carry the diagnosis forward;
        # the patient demographic record has no reason to mention it.
        assert result["admission"] is True
        assert result["journey"] is True
        assert result["notes"] is True

    def test_opcs4_code_flows_from_generation_through_to_final_notes(self, monkeypatch, sample_patient):
        """Same round trip as the ICD-10 test above, but for a procedure code
        driven through the opcs4-only prompt path and checked with
        check_procedure_reflected instead of check_diagnosis_reflected."""
        code = "K40.1"
        description = opcs4.lookup_code(code)["description"]

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            if "Return ONLY a valid JSON array" in prompt:
                return json.dumps(
                    [
                        {
                            "event_type": "operation",
                            "event_date": "2026-07-08",
                            "event_time": "08:00",
                            "event_order": 1,
                            "brief_description": f"Patient undergoes {description}",
                            "clinician_type": "Consultant Cardiac Surgeon",
                        }
                    ]
                )
            if "Write the note text only" in prompt:
                return f"Operation note: {code} ({description}) performed uneventfully."
            return json.dumps(
                {
                    "admission_type": "elective",
                    "planned_procedure": description,
                    "indication": f"Elective {description}",
                    "estimated_los_days": 8,
                }
            )

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        admission = processing.generate_from_codes(
            icd10_codes=[],
            opcs4_codes=[code],
            patient_details=sample_patient,
            admission_date="2026-07-08",
            admission_time="08:00",
        )
        journey = processing.generate_journey(
            patient=sample_patient,
            admission=admission,
            admission_date="2026-07-08",
            discharge_date="2026-07-16",
            possible_event_types=["operation", "post-anaesthesia recovery"],
        )
        note_text = processing.generate_clinical_note(
            patient=sample_patient,
            admission=admission,
            event=journey[0],
            previous_events=[],
            note_template_str="[Section]\nguidance",
            style_instructions="style",
        )
        notes = [{"clean_note_text": note_text}]

        result = processing.check_procedure_reflected(code, sample_patient, admission, journey, notes)

        assert result["admission"] is True
        assert result["journey"] is True
        assert result["notes"] is True
