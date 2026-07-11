"""Tests for the targeted corrective retry step (main.step_verify_code_reflection
with max_correction_attempts > 0) and its wiring into main.run_pipeline,
plus the --research-unknown-codes forward-pass grounding path.

Complements tests/test_pipeline_reflection.py, which covers the plain
check-only backward pass. These tests prove the *correction* actually
fires, actually stops early once a code is reflected, and can be disabled.
"""

from __future__ import annotations

import json

import pytest

import main
from src import processing
from src.codes import research as code_research

@pytest.fixture(autouse=True)
def _clear_research_cache():
    code_research.clear_cache()
    yield
    code_research.clear_cache()

# Prompt markers used to distinguish which template a mocked call_llm call
# is responding to - mirrors the marker-dispatch pattern used throughout
# the rest of the test suite (see test_backward_pass.py).
_PATIENT_MARKER = "no resemblance to any real person"
_JOURNEY_MARKER = "Return ONLY a valid JSON array"
_NOTE_MARKER = "Write the note text only"
_CORRECT_ADMISSION_MARKER = "revising a synthetic admission record"
_CORRECT_NOTE_MARKER = "revising a synthetic clinical note"


def _fake_patient_response():
    return json.dumps(
        {
            "full_name": "Test Patient",
            "first_name": "Test",
            "surname": "Patient",
            "age": 60,
            "sex": "Male",
            "nhs_number": "123 456 7890",
            "mrn": "1234567",
        }
    )


def _fake_journey_response(mention_code: bool, code: str, description: str):
    description_part = f" for suspected {description}" if mention_code else ""
    return json.dumps(
        [
            {
                "event_type": "ED event",
                "event_date": "2026-07-08",
                "event_time": "10:00",
                "event_order": 1,
                "brief_description": f"ED assessment{description_part}",
                "clinician_type": "Emergency Physician",
            }
        ]
    )


def _make_correction_capable_fake_call_llm(code: str, description: str, *, call_log: list[str]):
    """Fake call_llm: initial generation never mentions the code, but a
    correction prompt (admission or note) always successfully incorporates
    it - lets tests exercise "miss then get fixed" deterministically."""

    def fake(prompt, model=None, temp=0.7, **kwargs):
        if _PATIENT_MARKER in prompt:
            call_log.append("patient")
            return _fake_patient_response()
        if _JOURNEY_MARKER in prompt:
            call_log.append("journey")
            return _fake_journey_response(mention_code=False, code=code, description=description)
        if _CORRECT_ADMISSION_MARKER in prompt:
            call_log.append("correct_admission")
            return json.dumps(
                {
                    "chief_complaint": f"Presents with {description}",
                    "working_diagnosis": f"{code} - {description}",
                }
            )
        if _CORRECT_NOTE_MARKER in prompt:
            call_log.append("correct_note")
            return f"Clinical note: patient managed for {code} ({description})."
        if _NOTE_MARKER in prompt:
            call_log.append("note")
            return "Clinical note: patient reviewed, stable, plan continued."
        # Initial admission generation - deliberately generic, no code.
        call_log.append("admission")
        return json.dumps(
            {
                "admission_type": "emergency",
                "chief_complaint": "Generic presentation",
                "working_diagnosis": "Under investigation",
                "estimated_los_days": 5,
            }
        )

    return fake


def _make_never_reflects_fake_call_llm(code: str, description: str, *, call_log: list[str]):
    """Fake call_llm where even correction prompts fail to incorporate the
    code - used to prove bounded retries actually stop, not loop forever."""

    def fake(prompt, model=None, temp=0.7, **kwargs):
        if _PATIENT_MARKER in prompt:
            return _fake_patient_response()
        if _JOURNEY_MARKER in prompt:
            return _fake_journey_response(mention_code=False, code=code, description=description)
        if _CORRECT_ADMISSION_MARKER in prompt:
            call_log.append("correct_admission")
            return json.dumps({"chief_complaint": "Still generic"})
        if _CORRECT_NOTE_MARKER in prompt:
            call_log.append("correct_note")
            return "Still generic note text."
        if _NOTE_MARKER in prompt:
            return "Clinical note: patient reviewed, stable, plan continued."
        return json.dumps(
            {
                "admission_type": "emergency",
                "chief_complaint": "Generic presentation",
                "working_diagnosis": "Under investigation",
                "estimated_los_days": 5,
            }
        )

    return fake


class TestCorrectionFixesAMiss:
    def test_correction_makes_previously_unreflected_code_pass(self, tmp_path, monkeypatch):
        code = "J18.1"
        description = "Lobar pneumonia, unspecified organism"
        call_log: list[str] = []

        monkeypatch.setattr(
            processing, "call_llm", _make_correction_capable_fake_call_llm(code, description, call_log=call_log)
        )

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "1",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        # The correction prompts must actually have fired.
        assert "correct_admission" in call_log
        assert "correct_note" in call_log

        summary = json.loads((tmp_path / "generation_summary.json").read_text())
        check = summary["code_reflection_check"]
        assert check["n_codes_checked"] == 1
        assert check["n_codes_reflected"] == 1
        assert check["unreflected_codes"] == []

    def test_notes_csv_reflects_the_corrected_text(self, tmp_path, monkeypatch):
        code = "J18.1"
        description = "Lobar pneumonia, unspecified organism"
        call_log: list[str] = []

        monkeypatch.setattr(
            processing, "call_llm", _make_correction_capable_fake_call_llm(code, description, call_log=call_log)
        )

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "1",
            ]
        )
        main.run_pipeline(args)

        notes_csv = (tmp_path / "synthetic_clinical_notes.csv").read_text()
        assert code in notes_csv


class TestCorrectionCanBeDisabled:
    def test_zero_max_attempts_never_calls_correction_prompts(self, tmp_path, monkeypatch):
        code = "J18.1"
        description = "Lobar pneumonia, unspecified organism"
        call_log: list[str] = []

        monkeypatch.setattr(
            processing, "call_llm", _make_correction_capable_fake_call_llm(code, description, call_log=call_log)
        )

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "0",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        assert "correct_admission" not in call_log
        assert "correct_note" not in call_log

        summary = json.loads((tmp_path / "generation_summary.json").read_text())
        check = summary["code_reflection_check"]
        assert check["n_codes_reflected"] == 0
        assert check["unreflected_codes"] == [code]

    def test_default_max_attempts_is_one(self, tmp_path):
        """--max-correction-attempts defaults to 1 (correction on by default),
        matching the .env.example / docstring documentation."""
        args = main.parse_args(
            ["--diagnostic-codes", "I21.0", "--output-dir", str(tmp_path)]
        )
        assert args.max_correction_attempts == 1


class TestCorrectionStopsEarly:
    def test_does_not_attempt_a_second_correction_once_reflected(self, tmp_path, monkeypatch):
        code = "J18.1"
        description = "Lobar pneumonia, unspecified organism"
        call_log: list[str] = []

        monkeypatch.setattr(
            processing, "call_llm", _make_correction_capable_fake_call_llm(code, description, call_log=call_log)
        )

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "3",  # generously allow up to 3
            ]
        )
        main.run_pipeline(args)

        # Correction succeeds on the first attempt, so only one round of
        # correct_admission/correct_note calls should have happened even
        # though 3 attempts were allowed.
        assert call_log.count("correct_admission") == 1
        assert call_log.count("correct_note") == 1

    def test_bounded_retries_stop_even_when_never_reflected(self, tmp_path, monkeypatch):
        code = "J18.1"
        description = "Lobar pneumonia, unspecified organism"
        call_log: list[str] = []

        monkeypatch.setattr(
            processing, "call_llm", _make_never_reflects_fake_call_llm(code, description, call_log=call_log)
        )

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "2",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0  # a still-unreflected code doesn't fail the run

        assert call_log.count("correct_admission") == 2
        assert call_log.count("correct_note") == 2

        summary = json.loads((tmp_path / "generation_summary.json").read_text())
        assert summary["code_reflection_check"]["unreflected_codes"] == [code]


class TestResearchUnknownCodesGroundsGeneration:
    def test_uncurated_code_context_comes_from_search_when_enabled(
        self, tmp_path, monkeypatch, fake_search_endpoint
    ):
        code = "ZZ99.9"  # not in the curated ICD-10 dictionary
        fake_search_endpoint.set_items(
            [{"title": code, "snippet": "Rare fictitious research condition", "link": "https://example.com"}]
        )

        captured_admission_prompt = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            if _PATIENT_MARKER in prompt:
                return _fake_patient_response()
            if _JOURNEY_MARKER in prompt:
                return _fake_journey_response(mention_code=False, code=code, description="")
            if _NOTE_MARKER in prompt:
                return "Clinical note text."
            if "clinical coding specialist" in prompt:
                # research_code_prompt synthesis call
                return json.dumps(
                    {
                        "description": "Researched Rare Condition",
                        "specialty": "General Medicine",
                        "type": "emergency",
                        "typical_los_days_min": 2,
                        "typical_los_days_max": 4,
                        "confidence": "high",
                    }
                )
            # Admission generation - capture it to check research grounding made it in.
            captured_admission_prompt["prompt"] = prompt
            return json.dumps({"admission_type": "emergency", "estimated_los_days": 3})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--research-unknown-codes",
                "--max-correction-attempts", "0",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        prompt = captured_admission_prompt["prompt"]
        assert "Researched Rare Condition" in prompt
        assert "Web-researched" in prompt

    def test_research_disabled_by_default_uses_generic_fallback(self, tmp_path, monkeypatch):
        code = "ZZ99.9"
        captured_admission_prompt = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            if _PATIENT_MARKER in prompt:
                return _fake_patient_response()
            if _JOURNEY_MARKER in prompt:
                return _fake_journey_response(mention_code=False, code=code, description="")
            if _NOTE_MARKER in prompt:
                return "Clinical note text."
            captured_admission_prompt["prompt"] = prompt
            return json.dumps({"admission_type": "emergency", "estimated_los_days": 3})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "0",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        prompt = captured_admission_prompt["prompt"]
        assert "not found in reference dictionary" in prompt
        assert "Web-researched" not in prompt
