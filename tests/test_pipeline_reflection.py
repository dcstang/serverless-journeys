"""Proves the backward-pass reflection check is wired into the actual
generation pipeline (main.py), not just callable in isolation from tests.

Runs main.run_pipeline() end-to-end with call_llm mocked (no real network
or API keys needed), then inspects the generated admission records and
generation_summary.json for the code_reflection_check results that
main.step_verify_code_reflection is supposed to attach.
"""

from __future__ import annotations

import json

import pytest

import main
from src import processing


def _make_fake_call_llm(code: str, description: str, *, mention_code: bool):
    """Build a fake call_llm that dispatches on prompt content, mirroring
    the real prompt structure (patient / admission / journey / note)."""

    def fake(prompt, model=None, temp=0.7, **kwargs):
        if "no resemblance to any real person" in prompt:
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
        if "Return ONLY a valid JSON array" in prompt:
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
        if "Write the note text only" in prompt:
            if mention_code:
                return f"Clinical note: patient managed for {code} ({description})."
            return "Clinical note: patient reviewed, stable, plan continued."
        # Admission generation
        if mention_code:
            return json.dumps(
                {
                    "admission_type": "emergency",
                    "chief_complaint": f"Presents with {description}",
                    "working_diagnosis": f"{code} - {description}",
                    "estimated_los_days": 5,
                }
            )
        return json.dumps(
            {
                "admission_type": "emergency",
                "chief_complaint": "Generic presentation",
                "working_diagnosis": "Under investigation",
                "estimated_los_days": 5,
            }
        )

    return fake


class TestReflectionCheckWiredIntoRealPipeline:
    def test_reflected_code_is_recorded_in_admission_and_summary(self, tmp_path, monkeypatch):
        code = "I21.0"
        description = "Acute transmural myocardial infarction of anterior wall (STEMI)"

        monkeypatch.setattr(processing, "call_llm", _make_fake_call_llm(code, description, mention_code=True))

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--llm-provider", "anthropic",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        admissions_path = tmp_path / "synthetic_admissions.csv"
        assert admissions_path.exists()
        admissions_csv = admissions_path.read_text()
        assert "code_reflection_check" in admissions_csv
        # The reflection dict is JSON-serialised into the CSV cell; the code
        # and a True result must both be present in that cell's content.
        assert code in admissions_csv

        summary = json.loads((tmp_path / "generation_summary.json").read_text())
        check = summary["code_reflection_check"]
        assert check["n_codes_checked"] == 1
        assert check["n_codes_reflected"] == 1
        assert check["unreflected_codes"] == []

    def test_unreflected_code_is_flagged_in_summary_and_logged(self, tmp_path, monkeypatch, caplog):
        code = "J18.1"
        description = "Lobar pneumonia, unspecified organism"

        monkeypatch.setattr(processing, "call_llm", _make_fake_call_llm(code, description, mention_code=False))

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--llm-provider", "anthropic",
            ]
        )
        with caplog.at_level("WARNING"):
            exit_code = main.run_pipeline(args)
        assert exit_code == 0

        summary = json.loads((tmp_path / "generation_summary.json").read_text())
        check = summary["code_reflection_check"]
        assert check["n_codes_checked"] == 1
        assert check["n_codes_reflected"] == 0
        assert check["unreflected_codes"] == [code]

        assert any(code in record.message for record in caplog.records)

    def test_reflection_check_is_skipped_in_test_mode(self, tmp_path, monkeypatch):
        """Test mode uses stub data with no clinical content, so the
        reflection check should not run (and shouldn't crash trying)."""
        args = main.parse_args(
            [
                "--test-mode",
                "--diagnostic-codes", "I21.0",
                "--output-dir", str(tmp_path),
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        summary = json.loads((tmp_path / "generation_summary.json").read_text())
        assert "code_reflection_check" not in summary

    def test_unknown_code_system_fails_cleanly(self, tmp_path):
        """An invalid --diagnostic-code-system should fail fast with a clear
        error, not an unhandled traceback."""
        args = main.parse_args(
            [
                "--test-mode",
                "--diagnostic-codes", "I21.0",
                "--diagnostic-code-system", "not-a-real-system",
                "--output-dir", str(tmp_path),
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 1

    def test_icd10_codes_flag_no_longer_exists(self, tmp_path):
        """--icd10-codes/--opcs4-codes were removed in favour of the generic
        --diagnostic-codes/--procedure-codes flags - no silent alias."""
        with pytest.raises(SystemExit):
            main.parse_args(
                [
                    "--test-mode",
                    "--icd10-codes", "I21.0",
                    "--output-dir", str(tmp_path),
                ]
            )


class TestNEventsPerPatientReachesJourneyPrompt:
    def test_default_flag_value_is_eight(self, tmp_path):
        args = main.parse_args(["--output-dir", str(tmp_path)])
        assert args.n_events_per_patient == 8

    def test_custom_value_reaches_the_journey_prompt(self, tmp_path, monkeypatch):
        code = "I21.0"
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kwargs):
            if "no resemblance to any real person" in prompt:
                return json.dumps(
                    {
                        "full_name": "Test Patient", "first_name": "Test", "surname": "Patient",
                        "age": 60, "sex": "Male", "nhs_number": "123 456 7890", "mrn": "1234567",
                    }
                )
            if "Return ONLY a valid JSON array" in prompt:
                captured["journey_prompt"] = prompt
                return json.dumps([])
            if "Write the note text only" in prompt:
                return "Clinical note text."
            return json.dumps({"admission_type": "emergency", "estimated_los_days": 5})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--n-events-per-patient", "15",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "0",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        assert "TARGET NUMBER OF EVENTS: approximately 15" in captured["journey_prompt"]
