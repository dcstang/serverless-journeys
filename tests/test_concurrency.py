"""Tests for --concurrency: fanning patient generation out across a thread
pool (main.py's _generate_one_patient / ThreadPoolExecutor wiring), while
keeping output ordering and failure isolation identical to the sequential
(--concurrency 1, default) path.
"""

from __future__ import annotations

import json
import threading

import main
from src import processing

_PATIENT_MARKER = "no resemblance to any real person"
_JOURNEY_MARKER = "Return ONLY a valid JSON array"
_NOTE_MARKER = "Write the note text only"


def _fake_patient_response(patient_idx: int):
    return json.dumps(
        {
            "full_name": f"Test Patient {patient_idx}",
            "first_name": "Test",
            "surname": f"Patient{patient_idx}",
            "age": 60,
            "sex": "Male",
            "nhs_number": f"123 456 {7890 + patient_idx}",
            "mrn": f"123456{patient_idx}",
        }
    )


def _fake_journey_response():
    return json.dumps(
        [
            {
                "event_type": "ED event", "event_date": "2026-07-08", "event_time": "10:00",
                "event_order": 1, "brief_description": "ED assessment", "clinician_type": "Doctor",
            }
        ]
    )


class TestConcurrentGeneration:
    def test_concurrency_produces_same_patient_count_as_sequential(self, tmp_path, monkeypatch):
        call_lock = threading.Lock()
        max_concurrent = {"value": 0}
        in_flight = {"value": 0}

        def fake(prompt, model=None, temp=0.7, **kwargs):
            with call_lock:
                in_flight["value"] += 1
                max_concurrent["value"] = max(max_concurrent["value"], in_flight["value"])
            try:
                if _PATIENT_MARKER in prompt:
                    return _fake_patient_response(in_flight["value"])
                if _JOURNEY_MARKER in prompt:
                    return _fake_journey_response()
                if _NOTE_MARKER in prompt:
                    return "Patient reviewed, stable."
                return json.dumps({"admission_type": "emergency", "estimated_los_days": 5})
            finally:
                with call_lock:
                    in_flight["value"] -= 1

        monkeypatch.setattr(processing, "call_llm", fake)

        args = main.parse_args(
            [
                "--n-patients", "4",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "0",
                "--concurrency", "4",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        patients_csv = (tmp_path / "synthetic_patients.csv").read_text()
        # header + 4 patient rows
        assert len(patients_csv.strip().splitlines()) == 5

    def test_concurrency_output_order_matches_sequential(self, tmp_path, monkeypatch):
        """Output row order must be deterministic (patient_idx order),
        regardless of which worker thread finishes first."""

        def fake(prompt, model=None, temp=0.7, **kwargs):
            if _PATIENT_MARKER in prompt:
                return _fake_patient_response(0)
            if _JOURNEY_MARKER in prompt:
                return _fake_journey_response()
            if _NOTE_MARKER in prompt:
                return "Patient reviewed, stable."
            return json.dumps({"admission_type": "emergency", "estimated_los_days": 5})

        monkeypatch.setattr(processing, "call_llm", fake)

        sequential_args = main.parse_args(
            [
                "--n-patients", "3",
                "--output-dir", str(tmp_path / "sequential"),
                "--max-correction-attempts", "0",
            ]
        )
        concurrent_args = main.parse_args(
            [
                "--n-patients", "3",
                "--output-dir", str(tmp_path / "concurrent"),
                "--max-correction-attempts", "0",
                "--concurrency", "3",
            ]
        )

        assert main.run_pipeline(sequential_args) == 0
        assert main.run_pipeline(concurrent_args) == 0

        # Row order must match (patient_idx order); UUIDs differ per run so
        # compare everything but the trailing clinical_note_id/id column.
        def _rows_without_ids(csv_text: str) -> list[str]:
            lines = csv_text.strip().splitlines()
            return [",".join(line.split(",")[:-1]) for line in lines]

        sequential_csv = (tmp_path / "sequential" / "synthetic_journeys.csv").read_text()
        concurrent_csv = (tmp_path / "concurrent" / "synthetic_journeys.csv").read_text()
        assert _rows_without_ids(sequential_csv) == _rows_without_ids(concurrent_csv)

    def test_one_patient_failure_does_not_abort_others_under_concurrency(self, tmp_path, monkeypatch):
        def fake(prompt, model=None, temp=0.7, **kwargs):
            if _PATIENT_MARKER in prompt:
                # Every other patient-generation call fails outright.
                if not hasattr(fake, "_count"):
                    fake._count = 0
                fake._count += 1
                if fake._count % 2 == 0:
                    raise RuntimeError("simulated LLM failure")
                return _fake_patient_response(fake._count)
            if _JOURNEY_MARKER in prompt:
                return _fake_journey_response()
            if _NOTE_MARKER in prompt:
                return "Patient reviewed, stable."
            return json.dumps({"admission_type": "emergency", "estimated_los_days": 5})

        monkeypatch.setattr(processing, "call_llm", fake)

        args = main.parse_args(
            [
                "--n-patients", "4",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "0",
                "--concurrency", "4",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        patients_csv = (tmp_path / "synthetic_patients.csv").read_text()
        # header + 2 successful patient rows (the other 2 failed and were skipped)
        assert len(patients_csv.strip().splitlines()) == 3

    def test_test_mode_forces_sequential_regardless_of_concurrency_flag(self, tmp_path):
        args = main.parse_args(
            [
                "--test-mode",
                "--output-dir", str(tmp_path),
                "--concurrency", "8",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        patients_csv = (tmp_path / "synthetic_patients.csv").read_text()
        assert len(patients_csv.strip().splitlines()) == 2  # header + 1 (test mode forces n_patients=1)
