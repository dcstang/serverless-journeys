"""Tests for the end-of-run quality evaluation metrics
(calculate_readability_metrics/calculate_fluency/calculate_groundedness/
calculate_relevance/evaluate_note in src/processing.py), the aggregation
helper, and the --evaluate-notes pipeline wiring in main.py.
"""

from __future__ import annotations

import json

import main
from src import processing

_FLUENCY_MARKER = "linguistic fluency"
_GROUNDEDNESS_MARKER = "grounded in the"
_RELEVANCE_MARKER = "clinical relevance of a synthetic clinical note"


class TestCalculateReadabilityMetrics:
    def test_returns_expected_keys_for_real_text(self):
        text = (
            "Patient presents with acute chest pain radiating to the left arm. "
            "ECG shows ST elevation in leads II, III, and aVF, consistent with "
            "an inferior STEMI."
        )
        metrics = processing.calculate_readability_metrics(text)

        assert set(metrics) == {
            "flesch_reading_ease",
            "flesch_kincaid_grade",
            "gunning_fog",
            "word_count",
        }
        assert metrics["word_count"] > 0

    def test_blank_text_returns_empty_dict(self):
        assert processing.calculate_readability_metrics("") == {}
        assert processing.calculate_readability_metrics("   ") == {}

    def test_no_llm_call_involved(self, monkeypatch):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("calculate_readability_metrics must not call the LLM")

        monkeypatch.setattr(processing, "call_llm", fail_if_called)
        processing.calculate_readability_metrics("Some clinical note text here.")


class TestLlmJudgedMetrics:
    def test_calculate_fluency_sends_note_and_parses_score(self, monkeypatch):
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kw):
            captured["prompt"] = prompt
            return json.dumps({"fluency_score": 0.92, "grammar_score": 0.9})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        result = processing.calculate_fluency("The patient was reviewed today.")

        assert "The patient was reviewed today." in captured["prompt"]
        assert result["fluency_score"] == 0.92

    def test_calculate_groundedness_sends_note_and_reference(self, monkeypatch):
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kw):
            captured["prompt"] = prompt
            return json.dumps({"groundedness_score": 0.75, "is_grounded": True})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        result = processing.calculate_groundedness("Note text", "Reference material here")

        assert "Note text" in captured["prompt"]
        assert "Reference material here" in captured["prompt"]
        assert result["groundedness_score"] == 0.75

    def test_calculate_relevance_sends_note_and_reference(self, monkeypatch):
        captured = {}

        def fake_call_llm(prompt, model=None, temp=0.7, **kw):
            captured["prompt"] = prompt
            return json.dumps({"relevance_score": 0.6, "is_relevant": False})

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        result = processing.calculate_relevance("Note text", "Reference material here")

        assert "Note text" in captured["prompt"]
        assert "Reference material here" in captured["prompt"]
        assert result["relevance_score"] == 0.6


class TestEvaluateNote:
    def test_combines_readability_and_llm_scores(self, monkeypatch):
        def fake_call_llm(prompt, model=None, temp=0.7, **kw):
            if _FLUENCY_MARKER in prompt:
                return json.dumps({"fluency_score": 0.9})
            if _GROUNDEDNESS_MARKER in prompt:
                return json.dumps({"groundedness_score": 0.8})
            if _RELEVANCE_MARKER in prompt:
                return json.dumps({"relevance_score": 0.7})
            raise AssertionError(f"unexpected prompt: {prompt[:80]}")

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        result = processing.evaluate_note("Some clinical note text.", "reference material")

        assert result["fluency_score"] == 0.9
        assert result["groundedness_score"] == 0.8
        assert result["relevance_score"] == 0.7
        assert "flesch_reading_ease" in result

    def test_one_failing_metric_does_not_abort_the_others(self, monkeypatch):
        def fake_call_llm(prompt, model=None, temp=0.7, **kw):
            if _FLUENCY_MARKER in prompt:
                raise RuntimeError("LLM exploded")
            if _GROUNDEDNESS_MARKER in prompt:
                return json.dumps({"groundedness_score": 0.8})
            if _RELEVANCE_MARKER in prompt:
                return json.dumps({"relevance_score": 0.7})
            raise AssertionError(f"unexpected prompt: {prompt[:80]}")

        monkeypatch.setattr(processing, "call_llm", fake_call_llm)

        result = processing.evaluate_note("Some clinical note text.", "reference material")

        assert result["fluency_score"] is None
        assert result["groundedness_score"] == 0.8
        assert result["relevance_score"] == 0.7


class TestAverageNoteQualityMetrics:
    def test_averages_present_scores_and_skips_none(self):
        notes = [
            [
                {"fluency_score": 0.8, "groundedness_score": None, "flesch_reading_ease": 60.0},
                {"fluency_score": 0.6, "groundedness_score": 0.9, "flesch_reading_ease": 70.0},
            ]
        ]

        result = processing.average_note_quality_metrics(notes)

        assert result["avg_fluency_score"] == 0.7
        assert result["avg_groundedness_score"] == 0.9
        assert result["avg_flesch_reading_ease"] == 65.0
        assert "avg_relevance_score" not in result

    def test_no_scores_anywhere_returns_empty_dict(self):
        notes = [[{"clean_note_text": "text", "note_type": "ED event"}]]
        assert processing.average_note_quality_metrics(notes) == {}

    def test_empty_notes_returns_empty_dict(self):
        assert processing.average_note_quality_metrics([]) == {}


# ---------------------------------------------------------------------------
# Pipeline-level wiring
# ---------------------------------------------------------------------------

_PATIENT_MARKER = "no resemblance to any real person"
_JOURNEY_MARKER = "Return ONLY a valid JSON array"
_NOTE_MARKER = "Write the note text only"


def _fake_pipeline_call_llm_with_evaluation(*, note_text: str = "Patient reviewed, stable."):
    def fake(prompt, model=None, temp=0.7, **kwargs):
        if _PATIENT_MARKER in prompt:
            return json.dumps(
                {
                    "full_name": "Test Patient", "first_name": "Test", "surname": "Patient",
                    "age": 60, "sex": "Male", "nhs_number": "123 456 7890", "mrn": "1234567",
                }
            )
        if _JOURNEY_MARKER in prompt:
            return json.dumps(
                [
                    {
                        "event_type": "ED event", "event_date": "2026-07-08", "event_time": "10:00",
                        "event_order": 1, "brief_description": "ED assessment", "clinician_type": "Doctor",
                    }
                ]
            )
        if _NOTE_MARKER in prompt:
            return note_text
        if _FLUENCY_MARKER in prompt:
            return json.dumps({"fluency_score": 0.85})
        if _GROUNDEDNESS_MARKER in prompt:
            return json.dumps({"groundedness_score": 0.75})
        if _RELEVANCE_MARKER in prompt:
            return json.dumps({"relevance_score": 0.65})
        return json.dumps({"admission_type": "emergency", "estimated_los_days": 5})

    return fake


class TestEvaluateNotesWiredIntoPipeline:
    def test_evaluate_notes_flag_adds_score_columns_and_summary(self, tmp_path, monkeypatch):
        monkeypatch.setattr(processing, "call_llm", _fake_pipeline_call_llm_with_evaluation())

        args = main.parse_args(
            [
                "--diagnostic-codes", "I21.0",
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "0",
                "--evaluate-notes",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        notes_csv = (tmp_path / "synthetic_clinical_notes.csv").read_text()
        assert "fluency_score" in notes_csv
        assert "0.85" in notes_csv

        summary = json.loads((tmp_path / "generation_summary.json").read_text())
        quality = summary["note_quality_metrics"]
        assert quality["avg_fluency_score"] == 0.85
        assert quality["avg_groundedness_score"] == 0.75
        assert quality["avg_relevance_score"] == 0.65

    def test_evaluate_notes_off_by_default_no_extra_llm_calls(self, tmp_path, monkeypatch):
        call_log: list[str] = []

        def fake(prompt, model=None, temp=0.7, **kwargs):
            if _FLUENCY_MARKER in prompt or _GROUNDEDNESS_MARKER in prompt or _RELEVANCE_MARKER in prompt:
                call_log.append("evaluation")
            return _fake_pipeline_call_llm_with_evaluation()(prompt, model=model, temp=temp, **kwargs)

        monkeypatch.setattr(processing, "call_llm", fake)

        args = main.parse_args(
            [
                "--diagnostic-codes", "I21.0",
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "0",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0
        assert call_log == []

        summary = json.loads((tmp_path / "generation_summary.json").read_text())
        assert "note_quality_metrics" not in summary

    def test_evaluation_runs_after_correction_scores_final_text(self, tmp_path, monkeypatch):
        """If correction rewrites the note, evaluation should see the
        corrected text, not the pre-correction draft."""
        code = "I21.0"
        call_log: list[str] = []

        def fake(prompt, model=None, temp=0.7, **kwargs):
            if _PATIENT_MARKER in prompt:
                return json.dumps(
                    {
                        "full_name": "Test Patient", "first_name": "Test", "surname": "Patient",
                        "age": 60, "sex": "Male", "nhs_number": "123 456 7890", "mrn": "1234567",
                    }
                )
            if _JOURNEY_MARKER in prompt:
                return json.dumps(
                    [
                        {
                            "event_type": "ED event", "event_date": "2026-07-08", "event_time": "10:00",
                            "event_order": 1, "brief_description": "ED assessment", "clinician_type": "Doctor",
                        }
                    ]
                )
            if "revising a synthetic clinical note" in prompt:
                return f"Clinical note: patient managed for {code} (STEMI)."
            if "revising a synthetic admission record" in prompt:
                return json.dumps({"working_diagnosis": f"{code} - STEMI"})
            if _NOTE_MARKER in prompt:
                return "Generic note with no code mentioned."
            if _FLUENCY_MARKER in prompt:
                call_log.append(prompt)
                return json.dumps({"fluency_score": 0.5})
            if _GROUNDEDNESS_MARKER in prompt or _RELEVANCE_MARKER in prompt:
                return json.dumps({"groundedness_score": 0.5, "relevance_score": 0.5})
            return json.dumps({"admission_type": "emergency", "estimated_los_days": 5})

        monkeypatch.setattr(processing, "call_llm", fake)

        args = main.parse_args(
            [
                "--diagnostic-codes", code,
                "--n-patients", "1",
                "--output-dir", str(tmp_path),
                "--max-correction-attempts", "1",
                "--evaluate-notes",
            ]
        )
        exit_code = main.run_pipeline(args)
        assert exit_code == 0

        # The prompt sent to the fluency evaluator must contain the
        # corrected note text (with the code), not the generic pre-correction draft.
        assert any(code in p for p in call_log)
