"""
main.py - CLI entrypoint for serverless-journeys Nebius serverless job.

Generates synthetic NHS patient journeys with clinical notes, driven by
diagnostic codes and/or procedure codes from any registered coding
standard (see src/codes/registry.py). Defaults to ICD-10 for diagnoses and
OPCS-4 for procedures, but --diagnostic-code-system / --procedure-code-system
can select any other registered system (e.g. ICD-11, SNOMED CT, CPT).

Usage:
    python main.py [--diagnostic-codes I21.0,J18.1] [--procedure-codes K40.1] \\
                   [--n-patients 5] [--output-dir data/output] \\
                   [--model claude-sonnet-4-6] [--llm-provider anthropic]

    # Run against a Nebius serverless GPU endpoint instead:
    python main.py --llm-provider nebius \\
                   --model meta-llama/Meta-Llama-3.1-70B-Instruct-fast \\
                   --diagnostic-codes I21.0 --n-patients 10

    # Research uncurated codes via Google Search and let the pipeline
    # correct any generated content that ends up missing a driving code:
    python main.py --diagnostic-codes M75.100 --research-unknown-codes \\
                   --max-correction-attempts 2

After each patient is generated, a backward-pass check confirms each
driving code is actually reflected in the admission/notes (not just passed
in as a parameter); on a miss it applies a targeted corrective regeneration
(see --max-correction-attempts). For codes with no curated dictionary entry
in src/codes/icd10.py / opcs4.py, --research-unknown-codes looks the code up
via Google Custom Search first so generation (and any correction) is
grounded in what the code actually means, not a guess from the bare code.

Environment variables (override CLI args or provide defaults):
    ANTHROPIC_API_KEY       - required for anthropic provider
    OPENAI_API_KEY          - required for openai provider
    NEBIUS_API_KEY          - required for nebius provider
    NEBIUS_BASE_URL         - Nebius endpoint (default: https://api.studio.nebius.com/v1/)
    LLM_PROVIDER            - 'anthropic', 'openai', or 'nebius'
    MODEL                   - model identifier (e.g. any model id on your Nebius endpoint)
    N_PATIENTS              - number of patients to generate
    DIAGNOSTIC_CODES        - comma-separated diagnostic codes
    DIAGNOSTIC_CODE_SYSTEM  - diagnostic coding standard, default 'icd10'
    PROCEDURE_CODES         - comma-separated procedure codes
    PROCEDURE_CODE_SYSTEM   - procedure coding standard, default 'opcs4'
    MAX_CORRECTION_ATTEMPTS - corrective regenerations per missed code, default 1
    RESEARCH_UNKNOWN_CODES  - 'true' to research uncurated codes via web search
    GOOGLE_SEARCH_API_KEY   - required if RESEARCH_UNKNOWN_CODES is enabled
    GOOGLE_SEARCH_CSE_ID    - required if RESEARCH_UNKNOWN_CODES is enabled
    N_EVENTS_PER_PATIENT    - approximate journey events per patient (soft target), default 8
    TYPO_RATE               - proportion of words to corrupt when APPLY_TYPOS is set, default 0.02
    MAX_LLM_ATTEMPTS        - retry attempts per LLM call, default 5
    MIN_LOS_DAYS/MAX_LOS_DAYS - suggested LOS range for no-codes random admissions, default 1/14
    OUTPUT_DIR              - output directory path
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import Any

# Load .env file if present (development convenience)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("serverless-journeys")

# ---------------------------------------------------------------------------
# Imports from project modules (after .env is loaded)
# ---------------------------------------------------------------------------


def _import_project_modules():
    """Import project modules, returning them as a namespace dict."""
    import importlib  # noqa: PLC0415

    modules: dict[str, Any] = {}

    try:
        modules["config"] = importlib.import_module("config.config")
        modules["processing"] = importlib.import_module("src.processing")
        # Importing icd10/opcs4 registers them as CodeSystems (see
        # src/codes/registry.py) - keep these imports even though this
        # module mostly talks to the generic registry from here on.
        modules["icd10"] = importlib.import_module("src.codes.icd10")
        modules["opcs4"] = importlib.import_module("src.codes.opcs4")
        modules["registry"] = importlib.import_module("src.codes.registry")
        modules["doc_templates"] = importlib.import_module("src.doc_templates")
    except ImportError as exc:
        logger.error("Failed to import project modules: %s", exc)
        raise

    return modules


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, with environment variable fallbacks.

    CLI args take precedence over environment variables.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic NHS patient journeys with clinical notes, "
            "driven by ICD-10 / OPCS-4 codes."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--diagnostic-codes",
        type=str,
        default=os.environ.get("DIAGNOSTIC_CODES", ""),
        help=(
            "Comma-separated diagnostic codes to drive admission generation "
            "(e.g. 'I21.0,J18.1'). Overrides DIAGNOSTIC_CODES env var. "
            "Interpreted under --diagnostic-code-system."
        ),
    )
    parser.add_argument(
        "--diagnostic-code-system",
        type=str,
        default=os.environ.get("DIAGNOSTIC_CODE_SYSTEM", "icd10"),
        help=(
            "Coding standard the --diagnostic-codes belong to (default: 'icd10'). "
            "Any system registered in src/codes/registry.py may be used - see "
            "that module for how to add support for another standard (e.g. "
            "ICD-11, SNOMED CT)."
        ),
    )
    parser.add_argument(
        "--procedure-codes",
        type=str,
        default=os.environ.get("PROCEDURE_CODES", ""),
        help=(
            "Comma-separated procedure codes (e.g. 'K40.1,W37.1'). Overrides "
            "PROCEDURE_CODES env var. Interpreted under --procedure-code-system."
        ),
    )
    parser.add_argument(
        "--procedure-code-system",
        type=str,
        default=os.environ.get("PROCEDURE_CODE_SYSTEM", "opcs4"),
        help=(
            "Coding standard the --procedure-codes belong to (default: 'opcs4'). "
            "Any system registered in src/codes/registry.py may be used (e.g. "
            "CPT, HCPCS)."
        ),
    )
    parser.add_argument(
        "--n-events-per-patient",
        type=int,
        default=int(os.environ.get("N_EVENTS_PER_PATIENT", "8")),
        help=(
            "Approximate number of journey events (and therefore clinical notes) to "
            "aim for per patient (default: 8). A soft guide passed into the journey "
            "prompt, not a hard cap - actual length of stay and admission type still "
            "shape the real count."
        ),
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=int(os.environ.get("N_PATIENTS", "5")),
        help="Number of synthetic patients to generate (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("OUTPUT_DIR", "data/output"),
        help="Output directory for generated CSV files (default: data/output).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("MODEL", ""),
        help=(
            "LLM model identifier. Defaults to 'claude-sonnet-4-6' for Anthropic, "
            "'gpt-4o' for OpenAI, or 'meta-llama/Meta-Llama-3.1-70B-Instruct-fast' "
            "for Nebius. For Nebius this can be set to any model deployed on your "
            "serverless endpoint."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=os.environ.get("LLM_PROVIDER", "anthropic"),
        choices=["anthropic", "openai", "nebius"],
        help="LLM provider to use (default: anthropic). Use 'nebius' to run "
        "against a Nebius AI Studio serverless GPU endpoint.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        default=os.environ.get("TEST_MODE", "false").lower() == "true",
        help=(
            "Test mode: generate 1 patient with minimal events using a simple prompt. "
            "Does not call the LLM - uses stub data. Useful for integration testing."
        ),
    )
    parser.add_argument(
        "--research-unknown-codes",
        action="store_true",
        default=os.environ.get("RESEARCH_UNKNOWN_CODES", "false").lower() == "true",
        help=(
            "Look up diagnostic/procedure codes with no curated dictionary entry via "
            "Google Custom Search before generation, so the LLM has real clinical "
            "grounding instead of guessing from the bare code alone. Requires "
            "GOOGLE_SEARCH_API_KEY and GOOGLE_SEARCH_CSE_ID. Adds latency/cost per "
            "unique uncurated code (results are cached for the run)."
        ),
    )
    parser.add_argument(
        "--max-correction-attempts",
        type=int,
        default=int(os.environ.get("MAX_CORRECTION_ATTEMPTS", "1")),
        help=(
            "Max targeted corrective regenerations per driving code that the "
            "backward-pass check finds missing from the generated admission/notes "
            "(edits just the admission record and first note, not a full patient "
            "re-run). Set to 0 to disable correction (check-only). Default: 1."
        ),
    )
    parser.add_argument(
        "--evaluate-notes",
        action="store_true",
        default=os.environ.get("EVALUATE_NOTES", "false").lower() == "true",
        help=(
            "Score generated notes after generation completes: objective readability "
            "(free) plus LLM-judged fluency/groundedness/relevance (3 extra LLM calls "
            "per note). Scores are attached as extra columns on each note and averaged "
            "into generation_summary.json / the run summary. Off by default."
        ),
    )
    parser.add_argument(
        "--apply-abbreviations",
        action="store_true",
        default=os.environ.get("APPLY_ABBREVIATIONS", "false").lower() == "true",
        help="Apply clinical abbreviation augmentation to generated notes.",
    )
    parser.add_argument(
        "--apply-typos",
        action="store_true",
        default=os.environ.get("APPLY_TYPOS", "false").lower() == "true",
        help="Apply realistic typo augmentation to generated notes.",
    )
    parser.add_argument(
        "--typo-rate",
        type=float,
        default=float(os.environ.get("TYPO_RATE", "0.02")),
        help="Approximate proportion of words to corrupt when --apply-typos is set (default: 0.02).",
    )
    parser.add_argument(
        "--max-llm-attempts",
        type=int,
        default=int(os.environ.get("MAX_LLM_ATTEMPTS", "5")),
        help="Max retry attempts per LLM call before giving up (default: 5).",
    )
    parser.add_argument(
        "--min-los-days",
        type=int,
        default=int(os.environ.get("MIN_LOS_DAYS", "1")),
        help=(
            "Minimum length of stay (days) suggested to the LLM for randomly generated "
            "admissions when no diagnostic/procedure codes are given (default: 1)."
        ),
    )
    parser.add_argument(
        "--max-los-days",
        type=int,
        default=int(os.environ.get("MAX_LOS_DAYS", "14")),
        help=(
            "Maximum length of stay (days) suggested to the LLM for randomly generated "
            "admissions when no diagnostic/procedure codes are given (default: 14)."
        ),
    )
    parser.add_argument(
        "--admission-date",
        type=str,
        default=os.environ.get("ADMISSION_DATE", datetime.today().strftime("%Y-%m-%d")),
        help="Default admission date (ISO format YYYY-MM-DD, default: today).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose (DEBUG) logging.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Test mode stub data
# ---------------------------------------------------------------------------


def _stub_patient(index: int) -> dict:
    """Return stub patient data for test mode."""
    return {
        "person_id": str(uuid.uuid4()),
        "patient_id": str(uuid.uuid4()),
        "full_name": f"Test Patient {index}",
        "first_name": "Test",
        "surname": f"Patient{index}",
        "title": "Mr",
        "age": 65,
        "sex": "Male",
        "date_of_birth": "1960-01-01",
        "nhs_number": f"123 456 {7000 + index:04d}",
        "mrn": f"MRN{1000 + index:04d}",
        "address_line_1": "1 Test Street",
        "address_city": "London",
        "postcode": "SW1A 1AA",
        "allergies": ["NKDA"],
        "past_medical_history": ["Hypertension", "Type 2 diabetes"],
        "medications": ["Ramipril 5mg OD", "Metformin 500mg BD"],
    }


def _stub_admission(
    patient: dict,
    admission_date: str,
    diagnostic_codes: list,
    diagnostic_code_system: str,
    procedure_codes: list,
    procedure_code_system: str,
) -> dict:
    """Return stub admission data for test mode."""
    return {
        "admission_id": str(uuid.uuid4()),
        "patient_id": patient["person_id"],
        "admission_type": "emergency",
        "admission_method": "Emergency department",
        "admission_date": admission_date,
        "diagnostic_codes": diagnostic_codes,
        "diagnostic_code_system": diagnostic_code_system if diagnostic_codes else None,
        "procedure_codes": procedure_codes,
        "procedure_code_system": procedure_code_system if procedure_codes else None,
        "specialty": "General Medicine",
        "ward": "Medical Assessment Unit (MAU)",
        "estimated_los_days": 4,
        "chief_complaint": "Test admission",
        "working_diagnosis": "Test diagnosis",
        "management_plan": "Test management",
    }


def _stub_journey(admission_date: str) -> list[dict]:
    """Return stub journey for test mode."""
    return [
        {
            "event_type": "ED event",
            "event_date": admission_date,
            "event_time": "14:00",
            "event_order": 1,
            "brief_description": "Triage and initial assessment in ED",
            "clinician_type": "Emergency Physician",
        },
        {
            "event_type": "post take ward round",
            "event_date": admission_date,
            "event_time": "20:00",
            "event_order": 2,
            "brief_description": "Post-take ward round by medical registrar",
            "clinician_type": "Medical Registrar",
        },
    ]


def _stub_note(event: dict, patient: dict) -> str:
    """Return stub clinical note for test mode."""
    return (
        f"TEST NOTE - {event['event_type'].upper()}\n"
        f"Date: {event['event_date']} {event['event_time']}\n"
        f"Patient: {patient.get('full_name', 'Test Patient')}\n"
        f"Clinician: Test Clinician\n\n"
        f"This is a stub clinical note generated in test mode.\n"
        f"Event: {event.get('brief_description', 'N/A')}"
    )


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def step_parse_codes(
    diagnostic_codes_str: str,
    diagnostic_code_system: str,
    procedure_codes_str: str,
    procedure_code_system: str,
    mods: dict,
) -> tuple[list[str], list[str]]:
    """Parse and validate diagnostic/procedure code strings.

    Works for any code system registered in src/codes/registry.py, not just
    the built-in ICD-10/OPCS-4 - diagnostic_code_system and
    procedure_code_system select which registered CodeSystem to validate
    and look up descriptions against.

    Args:
        diagnostic_codes_str: Comma-separated diagnostic code string.
        diagnostic_code_system: Registry key for the diagnostic code system.
        procedure_codes_str: Comma-separated procedure code string.
        procedure_code_system: Registry key for the procedure code system.
        mods: Module namespace dict.

    Returns:
        Tuple of (diagnostic_codes, procedure_codes) lists.
    """
    registry = mods["registry"]

    diagnostic_codes = registry.parse_codes(diagnostic_codes_str)
    procedure_codes = registry.parse_codes(procedure_codes_str)

    if diagnostic_codes:
        diag_system = registry.get_code_system(diagnostic_code_system)
        logger.info("%s (diagnostic) codes: %s", diag_system.name, ", ".join(diagnostic_codes))
        for code in diagnostic_codes:
            info = registry.lookup_code(diag_system, code)
            if info:
                specialty = info.get(diag_system.specialty_field, "")
                logger.info("  %s: %s (%s)", code, info.get("description", ""), specialty)
            else:
                logger.warning("  %s: not found in curated %s dictionary", code, diag_system.name)

    if procedure_codes:
        proc_system = registry.get_code_system(procedure_code_system)
        logger.info("%s (procedure) codes: %s", proc_system.name, ", ".join(procedure_codes))
        for code in procedure_codes:
            info = registry.lookup_code(proc_system, code)
            if info:
                specialty = info.get(proc_system.specialty_field, "")
                logger.info("  %s: %s (%s)", code, info.get("description", ""), specialty)
            else:
                logger.warning("  %s: not found in curated %s dictionary", code, proc_system.name)

    if not diagnostic_codes and not procedure_codes:
        logger.info("No codes provided - will generate random admissions")

    return diagnostic_codes, procedure_codes


def step_generate_patient(
    patient_index: int,
    mods: dict,
    model: str | None,
    test_mode: bool,
) -> dict:
    """Generate a single synthetic patient record.

    Args:
        patient_index: 1-based index for logging.
        mods: Module namespace dict.
        model: Optional model override.
        test_mode: If True, return stub data.

    Returns:
        Patient demographics dict.
    """
    if test_mode:
        return _stub_patient(patient_index)

    logger.info("  Generating patient demographics...")
    patient = mods["processing"].generate_patient(model=model)
    patient.setdefault("person_id", str(uuid.uuid4()))
    patient.setdefault("patient_id", patient["person_id"])
    logger.info("  Generated: %s", patient.get("full_name", "Unknown"))
    return patient


def step_generate_admission(
    patient: dict,
    diagnostic_codes: list[str],
    diagnostic_code_system: str,
    procedure_codes: list[str],
    procedure_code_system: str,
    admission_date: str,
    mods: dict,
    model: str | None,
    test_mode: bool,
    enable_code_research: bool = False,
    min_los_days: int = 1,
    max_los_days: int = 14,
) -> dict:
    """Generate an admission record for a patient.

    Args:
        patient: Patient demographics dict.
        diagnostic_codes: List of diagnostic codes (may be empty).
        diagnostic_code_system: Registry key for the diagnostic code system.
        procedure_codes: List of procedure codes (may be empty).
        procedure_code_system: Registry key for the procedure code system.
        admission_date: ISO date string.
        mods: Module namespace dict.
        model: Optional model override.
        test_mode: If True, return stub data.
        enable_code_research: If True, research uncurated codes via web
            search before generation (see src.codes.research).
        min_los_days: Minimum LOS to suggest when no codes are given.
        max_los_days: Maximum LOS to suggest when no codes are given.

    Returns:
        Admission details dict.
    """
    if test_mode:
        return _stub_admission(
            patient, admission_date, diagnostic_codes, diagnostic_code_system,
            procedure_codes, procedure_code_system,
        )

    logger.info("  Generating admission...")
    admission_time = mods["processing"].random_24_hour_time()
    admission = mods["processing"].generate_from_codes(
        diagnostic_codes=diagnostic_codes,
        procedure_codes=procedure_codes,
        patient_details=patient,
        admission_date=admission_date,
        admission_time=admission_time,
        model=model,
        diagnostic_code_system=diagnostic_code_system,
        procedure_code_system=procedure_code_system,
        enable_code_research=enable_code_research,
        min_los_days=min_los_days,
        max_los_days=max_los_days,
    )
    admission["admission_id"] = str(uuid.uuid4())
    admission["patient_id"] = patient.get("person_id", str(uuid.uuid4()))
    admission["admission_date"] = admission_date
    admission["admission_time"] = admission_time
    logger.info(
        "  Admission type: %s, LOS: %s days",
        admission.get("admission_type", "unknown"),
        admission.get("estimated_los_days", "?"),
    )
    return admission


def step_generate_journey(
    patient: dict,
    admission: dict,
    admission_date: str,
    mods: dict,
    model: str | None,
    test_mode: bool,
    target_event_count: int = 8,
) -> tuple[list[dict], str, str]:
    """Generate a patient journey (sequence of events).

    Args:
        patient: Patient demographics dict.
        admission: Admission details dict.
        admission_date: ISO date string.
        mods: Module namespace dict.
        model: Optional model override.
        test_mode: If True, return stub data.
        target_event_count: Approximate number of events to aim for - a
            soft guide, not a hard cap (see processing.generate_journey).

    Returns:
        Tuple of (journey_events, admission_date, discharge_date).
    """
    los_days = int(admission.get("estimated_los_days", 4))
    admission_dt, discharge_dt = mods["processing"].create_admission_window(
        admission_date, los_days
    )

    if test_mode:
        journey = _stub_journey(admission_dt)
        return journey, admission_dt, discharge_dt

    logger.info(
        "  Generating journey (admission: %s, discharge: %s)...",
        admission_dt,
        discharge_dt,
    )
    journey = mods["processing"].generate_journey(
        patient=patient,
        admission=admission,
        admission_date=admission_dt,
        discharge_date=discharge_dt,
        possible_event_types=mods["config"].possible_event_types,
        model=model,
        target_event_count=target_event_count,
    )
    logger.info("  Generated %d journey events", len(journey))
    return journey, admission_dt, discharge_dt


def step_generate_notes(
    patient: dict,
    admission: dict,
    journey: list[dict],
    mods: dict,
    model: str | None,
    test_mode: bool,
    apply_abbreviations: bool,
    apply_typos: bool,
    typo_rate: float = 0.02,
) -> list[dict]:
    """Generate clinical notes for each event in a patient journey.

    Args:
        patient: Patient demographics dict.
        admission: Admission details dict.
        journey: List of journey event dicts.
        mods: Module namespace dict.
        model: Optional model override.
        test_mode: If True, return stub notes.
        apply_abbreviations: Whether to apply abbreviation augmentation.
        apply_typos: Whether to apply typo augmentation.
        typo_rate: Approximate proportion of words to corrupt when apply_typos is True.

    Returns:
        List of clinical note dicts.
    """
    from src.doc_templates import document_templates, template_sections_to_combine  # noqa: PLC0415

    notes = []
    previous_events: list[dict] = []

    for i, event in enumerate(journey, 1):
        event_type = event.get("event_type", "misc")
        logger.info(
            "    Note %d/%d: %s (%s)",
            i,
            len(journey),
            event_type,
            event.get("event_date", ""),
        )

        if test_mode:
            note_text = _stub_note(event, patient)
        else:
            # Build template string for this event type
            template = document_templates.get(event_type, document_templates["misc"])
            sections = template_sections_to_combine.get(
                event_type,
                list(template.keys()),
            )
            template_str = mods["processing"].combine_template_sections(template, sections)

            # Get style instructions
            style = mods["config"].style_instructions.get(
                event_type,
                mods["config"].default_style_instructions,
            )

            # Determine red flags
            red_flags_resp = mods["processing"].call_llm(
                mods["doc_templates"].__dict__
                if False  # skip actual red flags call to save LLM calls
                else "",
                model=model,
                temp=0.3,
            ) if False else "None identified"

            # Generate the note
            try:
                note_text = mods["processing"].generate_clinical_note(
                    patient=patient,
                    admission=admission,
                    event=event,
                    previous_events=previous_events,
                    note_template_str=template_str,
                    style_instructions=style,
                    add_examination=True,
                    red_flags=red_flags_resp,
                    model=model,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Note generation failed for event %d: %s", i, exc)
                note_text = f"[Note generation failed: {exc}]"

            # Augmentation
            if apply_abbreviations and len(note_text) > 100:
                try:
                    note_text = mods["processing"].add_abbreviations_to_strings(
                        note_text, model=model
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Abbreviation augmentation failed: %s", exc)

            if apply_typos and len(note_text) > 100:
                try:
                    note_text = mods["processing"].add_typos_to_string(note_text, typo_rate=typo_rate)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Typo augmentation failed: %s", exc)

        note_record = {
            "clinical_note_id": str(uuid.uuid4()),
            "admission_id": admission.get("admission_id", ""),
            "encounter_id": admission.get("encounter_id", admission.get("admission_id", "")),
            "person_id": patient.get("person_id", ""),
            "note_type": event_type,
            "note_subject": event.get("brief_description", event_type),
            "creation_timestamp": f"{event.get('event_date', '')} {event.get('event_time', '00:00')}",
            "clean_note_text": note_text,
            "raw_blob_content": note_text,
            "note_state": "active",
            "event_order": event.get("event_order", i),
        }
        notes.append(note_record)
        previous_events.append(event)

    return notes


def _select_note_for_correction(notes: list[dict], role: str) -> dict | None:
    """Pick which note a corrective rewrite should target.

    A procedure code belongs in an 'operation' note if the journey
    generated one; anything else (including diagnostic codes) falls back
    to the earliest note, which is typically the admission-defining
    ED/ward-round note where a diagnosis first gets documented.

    Args:
        notes: Generated clinical note list.
        role: 'diagnostic' or 'procedure'.

    Returns:
        The note dict to correct, or None if notes is empty.
    """
    if not notes:
        return None
    if role == "procedure":
        for note in notes:
            if note.get("note_type") == "operation":
                return note
    return notes[0]


def step_verify_code_reflection(
    patient: dict,
    admission: dict,
    journey: list[dict],
    notes: list[dict],
    diagnostic_codes: list[str],
    diagnostic_code_system: str,
    procedure_codes: list[str],
    procedure_code_system: str,
    mods: dict,
    patient_idx: int,
    model: str | None,
    enable_code_research: bool,
    max_correction_attempts: int,
) -> dict:
    """Backward-pass check, with bounded targeted correction on a miss.

    Runs src.processing.check_code_reflected for every diagnostic and
    procedure code that drove this admission. For any code not reflected in
    the admission or notes, attempts up to max_correction_attempts targeted
    corrective regenerations (processing.correct_admission_for_code /
    correct_note_for_code - editing just the admission record and one
    journey note, not a full patient re-run), re-checking after each
    attempt and stopping early once the code is reflected. `admission` and
    `notes` are mutated in place by any corrections applied. Codes still
    unreflected after all attempts are logged as warnings.

    Args:
        patient: Generated patient dict.
        admission: Generated admission dict (mutated in place on correction).
        journey: Generated journey event list.
        notes: Generated clinical note list (one entry mutated in place on
            correction, if any notes exist - see _select_note_for_correction).
        diagnostic_codes: Diagnostic codes that drove this admission.
        diagnostic_code_system: Registry key for the diagnostic code system.
        procedure_codes: Procedure codes that drove this admission.
        procedure_code_system: Registry key for the procedure code system.
        mods: Module namespace dict.
        patient_idx: 1-based patient index, for log messages.
        model: Optional LLM model override for correction calls.
        enable_code_research: Whether to research uncurated codes for
            correction context (see src.codes.research).
        max_correction_attempts: Max corrective regeneration attempts per
            unreflected code. 0 disables correction (check-only, matching
            the previous behaviour).

    Returns:
        Dict mapping each code to its final per-part reflection result
        (after any corrections). Keys are "<role>:<code>" (role is
        'diagnostic' or 'procedure') rather than the bare code, since a
        diagnostic and procedure code can share the same string (e.g.
        'J18.1' is a valid key in both the curated ICD-10 and OPCS-4
        dictionaries) and a bare-code key would silently let one
        overwrite the other's result.
    """
    processing = mods["processing"]
    report: dict[str, dict[str, bool]] = {}

    all_codes = [(c, diagnostic_code_system, "diagnostic") for c in diagnostic_codes] + [
        (c, procedure_code_system, "procedure") for c in procedure_codes
    ]

    for code, code_system, role in all_codes:
        result = processing.check_code_reflected(code, code_system, patient, admission, journey, notes)

        attempts = 0
        while not (result["admission"] or result["notes"]) and attempts < max_correction_attempts:
            attempts += 1
            logger.info(
                "  Patient %d: correcting %s code %s (%s), attempt %d/%d",
                patient_idx, role, code, code_system, attempts, max_correction_attempts,
            )
            code_context = processing.get_code_context(
                code_system, code, enable_research=enable_code_research, model=model
            )
            admission.update(
                processing.correct_admission_for_code(admission, code, code_context, model=model)
            )
            target_note = _select_note_for_correction(notes, role)
            if target_note is not None:
                corrected_text = processing.correct_note_for_code(
                    target_note.get("clean_note_text", ""), code, code_context, model=model
                )
                target_note["clean_note_text"] = corrected_text
                target_note["raw_blob_content"] = corrected_text

            result = processing.check_code_reflected(
                code, code_system, patient, admission, journey, notes
            )

        report[f"{role}:{code}"] = result
        if not (result["admission"] or result["notes"]):
            suffix = f" after {attempts} correction attempt(s)" if attempts else ""
            logger.warning(
                "  Patient %d: %s code %s (%s) not reflected in admission or notes%s",
                patient_idx, role, code, code_system, suffix,
            )
        elif attempts:
            logger.info(
                "  Patient %d: %s code %s (%s) reflected after %d correction attempt(s)",
                patient_idx, role, code, code_system, attempts,
            )

    return report


def step_evaluate_notes(
    admission: dict,
    patient: dict,
    notes: list[dict],
    mods: dict,
    model: str | None,
) -> None:
    """Score each note's readability and (LLM-judged) fluency/groundedness/
    relevance, attaching the scores directly onto the note record.

    Runs after any correction (step_verify_code_reflection) so it scores
    the final note text, not a pre-correction draft. Mutates each dict in
    `notes` in place by merging in the extra score columns - failures on
    an individual note are logged and leave that note's scores as None
    rather than aborting the whole patient.

    Args:
        admission: Generated admission dict, used as reference material
            for groundedness/relevance judging.
        patient: Generated patient dict, included in reference material.
        notes: Generated clinical note list (mutated in place).
        mods: Module namespace dict.
        model: Optional LLM model override.
    """
    processing = mods["processing"]
    reference_material = json.dumps(
        {"patient": patient, "admission": admission}, indent=2, default=str
    )

    for note in notes:
        try:
            scores = processing.evaluate_note(
                note.get("clean_note_text", ""), reference_material, model=model
            )
            note.update(scores)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  Note evaluation failed for note %s: %s", note.get("clinical_note_id"), exc)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_outputs(
    all_patients: list[dict],
    all_admissions: list[dict],
    all_journeys: list[list[dict]],
    all_notes: list[list[dict]],
    all_code_reflection_reports: list[dict],
    output_dir: str,
    mods: dict,
) -> None:
    """Save all generated data to CSV files in output_dir.

    Args:
        all_patients: List of patient dicts.
        all_admissions: List of admission dicts.
        all_journeys: List of journey event lists.
        all_notes: List of clinical note lists.
        all_code_reflection_reports: List of per-patient code reflection
            check reports (see step_verify_code_reflection).
        output_dir: Directory to write CSV files.
        mods: Module namespace dict.
    """
    import pandas as pd  # noqa: PLC0415

    os.makedirs(output_dir, exist_ok=True)

    # Patients
    if all_patients:
        patients_df = pd.DataFrame(all_patients)
        patients_path = os.path.join(output_dir, "synthetic_patients.csv")
        patients_df.to_csv(patients_path, index=False)
        logger.info("Saved %d patients to %s", len(patients_df), patients_path)

    # Admissions
    if all_admissions:
        # Serialise list/dict fields to JSON strings for CSV
        admissions_serialised = []
        for adm in all_admissions:
            adm_copy = {}
            for k, v in adm.items():
                if isinstance(v, (list, dict)):
                    adm_copy[k] = json.dumps(v)
                else:
                    adm_copy[k] = v
            admissions_serialised.append(adm_copy)
        admissions_df = pd.DataFrame(admissions_serialised)
        admissions_path = os.path.join(output_dir, "synthetic_admissions.csv")
        admissions_df.to_csv(admissions_path, index=False)
        logger.info("Saved %d admissions to %s", len(admissions_df), admissions_path)

    # Journeys (flat: one row per event)
    all_events_flat = []
    for patient_idx, journey in enumerate(all_journeys):
        for event in journey:
            event_copy = dict(event)
            event_copy["patient_index"] = patient_idx
            event_copy["admission_id"] = all_admissions[patient_idx].get("admission_id", "")
            all_events_flat.append(event_copy)
    if all_events_flat:
        journeys_df = pd.DataFrame(all_events_flat)
        journeys_path = os.path.join(output_dir, "synthetic_journeys.csv")
        journeys_df.to_csv(journeys_path, index=False)
        logger.info(
            "Saved %d journey events to %s", len(journeys_df), journeys_path
        )

    # Clinical notes (flat: one row per note)
    all_notes_flat = [note for patient_notes in all_notes for note in patient_notes]
    if all_notes_flat:
        notes_df = pd.DataFrame(all_notes_flat)
        notes_path = os.path.join(output_dir, "synthetic_clinical_notes.csv")
        notes_df.to_csv(notes_path, index=False)
        logger.info("Saved %d notes to %s", len(notes_df), notes_path)

    # Summary JSON
    summary = mods["processing"].build_output_info(
        all_patients, all_admissions, all_journeys, all_notes,
        code_reflection_reports=all_code_reflection_reports,
    )
    summary_path = os.path.join(output_dir, "generation_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved generation summary to %s", summary_path)


def print_summary(
    all_patients: list[dict],
    all_admissions: list[dict],
    all_journeys: list[list[dict]],
    all_notes: list[list[dict]],
    all_code_reflection_reports: list[dict],
    output_dir: str,
    mods: dict,
) -> None:
    """Print a human-readable summary of the generation run to stdout.

    Args:
        all_patients: List of patient dicts.
        all_admissions: List of admission dicts.
        all_journeys: List of journey event lists.
        all_notes: List of clinical note lists.
        all_code_reflection_reports: List of per-patient code reflection
            check reports (see step_verify_code_reflection).
        output_dir: Output directory path.
        mods: Module namespace dict.
    """
    total_events = sum(len(j) for j in all_journeys)
    total_notes = sum(len(n) for n in all_notes)

    note_type_counts: dict[str, int] = {}
    for patient_notes in all_notes:
        for note in patient_notes:
            nt = note.get("note_type", "unknown")
            note_type_counts[nt] = note_type_counts.get(nt, 0) + 1

    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Patients generated   : {len(all_patients)}")
    print(f"Admissions generated : {len(all_admissions)}")
    print(f"Journey events       : {total_events}")
    print(f"Clinical notes       : {total_notes}")
    if all_patients:
        print(
            f"Avg events/patient   : {total_events / len(all_patients):.1f}"
        )
        print(
            f"Avg notes/patient    : {total_notes / len(all_patients):.1f}"
        )
    print(f"\nNote type distribution:")
    for nt, count in sorted(note_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {nt:<35} {count:>4}")

    if all_code_reflection_reports:
        total_codes = 0
        reflected_codes = 0
        for report in all_code_reflection_reports:
            for result in report.values():
                total_codes += 1
                if result["admission"] or result["notes"]:
                    reflected_codes += 1
        print(f"\nDiagnostic/procedure code reflection (backward-pass check):")
        print(f"  {reflected_codes}/{total_codes} codes reflected in admission or notes")
        if reflected_codes < total_codes:
            print("  See generation_summary.json / logs for codes that weren't reflected.")

    quality_metrics = mods["processing"].average_note_quality_metrics(all_notes)
    if quality_metrics:
        print(f"\nNote quality metrics (averages):")
        for key, value in sorted(quality_metrics.items()):
            print(f"  {key:<28} {value:.2f}")

    print(f"\nOutput directory: {os.path.abspath(output_dir)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> int:
    """Run the full synthetic generation pipeline.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting serverless-journeys pipeline")
    logger.info("Provider: %s | Model: %s | Test mode: %s",
                args.llm_provider, args.model or "(default)", args.test_mode)

    # Set environment variables from args (before importing modules that read them)
    os.environ["LLM_PROVIDER"] = args.llm_provider
    if args.model:
        os.environ["MODEL"] = args.model
    os.environ["MAX_LLM_ATTEMPTS"] = str(args.max_llm_attempts)

    # Import project modules
    try:
        mods = _import_project_modules()
    except ImportError as exc:
        logger.error("Module import failed: %s", exc)
        return 1

    diagnostic_code_system = args.diagnostic_code_system
    procedure_code_system = args.procedure_code_system

    # Parse and validate codes
    try:
        diagnostic_codes, procedure_codes = step_parse_codes(
            args.diagnostic_codes, diagnostic_code_system,
            args.procedure_codes, procedure_code_system, mods,
        )
    except KeyError as exc:
        logger.error("Invalid code system: %s", exc)
        return 1

    # Determine model
    model: str | None = args.model or None

    # Collections for all generated data
    all_patients: list[dict] = []
    all_admissions: list[dict] = []
    all_journeys: list[list[dict]] = []
    all_notes: list[list[dict]] = []
    all_code_reflection_reports: list[dict] = []

    n_patients = 1 if args.test_mode else args.n_patients
    admission_date = args.admission_date

    # Main generation loop
    for patient_idx in range(1, n_patients + 1):
        logger.info(
            "Generating patient %d/%d...", patient_idx, n_patients
        )

        try:
            # Step 1: Generate patient demographics
            patient = step_generate_patient(patient_idx, mods, model, args.test_mode)

            # Step 2: Generate admission (code-driven or random)
            admission = step_generate_admission(
                patient, diagnostic_codes, diagnostic_code_system,
                procedure_codes, procedure_code_system,
                admission_date, mods, model, args.test_mode,
                args.research_unknown_codes,
                args.min_los_days, args.max_los_days,
            )

            # Step 3: Generate patient journey
            journey, adm_date, dis_date = step_generate_journey(
                patient, admission, admission_date, mods, model, args.test_mode,
                args.n_events_per_patient,
            )
            admission["discharge_date"] = dis_date

            # Step 4: Generate clinical notes for each event
            logger.info(
                "  Generating %d clinical notes...", len(journey)
            )
            notes = step_generate_notes(
                patient, admission, journey, mods, model,
                args.test_mode, args.apply_abbreviations, args.apply_typos,
                args.typo_rate,
            )

            # Step 5: Backward-pass check (+ targeted correction on a miss) -
            # do the driving codes actually show up in what got generated?
            # Skipped in test mode, since stub notes are placeholder text
            # with no clinical content.
            if not args.test_mode and (diagnostic_codes or procedure_codes):
                reflection_report = step_verify_code_reflection(
                    patient, admission, journey, notes,
                    diagnostic_codes, diagnostic_code_system,
                    procedure_codes, procedure_code_system,
                    mods, patient_idx, model,
                    args.research_unknown_codes, args.max_correction_attempts,
                )
                admission["code_reflection_check"] = reflection_report
                all_code_reflection_reports.append(reflection_report)

            # Step 6: Quality evaluation (opt-in) - scores the final note
            # text, after any correction above.
            if not args.test_mode and args.evaluate_notes:
                step_evaluate_notes(admission, patient, notes, mods, model)

            all_patients.append(patient)
            all_admissions.append(admission)
            all_journeys.append(journey)
            all_notes.append(notes)

            logger.info(
                "  Patient %d/%d complete: %d events, %d notes",
                patient_idx, n_patients, len(journey), len(notes)
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Patient %d/%d failed: %s", patient_idx, n_patients, exc,
                exc_info=args.verbose
            )
            # Continue with remaining patients
            continue

    if not all_patients:
        logger.error("No patients generated successfully. Exiting.")
        return 1

    # Save outputs
    logger.info("Saving outputs to %s...", args.output_dir)
    try:
        save_outputs(
            all_patients, all_admissions, all_journeys, all_notes,
            all_code_reflection_reports, args.output_dir, mods
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save outputs: %s", exc, exc_info=args.verbose)
        return 1

    # Print summary
    print_summary(
        all_patients, all_admissions, all_journeys, all_notes,
        all_code_reflection_reports, args.output_dir, mods
    )

    logger.info("Pipeline completed successfully.")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """Main entry point.

    Args:
        argv: Optional argument list for testing.
    """
    args = parse_args(argv)
    sys.exit(run_pipeline(args))


if __name__ == "__main__":
    main()
