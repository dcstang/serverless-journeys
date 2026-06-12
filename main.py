"""
main.py - CLI entrypoint for serverless-journeys Nebius serverless job.

Generates synthetic NHS patient journeys with clinical notes, driven by
ICD-10 diagnosis codes and/or OPCS-4 procedure codes.

Usage:
    python main.py [--icd10-codes I21.0,J18.1] [--opcs4-codes K40.1] \\
                   [--n-patients 5] [--output-dir data/output] \\
                   [--model claude-sonnet-4-6] [--llm-provider anthropic]

Environment variables (override CLI args or provide defaults):
    ANTHROPIC_API_KEY  - required for anthropic provider
    OPENAI_API_KEY     - required for openai provider
    LLM_PROVIDER       - 'anthropic' or 'openai'
    MODEL              - model identifier
    N_PATIENTS         - number of patients to generate
    ICD10_CODES        - comma-separated ICD-10 codes
    OPCS4_CODES        - comma-separated OPCS-4 codes
    OUTPUT_DIR         - output directory path
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
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
        modules["params"] = importlib.import_module("config.params")
        modules["config"] = importlib.import_module("config.config")
        modules["processing"] = importlib.import_module("src.processing")
        modules["icd10"] = importlib.import_module("src.codes.icd10")
        modules["opcs4"] = importlib.import_module("src.codes.opcs4")
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
        "--icd10-codes",
        type=str,
        default=os.environ.get("ICD10_CODES", ""),
        help=(
            "Comma-separated ICD-10 codes to drive admission generation "
            "(e.g. 'I21.0,J18.1'). Overrides ICD10_CODES env var."
        ),
    )
    parser.add_argument(
        "--opcs4-codes",
        type=str,
        default=os.environ.get("OPCS4_CODES", ""),
        help=(
            "Comma-separated OPCS-4 procedure codes "
            "(e.g. 'K40.1,W37.1'). Overrides OPCS4_CODES env var."
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
            "LLM model identifier. Defaults to 'claude-sonnet-4-6' for Anthropic "
            "or 'gpt-4o' for OpenAI."
        ),
    )
    parser.add_argument(
        "--llm-provider",
        type=str,
        default=os.environ.get("LLM_PROVIDER", "anthropic"),
        choices=["anthropic", "openai"],
        help="LLM provider to use (default: anthropic).",
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


def _stub_admission(patient: dict, admission_date: str, icd10_codes: list, opcs4_codes: list) -> dict:
    """Return stub admission data for test mode."""
    return {
        "admission_id": str(uuid.uuid4()),
        "patient_id": patient["person_id"],
        "admission_type": "emergency",
        "admission_method": "Emergency department",
        "admission_date": admission_date,
        "icd10_codes": icd10_codes,
        "opcs4_codes": opcs4_codes,
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
    icd10_str: str,
    opcs4_str: str,
    mods: dict,
) -> tuple[list[str], list[str]]:
    """Parse ICD-10 and OPCS-4 code strings into validated lists.

    Args:
        icd10_str: Comma-separated ICD-10 code string.
        opcs4_str: Comma-separated OPCS-4 code string.
        mods: Module namespace dict.

    Returns:
        Tuple of (icd10_codes, opcs4_codes) lists.
    """
    icd10_codes = mods["icd10"].parse_codes(icd10_str)
    opcs4_codes = mods["opcs4"].parse_codes(opcs4_str)

    if icd10_codes:
        logger.info("ICD-10 codes: %s", ", ".join(icd10_codes))
        for code in icd10_codes:
            info = mods["icd10"].lookup_code(code)
            if info:
                logger.info("  %s: %s (%s)", code, info["description"], info["specialty"])
            else:
                logger.warning("  %s: not found in curated dictionary", code)

    if opcs4_codes:
        logger.info("OPCS-4 codes: %s", ", ".join(opcs4_codes))
        for code in opcs4_codes:
            info = mods["opcs4"].lookup_code(code)
            if info:
                logger.info(
                    "  %s: %s (%s)", code, info["description"], info["surgical_specialty"]
                )
            else:
                logger.warning("  %s: not found in curated dictionary", code)

    if not icd10_codes and not opcs4_codes:
        logger.info("No codes provided - will generate random admissions")

    return icd10_codes, opcs4_codes


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
    icd10_codes: list[str],
    opcs4_codes: list[str],
    admission_date: str,
    mods: dict,
    model: str | None,
    test_mode: bool,
) -> dict:
    """Generate an admission record for a patient.

    Args:
        patient: Patient demographics dict.
        icd10_codes: List of ICD-10 codes (may be empty).
        opcs4_codes: List of OPCS-4 codes (may be empty).
        admission_date: ISO date string.
        mods: Module namespace dict.
        model: Optional model override.
        test_mode: If True, return stub data.

    Returns:
        Admission details dict.
    """
    if test_mode:
        return _stub_admission(patient, admission_date, icd10_codes, opcs4_codes)

    logger.info("  Generating admission...")
    admission_time = mods["processing"].random_24_hour_time()
    admission = mods["processing"].generate_from_codes(
        icd10_codes=icd10_codes,
        opcs4_codes=opcs4_codes,
        patient_details=patient,
        admission_date=admission_date,
        admission_time=admission_time,
        model=model,
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
) -> tuple[list[dict], str, str]:
    """Generate a patient journey (sequence of events).

    Args:
        patient: Patient demographics dict.
        admission: Admission details dict.
        admission_date: ISO date string.
        mods: Module namespace dict.
        model: Optional model override.
        test_mode: If True, return stub data.

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
                    note_text = mods["processing"].add_typos_to_string(note_text)
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


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def save_outputs(
    all_patients: list[dict],
    all_admissions: list[dict],
    all_journeys: list[list[dict]],
    all_notes: list[list[dict]],
    output_dir: str,
    mods: dict,
) -> None:
    """Save all generated data to CSV files in output_dir.

    Args:
        all_patients: List of patient dicts.
        all_admissions: List of admission dicts.
        all_journeys: List of journey event lists.
        all_notes: List of clinical note lists.
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
        all_patients, all_admissions, all_journeys, all_notes
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
    output_dir: str,
) -> None:
    """Print a human-readable summary of the generation run to stdout.

    Args:
        all_patients: List of patient dicts.
        all_admissions: List of admission dicts.
        all_journeys: List of journey event lists.
        all_notes: List of clinical note lists.
        output_dir: Output directory path.
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

    # Import project modules
    try:
        mods = _import_project_modules()
    except ImportError as exc:
        logger.error("Module import failed: %s", exc)
        return 1

    # Parse and validate codes
    icd10_codes, opcs4_codes = step_parse_codes(
        args.icd10_codes, args.opcs4_codes, mods
    )

    # Determine model
    model: str | None = args.model or None

    # Collections for all generated data
    all_patients: list[dict] = []
    all_admissions: list[dict] = []
    all_journeys: list[list[dict]] = []
    all_notes: list[list[dict]] = []

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
                patient, icd10_codes, opcs4_codes,
                admission_date, mods, model, args.test_mode
            )

            # Step 3: Generate patient journey
            journey, adm_date, dis_date = step_generate_journey(
                patient, admission, admission_date, mods, model, args.test_mode
            )
            admission["discharge_date"] = dis_date

            # Step 4: Generate clinical notes for each event
            logger.info(
                "  Generating %d clinical notes...", len(journey)
            )
            notes = step_generate_notes(
                patient, admission, journey, mods, model,
                args.test_mode, args.apply_abbreviations, args.apply_typos
            )

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
            args.output_dir, mods
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to save outputs: %s", exc, exc_info=args.verbose)
        return 1

    # Print summary
    print_summary(all_patients, all_admissions, all_journeys, all_notes, args.output_dir)

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
