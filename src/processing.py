"""
Core processing functions for synthetic clinical note generation.

Adapted from nhsengland/synthetic_clinical_notes to use:
  - Anthropic Claude API (default), OpenAI API, or Nebius vLLM
  - CSV-based I/O instead of Palantir Foundry
  - ICD-10 / OPCS-4 code-driven generation

Environment variables:
  LLM_PROVIDER     : 'anthropic' (default), 'openai', or 'nebius'
  ANTHROPIC_API_KEY: required when LLM_PROVIDER=anthropic
  OPENAI_API_KEY   : required when LLM_PROVIDER=openai
  NEBIUS_API_KEY   : required when LLM_PROVIDER=nebius
  NEBIUS_BASE_URL  : Nebius vLLM endpoint (default: https://api.studio.nebius.com/v1/)
  ANTHROPIC_MODEL  : override default Anthropic model
  OPENAI_MODEL     : override default OpenAI model
  NEBIUS_MODEL     : override default Nebius model
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import uuid
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"
DEFAULT_OPENAI_MODEL = "gpt-4o"
DEFAULT_NEBIUS_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct-fast"
DEFAULT_NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"

# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------


def _get_provider() -> str:
    """Return the configured LLM provider ('anthropic', 'openai', or 'nebius')."""
    return os.environ.get("LLM_PROVIDER", "anthropic").lower().strip()


def _get_anthropic_client():
    """Lazily import and return an Anthropic client."""
    try:
        import anthropic  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "anthropic package not installed. Run: pip install anthropic"
        ) from exc
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set."
        )
    return anthropic.Anthropic(api_key=api_key)


def _get_openai_client():
    """Lazily import and return an OpenAI client."""
    try:
        import openai  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        ) from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set."
        )
    return openai.OpenAI(api_key=api_key)


def _get_nebius_client():
    """Return an OpenAI-compatible client pointed at the Nebius vLLM endpoint."""
    try:
        import openai  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "openai package not installed. Run: pip install openai"
        ) from exc
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError(
            "NEBIUS_API_KEY environment variable not set."
        )
    base_url = os.environ.get("NEBIUS_BASE_URL", DEFAULT_NEBIUS_BASE_URL)
    return openai.OpenAI(api_key=api_key, base_url=base_url)


# ---------------------------------------------------------------------------
# call_llm: synchronous LLM call
# ---------------------------------------------------------------------------


def call_llm(
    prompt: str,
    model: str | None = None,
    temp: float = 0.7,
    max_attempts: int | None = None,
    chat_history: list[dict] | None = None,
) -> str:
    """Call the configured LLM and return the text response.

    Supports both Anthropic (Claude) and OpenAI (GPT) backends. Backend is
    selected via the LLM_PROVIDER environment variable.

    Args:
        prompt: The prompt text to send to the model.
        model: Model identifier. Defaults to claude-sonnet-4-6 (Anthropic) or
               gpt-4o (OpenAI) depending on provider.
        temp: Sampling temperature (0.0 - 1.0). Lower = more deterministic.
        max_attempts: Maximum number of retry attempts on failure. Defaults
            to the MAX_LLM_ATTEMPTS env var (or 5), read fresh on each call
            so CLI/env configuration takes effect without needing to thread
            a parameter through every generation function.
        chat_history: Optional list of prior message dicts
                      [{'role': 'user'/'assistant', 'content': '...'}].
                      The prompt is appended as the latest user message.

    Returns:
        The model's text response as a string.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
    if max_attempts is None:
        max_attempts = int(os.environ.get("MAX_LLM_ATTEMPTS", "5"))

    provider = _get_provider()

    if provider == "anthropic":
        return _call_anthropic(prompt, model, temp, max_attempts, chat_history)
    elif provider == "openai":
        return _call_openai(prompt, model, temp, max_attempts, chat_history)
    elif provider == "nebius":
        return _call_nebius(prompt, model, temp, max_attempts, chat_history)
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Must be 'anthropic', 'openai', or 'nebius'."
        )


def _call_anthropic(
    prompt: str,
    model: str | None,
    temp: float,
    max_attempts: int,
    chat_history: list[dict] | None,
) -> str:
    """Call Anthropic Claude API."""
    client = _get_anthropic_client()
    model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL)

    messages: list[dict] = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=temp,
                messages=messages,
            )
            return response.content[0].text
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Anthropic call failed (attempt %d/%d): %s. Retrying in %ds.",
                attempt,
                max_attempts,
                exc,
                wait,
            )
            import time  # noqa: PLC0415
            time.sleep(wait)

    raise RuntimeError(
        f"Anthropic LLM call failed after {max_attempts} attempts."
    ) from last_exc


def _call_openai(
    prompt: str,
    model: str | None,
    temp: float,
    max_attempts: int,
    chat_history: list[dict] | None,
) -> str:
    """Call OpenAI GPT API."""
    client = _get_openai_client()
    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    messages: list[dict] = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "OpenAI call failed (attempt %d/%d): %s. Retrying in %ds.",
                attempt,
                max_attempts,
                exc,
                wait,
            )
            import time  # noqa: PLC0415
            time.sleep(wait)

    raise RuntimeError(
        f"OpenAI LLM call failed after {max_attempts} attempts."
    ) from last_exc


def _call_nebius(
    prompt: str,
    model: str | None,
    temp: float,
    max_attempts: int,
    chat_history: list[dict] | None,
) -> str:
    """Call Nebius serverless vLLM endpoint (OpenAI-compatible API)."""
    client = _get_nebius_client()
    model = model or os.environ.get("NEBIUS_MODEL", DEFAULT_NEBIUS_MODEL)

    messages: list[dict] = []
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": prompt})

    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temp,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 2 ** attempt
            logger.warning(
                "Nebius call failed (attempt %d/%d): %s. Retrying in %ds.",
                attempt,
                max_attempts,
                exc,
                wait,
            )
            import time  # noqa: PLC0415
            time.sleep(wait)

    raise RuntimeError(
        f"Nebius LLM call failed after {max_attempts} attempts."
    ) from last_exc


# ---------------------------------------------------------------------------
# JSON parsing utility
# ---------------------------------------------------------------------------


def parse_llm_json(response: str) -> Any:
    """Parse JSON from an LLM response, handling common formatting issues.

    The LLM may wrap JSON in markdown code fences. This function strips
    those and attempts to parse the JSON.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed Python object (dict or list).

    Raises:
        json.JSONDecodeError: If valid JSON cannot be extracted.
    """
    text = response.strip()
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Abbreviation and typo augmentation
# ---------------------------------------------------------------------------


def add_abbreviations_to_strings(text: str, model: str | None = None) -> str:
    """Apply clinical abbreviation augmentation to a text string.

    Args:
        text: Clinical text to augment with abbreviations.
        model: Optional model override.

    Returns:
        Text with natural clinical abbreviations inserted.
    """
    from src.prompts import processing_prompts  # noqa: PLC0415

    if len(text.strip()) < 50:
        return text
    prompt = processing_prompts["add_abbreviations_prompt"].substitute(TEXT=text)
    return call_llm(prompt, model=model, temp=0.4)


def add_typos_to_string(text: str, typo_rate: float = 0.02) -> str:
    """Introduce realistic typos into a text string.

    Uses a simple heuristic approach: randomly swaps adjacent characters
    in a small proportion of words. For richer augmentation, set a higher
    typo_rate or use the 'typo' library if available.

    Args:
        text: Text to augment.
        typo_rate: Proportion of words to corrupt (0.0-1.0).

    Returns:
        Text with realistic typos inserted.
    """
    # Try to use the typo library if available
    try:
        from typo import StrErrer  # noqa: PLC0415

        errer = StrErrer(text)
        # Apply a random typo operation
        ops = [
            errer.char_swap,
            errer.missing_char,
            errer.extra_char,
            errer.nearby_char,
        ]
        op = random.choice(ops)
        result = op()
        if hasattr(result, "result"):
            return result.result
        return str(result)
    except (ImportError, Exception):
        pass

    # Fallback: simple character swap on a small number of words
    words = text.split()
    n_corrupt = max(1, int(len(words) * typo_rate))
    indices = random.sample(range(len(words)), min(n_corrupt, len(words)))

    for idx in indices:
        word = words[idx]
        if len(word) < 4:
            continue
        i = random.randint(0, len(word) - 2)
        word_list = list(word)
        word_list[i], word_list[i + 1] = word_list[i + 1], word_list[i]
        words[idx] = "".join(word_list)

    return " ".join(words)


# ---------------------------------------------------------------------------
# Template utilities
# ---------------------------------------------------------------------------


def combine_template_sections(
    template: dict[str, str],
    sections: list[str],
) -> str:
    """Combine specified sections of a document template into a prompt string.

    Args:
        template: Document template dict (section_name -> guidance text).
        sections: List of section names to include.

    Returns:
        Formatted string with section names as headers and guidance as body.
    """
    parts = []
    for section in sections:
        guidance = template.get(section, "")
        if guidance:
            header = section.replace("_", " ").title()
            parts.append(f"[{header}]\n{guidance}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Patient / admission data utilities
# ---------------------------------------------------------------------------


def clean_patient_details(patient: dict) -> dict:
    """Sanitise a patient details dict for use in prompts.

    Removes None values and converts complex types to strings.

    Args:
        patient: Raw patient dict.

    Returns:
        Cleaned dict safe for string interpolation.
    """
    cleaned = {}
    for key, value in patient.items():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            cleaned[key] = ""
        elif isinstance(value, (list, dict)):
            cleaned[key] = json.dumps(value)
        elif isinstance(value, pd.Timestamp):
            cleaned[key] = value.strftime("%Y-%m-%d")
        else:
            cleaned[key] = str(value)
    return cleaned


def clean_event_details(event: dict) -> dict:
    """Sanitise an event details dict for use in prompts.

    Args:
        event: Raw event dict.

    Returns:
        Cleaned dict safe for string interpolation.
    """
    return clean_patient_details(event)


# ---------------------------------------------------------------------------
# Date / time utilities
# ---------------------------------------------------------------------------


def create_admission_window(
    admission_date: str,
    los_days: int,
) -> tuple[str, str]:
    """Create admission and discharge dates from a start date and LOS.

    Args:
        admission_date: ISO date string (YYYY-MM-DD).
        los_days: Length of stay in days.

    Returns:
        Tuple of (admission_date_str, discharge_date_str) as ISO date strings.
    """
    adm_dt = datetime.fromisoformat(admission_date)
    dis_dt = adm_dt + timedelta(days=max(0, los_days))
    return adm_dt.strftime("%Y-%m-%d"), dis_dt.strftime("%Y-%m-%d")


def random_24_hour_time() -> str:
    """Return a random time string in HH:MM format.

    Returns:
        Time string e.g. '14:37'.
    """
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    return f"{hour:02d}:{minute:02d}"


# ---------------------------------------------------------------------------
# Output / reporting utilities
# ---------------------------------------------------------------------------


def build_output_info(
    patients: list[dict],
    admissions: list[dict],
    journeys: list[list[dict]],
    clinical_notes: list[list[dict]],
    code_reflection_reports: list[dict] | None = None,
) -> dict[str, Any]:
    """Build a summary info dict for a completed generation run.

    Args:
        patients: List of generated patient dicts.
        admissions: List of generated admission dicts.
        journeys: List of lists of journey event dicts.
        clinical_notes: List of lists of clinical note dicts.
        code_reflection_reports: Optional list of per-patient backward-pass
            reflection reports (see check_code_reflected /
            main.step_verify_code_reflection), one dict per patient mapping
            "<role>:<code>" (role is 'diagnostic' or 'procedure', since the
            same bare code string can appear in both a diagnostic and a
            procedure code system) -> {'patient':..., 'admission':...,
            'journey':..., 'notes':...}.

    Returns:
        Summary statistics dict.
    """
    total_notes = sum(len(notes) for notes in clinical_notes)
    total_events = sum(len(journey) for journey in journeys)

    note_types: dict[str, int] = {}
    for notes in clinical_notes:
        for note in notes:
            nt = note.get("note_type", "unknown")
            note_types[nt] = note_types.get(nt, 0) + 1

    summary: dict[str, Any] = {
        "n_patients": len(patients),
        "n_admissions": len(admissions),
        "n_journey_events": total_events,
        "n_clinical_notes": total_notes,
        "avg_events_per_patient": total_events / max(1, len(patients)),
        "avg_notes_per_patient": total_notes / max(1, len(patients)),
        "note_type_distribution": note_types,
        "generation_timestamp": datetime.utcnow().isoformat(),
    }

    if code_reflection_reports:
        total_codes = 0
        reflected_codes = 0
        unreflected_codes: list[str] = []
        for report in code_reflection_reports:
            for report_key, result in report.items():
                total_codes += 1
                # report_key is "<role>:<code>"; the summary's
                # unreflected_codes list is a user-facing field that predates
                # the role tag, so strip it back to the bare code here.
                code = report_key.split(":", 1)[1] if ":" in report_key else report_key
                if result["admission"] or result["notes"]:
                    reflected_codes += 1
                else:
                    unreflected_codes.append(code)
        summary["code_reflection_check"] = {
            "n_codes_checked": total_codes,
            "n_codes_reflected": reflected_codes,
            "unreflected_codes": unreflected_codes,
        }

    quality_metrics = average_note_quality_metrics(clinical_notes)
    if quality_metrics:
        summary["note_quality_metrics"] = quality_metrics

    return summary


_QUALITY_METRIC_KEYS = (
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog",
    "word_count",
    "fluency_score",
    "groundedness_score",
    "relevance_score",
)


def average_note_quality_metrics(clinical_notes: list[list[dict]]) -> dict[str, float]:
    """Average any quality-evaluation scores (see evaluate_note) present on notes.

    Notes without evaluation scores (evaluation wasn't run, or a metric
    failed and was left as None) are silently skipped per-metric.

    Args:
        clinical_notes: List of lists of clinical note dicts.

    Returns:
        Dict of avg_<metric> -> mean value, for whichever metrics were
        present on at least one note. Empty dict if none were evaluated.
    """
    values: dict[str, list[float]] = {key: [] for key in _QUALITY_METRIC_KEYS}
    for notes in clinical_notes:
        for note in notes:
            for key in _QUALITY_METRIC_KEYS:
                val = note.get(key)
                if val is not None:
                    values[key].append(val)

    return {
        f"avg_{key}": sum(vals) / len(vals)
        for key, vals in values.items()
        if vals
    }


# ---------------------------------------------------------------------------
# Code-driven generation
# ---------------------------------------------------------------------------


def _build_code_context(
    system: Any,
    code: str,
    enable_research: bool,
    model: str | None,
) -> str:
    """Build LLM-prompt-ready context for a single code.

    For codes not in the system's curated dictionary, uses web-search-backed
    research (src/codes/research.py) when enable_research is True, falling
    back to the generic "code not found" message otherwise or if research
    fails to produce a confident result.
    """
    from src.codes import registry  # noqa: PLC0415

    if enable_research and registry.lookup_code(system, code) is None:
        from src.codes import research  # noqa: PLC0415

        researched = research.get_researched_clinical_context(system, code, model=model)
        if researched is not None:
            return researched

    return registry.get_clinical_context(system, code)


def generate_from_codes(
    diagnostic_codes: list[str],
    procedure_codes: list[str],
    patient_details: dict,
    admission_date: str,
    admission_time: str,
    model: str | None = None,
    diagnostic_code_system: str = "icd10",
    procedure_code_system: str = "opcs4",
    enable_code_research: bool = False,
    min_los_days: int = 1,
    max_los_days: int = 14,
) -> dict:
    """Generate an admission record driven by diagnostic and/or procedure codes.

    Works with any registered code system (see src/codes/registry.py), not
    just ICD-10/OPCS-4 - diagnostic_code_system and procedure_code_system
    select which registered CodeSystem the codes belong to. Built-in
    systems are 'icd10' (diagnostic) and 'opcs4' (procedure); additional
    standards (ICD-11, SNOMED CT, CPT, etc.) can be added by registering a
    new CodeSystem, with no changes required here.

    If neither diagnostic_codes nor procedure_codes are given, falls back
    to a generic emergency admission prompt with a random presentation,
    guided by min_los_days/max_los_days (codes with a curated or researched
    typical_los_days already carry their own LOS guidance, so those bounds
    only apply to this uncoded fallback path).

    Args:
        diagnostic_codes: List of diagnostic codes (may be empty).
        procedure_codes: List of procedure codes (may be empty).
        patient_details: Dict of patient demographics.
        admission_date: ISO date string (YYYY-MM-DD).
        admission_time: HH:MM time string.
        model: Optional model override.
        diagnostic_code_system: Registry key for the diagnostic code system
            the codes belong to (default 'icd10').
        procedure_code_system: Registry key for the procedure code system
            the codes belong to (default 'opcs4').
        enable_code_research: If True, codes with no curated dictionary
            entry are researched via web search before generation (adds
            latency/cost per unique uncurated code - see src/codes/research.py).
        min_los_days: Minimum length of stay to suggest for the no-codes
            fallback path (default 1).
        max_los_days: Maximum length of stay to suggest for the no-codes
            fallback path (default 14).

    Returns:
        Dict containing generated admission data, enriched with
        diagnostic_codes/procedure_codes and the code system keys used.
    """
    from src.codes import registry  # noqa: PLC0415
    from src.prompts import patient_prompts  # noqa: PLC0415

    patient_str = json.dumps(clean_patient_details(patient_details), indent=2)

    if diagnostic_codes or procedure_codes:
        diag_system = registry.get_code_system(diagnostic_code_system)
        proc_system = registry.get_code_system(procedure_code_system)

        diagnoses_str = (
            "\n".join(
                _build_code_context(diag_system, c, enable_code_research, model)
                for c in diagnostic_codes
            )
            or "None specified."
        )
        procedures_str = (
            "\n".join(
                _build_code_context(proc_system, c, enable_code_research, model)
                for c in procedure_codes
            )
            or "None specified."
        )

        prompt = patient_prompts["code_driven_admission_prompt"].substitute(
            PATIENT_DETAILS=patient_str,
            DIAGNOSTIC_CODE_SYSTEM=diag_system.name,
            DIAGNOSES_CONTEXT=diagnoses_str,
            PROCEDURE_CODE_SYSTEM=proc_system.name,
            PROCEDURES_CONTEXT=procedures_str,
            ADMISSION_DATE=admission_date,
            ADMISSION_TIME=admission_time,
        )
    else:
        # No codes provided: use a generic emergency admission prompt
        prompt = patient_prompts["emergency_admission_prompt"].substitute(
            PATIENT_DETAILS=patient_str,
            DIAGNOSIS="Acute medical presentation (unspecified)",
            CHIEF_COMPLAINT="Acute illness requiring hospital admission",
            CONSULTANT="Dr Smith",
            SPECIALTY="General Medicine",
            ADMISSION_DATE=admission_date,
            ADMISSION_TIME=admission_time,
            MIN_LOS_DAYS=str(min_los_days),
            MAX_LOS_DAYS=str(max_los_days),
        )

    response = call_llm(prompt, model=model, temp=0.8)

    try:
        admission_data = parse_llm_json(response)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse admission JSON: %s", exc)
        admission_data = {
            "raw_response": response,
            "admission_date": admission_date,
            "parse_error": str(exc),
        }

    # Enrich with code metadata
    admission_data["diagnostic_codes"] = diagnostic_codes
    admission_data["diagnostic_code_system"] = diagnostic_code_system if diagnostic_codes else None
    admission_data["procedure_codes"] = procedure_codes
    admission_data["procedure_code_system"] = procedure_code_system if procedure_codes else None
    return admission_data


def get_code_context(
    code_system: str,
    code: str,
    enable_research: bool = False,
    model: str | None = None,
) -> str:
    """Resolve a registered code system by key and build LLM-ready context for a code.

    Public entry point for callers outside this module (e.g. main.py's
    corrective retry step) that only have a code system key string, not a
    CodeSystem instance.

    Args:
        code_system: Registry key of the code system (e.g. 'icd10').
        code: The code to build context for.
        enable_research: If True, research uncurated codes via web search.
        model: Optional model override for the research synthesis call.

    Returns:
        LLM-prompt-ready context string.
    """
    from src.codes import registry  # noqa: PLC0415

    system = registry.get_code_system(code_system)
    return _build_code_context(system, code, enable_research, model)


# ---------------------------------------------------------------------------
# Targeted backward-pass correction
# ---------------------------------------------------------------------------


def correct_admission_for_code(
    admission: dict,
    code: str,
    code_context: str,
    model: str | None = None,
) -> dict:
    """Revise an admission record so it explicitly reflects a given code.

    Used as a targeted corrective step after check_code_reflected finds a
    driving code missing from the admission's clinical narrative - re-runs
    generation for just this admission rather than the whole patient.

    Args:
        admission: Current admission dict.
        code: The code that should be reflected.
        code_context: LLM-ready clinical context for the code (see
            get_code_context).
        model: Optional model override.

    Returns:
        A new admission dict with narrative fields revised. Bookkeeping
        fields (diagnostic_codes/procedure_codes/*_system) are preserved
        from the original admission unchanged. On any failure (LLM error,
        unparseable response), returns the original admission unchanged.
    """
    from src.prompts import correction_prompts  # noqa: PLC0415

    narrative = {k: v for k, v in admission.items() if k not in _ADMISSION_CODE_BOOKKEEPING_FIELDS}
    prompt = correction_prompts["correct_admission_prompt"].substitute(
        CODE=code,
        CODE_CONTEXT=code_context,
        CURRENT_ADMISSION=json.dumps(narrative, indent=2, default=str),
    )

    try:
        response = call_llm(prompt, model=model, temp=0.4)
        revised_narrative = parse_llm_json(response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Admission correction failed for code %s: %s", code, exc)
        return admission

    if not isinstance(revised_narrative, dict):
        logger.warning("Admission correction for code %s returned non-dict JSON, discarding", code)
        return admission

    corrected = dict(admission)
    corrected.update(revised_narrative)
    # Bookkeeping fields are ours to manage, not the LLM's - restore them
    # in case the model echoed or altered them despite not being asked to.
    for field in _ADMISSION_CODE_BOOKKEEPING_FIELDS:
        if field in admission:
            corrected[field] = admission[field]
    return corrected


def correct_note_for_code(
    note_text: str,
    code: str,
    code_context: str,
    model: str | None = None,
) -> str:
    """Revise a clinical note's text so it explicitly reflects a given code.

    Args:
        note_text: Current note text.
        code: The code that should be reflected.
        code_context: LLM-ready clinical context for the code (see
            get_code_context).
        model: Optional model override.

    Returns:
        Revised note text, or the original text unchanged on any failure
        (LLM error, empty response).
    """
    from src.prompts import correction_prompts  # noqa: PLC0415

    prompt = correction_prompts["correct_note_prompt"].substitute(
        CODE=code,
        CODE_CONTEXT=code_context,
        CURRENT_NOTE=note_text,
    )

    try:
        revised = call_llm(prompt, model=model, temp=0.4)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Note correction failed for code %s: %s", code, exc)
        return note_text

    revised = revised.strip()
    return revised if revised else note_text


# ---------------------------------------------------------------------------
# Diagnosis reflection check (backward pass)
# ---------------------------------------------------------------------------


# Bookkeeping fields generate_from_codes mechanically attaches to every
# admission record. Excluded from the "admission" reflection check below,
# otherwise the check would trivially pass on these alone - the point is to
# confirm the code made it into the *clinical narrative*, not just that it
# was echoed back as metadata.
_ADMISSION_CODE_BOOKKEEPING_FIELDS = frozenset(
    {"diagnostic_codes", "diagnostic_code_system", "procedure_codes", "procedure_code_system"}
)


def _check_code_reflected(
    code: str,
    description: str,
    patient: dict,
    admission: dict,
    journey: list[dict],
    notes: list[dict],
) -> dict[str, bool]:
    """Shared matching logic for check_diagnosis_reflected / check_procedure_reflected."""
    needles = {code.strip().lower()}
    needles.update(word.lower() for word in re.findall(r"[A-Za-z]{4,}", description))

    def _contains(part: Any) -> bool:
        haystack = json.dumps(part, default=str).lower()
        return any(needle in haystack for needle in needles)

    admission_narrative = {
        k: v for k, v in admission.items() if k not in _ADMISSION_CODE_BOOKKEEPING_FIELDS
    }

    return {
        "patient": _contains(patient),
        "admission": _contains(admission_narrative),
        "journey": _contains(journey),
        "notes": _contains(notes),
    }


def check_code_reflected(
    code: str,
    code_system: str,
    patient: dict,
    admission: dict,
    journey: list[dict],
    notes: list[dict],
) -> dict[str, bool]:
    """Backward-pass check for a code from any registered system.

    The forward pass looks up a code (see generate_from_codes) and pushes
    its description into the generation prompts. This is the matching
    backward pass: given the generated patient, admission, journey, and
    notes, confirm the code actually made it into each part's text content.
    Works for any code system registered in src/codes/registry.py, not just
    ICD-10/OPCS-4.

    Args:
        code: Code to search for (e.g. 'I21.0').
        code_system: Registry key of the code system the code belongs to
            (e.g. 'icd10', 'opcs4').
        patient: Generated patient dict.
        admission: Generated admission dict.
        journey: Generated list of journey event dicts.
        notes: Generated list of clinical note dicts.

    Returns:
        Dict with keys 'patient', 'admission', 'journey', 'notes', each True
        if the code or a keyword from its description appears in that part.
    """
    from src.codes import registry  # noqa: PLC0415

    system = registry.get_code_system(code_system)
    info = registry.lookup_code(system, code)
    description = info["description"] if info else ""
    return _check_code_reflected(code, description, patient, admission, journey, notes)


def check_diagnosis_reflected(
    icd10_code: str,
    patient: dict,
    admission: dict,
    journey: list[dict],
    notes: list[dict],
    code_system: str = "icd10",
) -> dict[str, bool]:
    """Backward-pass check for a diagnosis across generated output.

    Convenience wrapper around check_code_reflected, defaulting to ICD-10.
    Pass code_system to check a diagnosis from a different registered
    diagnostic standard (e.g. 'icd11', 'snomed-ct').

    Args:
        icd10_code: Diagnostic code to search for (e.g. 'I21.0').
        patient: Generated patient dict.
        admission: Generated admission dict.
        journey: Generated list of journey event dicts.
        notes: Generated list of clinical note dicts.
        code_system: Registry key of the diagnostic code system (default 'icd10').

    Returns:
        Dict with keys 'patient', 'admission', 'journey', 'notes', each True
        if the code or a keyword from its description appears in that part.
    """
    return check_code_reflected(icd10_code, code_system, patient, admission, journey, notes)


def check_procedure_reflected(
    opcs4_code: str,
    patient: dict,
    admission: dict,
    journey: list[dict],
    notes: list[dict],
    code_system: str = "opcs4",
) -> dict[str, bool]:
    """Backward-pass check for a procedure across generated output.

    Convenience wrapper around check_code_reflected, defaulting to OPCS-4.
    Pass code_system to check a procedure from a different registered
    procedure standard (e.g. 'cpt', 'hcpcs').

    Args:
        opcs4_code: Procedure code to search for (e.g. 'K40.1').
        patient: Generated patient dict.
        admission: Generated admission dict.
        journey: Generated list of journey event dicts.
        notes: Generated list of clinical note dicts.
        code_system: Registry key of the procedure code system (default 'opcs4').

    Returns:
        Dict with keys 'patient', 'admission', 'journey', 'notes', each True
        if the code or a keyword from its description appears in that part.
    """
    return check_code_reflected(opcs4_code, code_system, patient, admission, journey, notes)


# ---------------------------------------------------------------------------
# Generate patient record (demographics)
# ---------------------------------------------------------------------------


def generate_patient(model: str | None = None) -> dict:
    """Generate a synthetic patient record using the LLM.

    Args:
        model: Optional model override.

    Returns:
        Dict of patient demographics.
    """
    from src.prompts import patient_prompts  # noqa: PLC0415

    prompt = patient_prompts["generate_patient_prompt"].template
    response = call_llm(prompt, model=model, temp=0.9)
    try:
        patient = parse_llm_json(response)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse patient JSON: %s", exc)
        patient = {
            "full_name": "Test Patient",
            "first_name": "Test",
            "surname": "Patient",
            "age": 65,
            "sex": "Male",
            "nhs_number": "123 456 7890",
            "mrn": "1234567",
            "parse_error": str(exc),
        }

    # Ensure required fields present
    patient.setdefault("person_id", str(uuid.uuid4()))
    patient.setdefault("patient_id", patient["person_id"])
    return patient


# ---------------------------------------------------------------------------
# Generate patient journey (sequence of events)
# ---------------------------------------------------------------------------


def generate_journey(
    patient: dict,
    admission: dict,
    admission_date: str,
    discharge_date: str,
    possible_event_types: list[str],
    model: str | None = None,
    target_event_count: int = 8,
) -> list[dict]:
    """Generate a sequence of clinical events for a patient journey.

    Args:
        patient: Patient demographics dict.
        admission: Admission details dict.
        admission_date: ISO date string.
        discharge_date: ISO date string.
        possible_event_types: List of valid event type names.
        model: Optional model override.
        target_event_count: Approximate number of events to aim for - a
            soft guide passed into the prompt, not a hard cap. The LLM is
            explicitly told actual clinical realism (LOS, admission type)
            takes priority over hitting this exactly.

    Returns:
        List of event dicts ordered chronologically.
    """
    from src.prompts import journey_prompts  # noqa: PLC0415

    patient_str = json.dumps(clean_patient_details(patient), indent=2)
    admission_str = json.dumps(clean_event_details(admission), indent=2)
    event_types_str = "\n".join(f"- {et}" for et in possible_event_types)

    prompt = journey_prompts["simple_patient_journey_prompt"].substitute(
        PATIENT_DETAILS=patient_str,
        ADMISSION_DETAILS=admission_str,
        ADMISSION_DATE=admission_date,
        DISCHARGE_DATE=discharge_date,
        POSSIBLE_EVENT_TYPES=event_types_str,
        TARGET_EVENT_COUNT=str(target_event_count),
    )

    response = call_llm(prompt, model=model, temp=0.8)
    try:
        journey = parse_llm_json(response)
        if not isinstance(journey, list):
            journey = [journey]
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse journey JSON: %s", exc)
        journey = [
            {
                "event_type": "general ward round",
                "event_date": admission_date,
                "event_time": "09:00",
                "event_order": 1,
                "brief_description": "Admission ward round",
                "clinician_type": "Consultant",
            }
        ]

    return sorted(journey, key=lambda e: (e.get("event_date", ""), e.get("event_time", "")))


# ---------------------------------------------------------------------------
# Generate a clinical note for a specific event
# ---------------------------------------------------------------------------


def generate_clinical_note(
    patient: dict,
    admission: dict,
    event: dict,
    previous_events: list[dict],
    note_template_str: str,
    style_instructions: str,
    add_examination: bool = True,
    red_flags: str = "None identified",
    model: str | None = None,
) -> str:
    """Generate a clinical note for a specific patient journey event.

    Args:
        patient: Patient demographics dict.
        admission: Admission details dict.
        event: Current event details dict.
        previous_events: Earlier events in the journey (for context).
        note_template_str: Formatted template string from combine_template_sections.
        style_instructions: Clinical writing style guidance string.
        add_examination: Whether to include examination findings.
        red_flags: String describing clinical red flags or "None identified".
        model: Optional model override.

    Returns:
        Generated clinical note text.
    """
    from src.prompts import note_prompts  # noqa: PLC0415

    patient_str = json.dumps(clean_patient_details(patient), indent=2)
    admission_str = json.dumps(clean_event_details(admission), indent=2)
    event_str = json.dumps(clean_event_details(event), indent=2)
    prev_str = json.dumps(
        [clean_event_details(e) for e in previous_events[-5:]], indent=2
    )

    prompt = note_prompts["clinical_note_prompt"].substitute(
        PATIENT_DETAILS=patient_str,
        ADMISSION_DETAILS=admission_str,
        EVENT_DETAILS=event_str,
        PREVIOUS_EVENTS=prev_str,
        NOTE_TEMPLATE=note_template_str,
        STYLE_INSTRUCTIONS=style_instructions,
        ADD_EXAMINATION=str(add_examination),
        RED_FLAGS=red_flags,
    )

    return call_llm(prompt, model=model, temp=0.85)


# ---------------------------------------------------------------------------
# Quality evaluation metrics
# ---------------------------------------------------------------------------


def calculate_readability_metrics(note_text: str) -> dict[str, Any]:
    """Compute objective readability metrics for a note (no LLM call).

    Args:
        note_text: Clinical note text.

    Returns:
        Dict with flesch_reading_ease, flesch_kincaid_grade, gunning_fog,
        and word_count, or an empty dict if note_text is blank.
    """
    if not note_text or not note_text.strip():
        return {}

    import textstat  # noqa: PLC0415

    return {
        "flesch_reading_ease": textstat.flesch_reading_ease(note_text),
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade(note_text),
        "gunning_fog": textstat.gunning_fog(note_text),
        "word_count": textstat.lexicon_count(note_text),
    }


def calculate_fluency(note_text: str, model: str | None = None) -> dict[str, Any]:
    """LLM-judged linguistic fluency of a note (grammar, register, coherence).

    Args:
        note_text: Clinical note text.
        model: Optional model override.

    Returns:
        Parsed JSON dict per evaluation_prompts['calculate_fluency_prompt']
        (fluency_score, grammar_score, clinical_register_score, etc.).
    """
    from src.prompts import evaluation_prompts  # noqa: PLC0415

    prompt = evaluation_prompts["calculate_fluency_prompt"].substitute(NOTE=note_text)
    response = call_llm(prompt, model=model, temp=0.2)
    return parse_llm_json(response)


def calculate_groundedness(note_text: str, reference_material: str, model: str | None = None) -> dict[str, Any]:
    """LLM-judged groundedness of a note against its reference context.

    Args:
        note_text: Clinical note text.
        reference_material: Reference context (e.g. admission/patient JSON)
            the note should be factually consistent with.
        model: Optional model override.

    Returns:
        Parsed JSON dict per evaluation_prompts['calculate_groundedness_prompt']
        (groundedness_score, is_grounded, hallucinations, etc.).
    """
    from src.prompts import evaluation_prompts  # noqa: PLC0415

    prompt = evaluation_prompts["calculate_groundedness_prompt"].substitute(
        NOTE=note_text, REFERENCE=reference_material
    )
    response = call_llm(prompt, model=model, temp=0.2)
    return parse_llm_json(response)


def calculate_relevance(note_text: str, reference_material: str, model: str | None = None) -> dict[str, Any]:
    """LLM-judged clinical relevance of a note to its stated context.

    Args:
        note_text: Clinical note text.
        reference_material: Reference context (e.g. admission/patient JSON)
            the note's content should be relevant to.
        model: Optional model override.

    Returns:
        Parsed JSON dict per evaluation_prompts['calculate_relevance_prompt']
        (relevance_score, is_relevant, missing_elements, etc.).
    """
    from src.prompts import evaluation_prompts  # noqa: PLC0415

    prompt = evaluation_prompts["calculate_relevance_prompt"].substitute(
        NOTE=note_text, REFERENCE=reference_material
    )
    response = call_llm(prompt, model=model, temp=0.2)
    return parse_llm_json(response)


def evaluate_note(
    note_text: str,
    reference_material: str,
    model: str | None = None,
) -> dict[str, Any]:
    """Evaluate a generated note: objective readability plus LLM-judged
    fluency/groundedness/relevance.

    Combines calculate_readability_metrics (free, always computed) with
    calculate_fluency/calculate_groundedness/calculate_relevance (3 LLM
    calls - meant to be gated behind an opt-in flag by callers, since this
    triples the LLM cost per note evaluated). Returns a flat dict so the
    scores can be merged directly onto a note record and flow straight
    into CSV output as extra columns. Any individual metric that fails to
    compute is set to None rather than aborting the whole evaluation.

    Args:
        note_text: Clinical note text to evaluate.
        reference_material: Reference context (e.g. admission/patient JSON)
            to check groundedness/relevance against.
        model: Optional model override.

    Returns:
        Flat dict: readability keys plus fluency_score, groundedness_score,
        relevance_score.
    """
    metrics: dict[str, Any] = calculate_readability_metrics(note_text)

    try:
        metrics["fluency_score"] = calculate_fluency(note_text, model=model).get("fluency_score")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fluency evaluation failed: %s", exc)
        metrics["fluency_score"] = None

    try:
        metrics["groundedness_score"] = calculate_groundedness(
            note_text, reference_material, model=model
        ).get("groundedness_score")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groundedness evaluation failed: %s", exc)
        metrics["groundedness_score"] = None

    try:
        metrics["relevance_score"] = calculate_relevance(
            note_text, reference_material, model=model
        ).get("relevance_score")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Relevance evaluation failed: %s", exc)
        metrics["relevance_score"] = None

    return metrics
