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

import asyncio
import json
import logging
import os
import random
import re
import string
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
    max_attempts: int = 5,
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
        max_attempts: Maximum number of retry attempts on failure.
        chat_history: Optional list of prior message dicts
                      [{'role': 'user'/'assistant', 'content': '...'}].
                      The prompt is appended as the latest user message.

    Returns:
        The model's text response as a string.

    Raises:
        RuntimeError: If all retry attempts fail.
    """
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
# call_llm_async: asynchronous wrapper
# ---------------------------------------------------------------------------


async def call_llm_async(
    prompt: str,
    model: str | None = None,
    temp: float = 0.7,
    max_attempts: int = 5,
    chat_history: list[dict] | None = None,
) -> str:
    """Async wrapper around call_llm using asyncio.to_thread.

    Args:
        prompt: The prompt text.
        model: Optional model override.
        temp: Sampling temperature.
        max_attempts: Max retry attempts.
        chat_history: Optional conversation history.

    Returns:
        The model's text response.
    """
    return await asyncio.to_thread(
        call_llm, prompt, model, temp, max_attempts, chat_history
    )


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
# Data cleaning utilities
# ---------------------------------------------------------------------------


def remove_failures(data: list[Any], failure_value: Any = None) -> list[Any]:
    """Remove failed/null entries from a list.

    Args:
        data: List potentially containing None or failure_value entries.
        failure_value: Additional value to treat as failure.

    Returns:
        Filtered list with failures removed.
    """
    return [item for item in data if item is not None and item != failure_value]


def clean_outputs(text: str, cleaning_type: str, model: str | None = None) -> str:
    """Use the LLM to clean a text output.

    Args:
        text: Text to clean.
        cleaning_type: Type of cleaning to perform (see processing_prompts).
        model: Optional model override.

    Returns:
        Cleaned text string.
    """
    from src.prompts import processing_prompts  # noqa: PLC0415

    prompt = processing_prompts["clean_outputs_prompt"].substitute(
        CLEANING_TYPE=cleaning_type,
        VALUE=text,
    )
    return call_llm(prompt, model=model, temp=0.3)


def clean_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to integer, returning default on failure.

    Args:
        value: Value to convert.
        default: Value to return if conversion fails.

    Returns:
        Integer value.
    """
    if value is None:
        return default
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Abbreviation and typo augmentation
# ---------------------------------------------------------------------------


def add_abbreviations_to_dict(
    data: dict[str, Any],
    sections_to_skip: list[str] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Apply clinical abbreviation augmentation to string values in a dict.

    Args:
        data: Dict with string values to augment.
        sections_to_skip: List of keys to skip during augmentation.
        model: Optional model override.

    Returns:
        New dict with abbreviated string values.
    """
    sections_to_skip = sections_to_skip or []
    result = {}
    for key, value in data.items():
        if key in sections_to_skip or not isinstance(value, str) or len(value) < 50:
            result[key] = value
        else:
            result[key] = add_abbreviations_to_strings(value, model=model)
    return result


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


def add_typos_to_dict(
    data: dict[str, Any],
    sections_to_skip: list[str] | None = None,
    typo_rate: float = 0.02,
) -> dict[str, Any]:
    """Apply realistic typo augmentation to string values in a dict.

    Args:
        data: Dict with string values to augment.
        sections_to_skip: List of keys to skip.
        typo_rate: Approximate proportion of words to introduce typos into.

    Returns:
        New dict with typo-augmented string values.
    """
    sections_to_skip = sections_to_skip or []
    result = {}
    for key, value in data.items():
        if key in sections_to_skip or not isinstance(value, str) or len(value) < 20:
            result[key] = value
        else:
            result[key] = add_typos_to_string(value, typo_rate=typo_rate)
    return result


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


def pop_nested_value(data: dict, key_path: list[str]) -> Any:
    """Remove and return a value from a nested dict using a key path.

    Args:
        data: Nested dict to modify.
        key_path: List of keys forming the path to the value.

    Returns:
        The removed value, or None if the path does not exist.
    """
    if not key_path:
        return None
    current = data
    for key in key_path[:-1]:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current.pop(key_path[-1], None)


def update_nested_value(data: dict, key_path: list[str], value: Any) -> dict:
    """Set a value in a nested dict using a key path, creating dicts as needed.

    Args:
        data: Nested dict to modify (modified in-place).
        key_path: List of keys forming the path.
        value: Value to set.

    Returns:
        The modified dict.
    """
    current = data
    for key in key_path[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[key_path[-1]] = value
    return data


# ---------------------------------------------------------------------------
# Patient / admission data utilities
# ---------------------------------------------------------------------------


def combine_patients_and_admissions(
    patients_df: pd.DataFrame,
    admissions_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge patient demographics with admission data.

    Args:
        patients_df: DataFrame with patient records (must have 'person_id').
        admissions_df: DataFrame with admission records (must have 'patient_id').

    Returns:
        Merged DataFrame.
    """
    return admissions_df.merge(
        patients_df,
        left_on="patient_id",
        right_on="person_id",
        how="left",
        suffixes=("_adm", "_pt"),
    )


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


def normalise_array_struct_column(
    df: pd.DataFrame,
    column: str,
    default: Any = None,
) -> pd.DataFrame:
    """Normalise a column containing JSON strings or Python objects.

    Attempts to parse string values as JSON; replaces parse failures with
    the default value.

    Args:
        df: DataFrame to modify.
        column: Column name to normalise.
        default: Value to use when parsing fails.

    Returns:
        DataFrame with normalised column (modified in-place copy).
    """
    df = df.copy()

    def _parse(val: Any) -> Any:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        if isinstance(val, str):
            try:
                return json.loads(val)
            except (json.JSONDecodeError, ValueError):
                return default
        return val

    df[column] = df[column].apply(_parse)
    return df


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
            code -> {'patient':..., 'admission':..., 'journey':..., 'notes':...}.

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
            for code, result in report.items():
                total_codes += 1
                if result["admission"] or result["notes"]:
                    reflected_codes += 1
                else:
                    unreflected_codes.append(code)
        summary["code_reflection_check"] = {
            "n_codes_checked": total_codes,
            "n_codes_reflected": reflected_codes,
            "unreflected_codes": unreflected_codes,
        }

    return summary


# ---------------------------------------------------------------------------
# read_write_data: CSV-based I/O
# ---------------------------------------------------------------------------


def read_data(
    file_path: str,
    schema: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read a CSV file into a DataFrame, optionally applying a schema.

    Args:
        file_path: Absolute or relative path to the CSV file.
        schema: Optional dtype schema dict (from schemas.py).

    Returns:
        DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    df = pd.read_csv(file_path, low_memory=False)

    if schema:
        for col, dtype in schema.items():
            if col in df.columns:
                try:
                    if dtype in ("string",):
                        df[col] = df[col].astype("string")
                    elif dtype in ("boolean",):
                        df[col] = df[col].astype("boolean")
                    elif dtype in ("Int32", "Int64"):
                        df[col] = pd.array(df[col], dtype=dtype)
                    elif dtype in ("datetime64[ns]",):
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    # object columns left as-is
                except Exception as exc:  # noqa: BLE001
                    logger.debug(
                        "Could not cast column %s to %s: %s", col, dtype, exc
                    )
    return df


def write_data(
    df: pd.DataFrame,
    file_path: str,
    create_dirs: bool = True,
) -> None:
    """Write a DataFrame to a CSV file.

    Args:
        df: DataFrame to write.
        file_path: Destination file path.
        create_dirs: If True, create parent directories if they don't exist.
    """
    if create_dirs:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    df.to_csv(file_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), file_path)


def read_write_data(
    input_path: str | None,
    output_path: str,
    processor_fn,
    schema: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Read data from CSV, apply processor function, write result to CSV.

    This is the main pipeline I/O function. If input_path is None, an empty
    DataFrame is passed to processor_fn.

    Args:
        input_path: Path to input CSV (or None for generation-from-scratch).
        output_path: Path to write output CSV.
        processor_fn: Callable(df) -> df to apply to the data.
        schema: Optional dtype schema for input reading.

    Returns:
        Processed DataFrame.
    """
    if input_path and os.path.exists(input_path):
        df_in = read_data(input_path, schema=schema)
    else:
        df_in = pd.DataFrame()

    df_out = processor_fn(df_in)
    write_data(df_out, output_path)
    return df_out


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
) -> dict:
    """Generate an admission record driven by diagnostic and/or procedure codes.

    Works with any registered code system (see src/codes/registry.py), not
    just ICD-10/OPCS-4 - diagnostic_code_system and procedure_code_system
    select which registered CodeSystem the codes belong to. Built-in
    systems are 'icd10' (diagnostic) and 'opcs4' (procedure); additional
    standards (ICD-11, SNOMED CT, CPT, etc.) can be added by registering a
    new CodeSystem, with no changes required here.

    If neither diagnostic_codes nor procedure_codes are given, falls back
    to a generic emergency admission prompt with a random presentation.

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
