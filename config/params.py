"""
User-configurable parameters for synthetic clinical note generation.

Adapted from nhsengland/synthetic_clinical_notes for use with
Anthropic Claude / OpenAI APIs and Nebius serverless deployment.

All parameters can be overridden via environment variables or CLI arguments
in main.py. Environment variable names are documented inline.
"""

import os

# ---------------------------------------------------------------------------
# LLM provider and model configuration
# ---------------------------------------------------------------------------

# LLM provider: 'anthropic' (default), 'openai', or 'nebius'
# Env var: LLM_PROVIDER
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "anthropic")

# Default model to use for generation
# Anthropic default: claude-sonnet-4-6
# OpenAI default:    gpt-4o
# Nebius default:    meta-llama/Meta-Llama-3.1-70B-Instruct-fast
# Env var: MODEL
_default_models = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4o",
    "nebius": "meta-llama/Meta-Llama-3.1-70B-Instruct-fast",
}
MODEL: str = os.environ.get("MODEL", _default_models.get(LLM_PROVIDER, "claude-sonnet-4-6"))

# Nebius vLLM endpoint base URL
# Env var: NEBIUS_BASE_URL
NEBIUS_BASE_URL: str = os.environ.get("NEBIUS_BASE_URL", "https://api.studio.nebius.com/v1/")

# Sampling temperature for main generation (higher = more varied output)
TEMPERATURE: float = float(os.environ.get("TEMPERATURE", "0.8"))

# Maximum LLM call retries on failure
MAX_LLM_ATTEMPTS: int = int(os.environ.get("MAX_LLM_ATTEMPTS", "5"))

# ---------------------------------------------------------------------------
# Code-driven generation: ICD-10 and OPCS-4 codes
# ---------------------------------------------------------------------------

# Comma-separated list of ICD-10 codes to drive patient admission generation.
# If empty, admissions are generated without a specific diagnosis code.
# Example: "I21.0,J18.1,K35.2"
# Env var: ICD10_CODES
ICD10_CODES: list[str] = [
    c.strip()
    for c in os.environ.get("ICD10_CODES", "").split(",")
    if c.strip()
]

# Comma-separated list of OPCS-4 procedure codes.
# If empty, no specific procedure is coded.
# Example: "K40.1,W37.1"
# Env var: OPCS4_CODES
OPCS4_CODES: list[str] = [
    c.strip()
    for c in os.environ.get("OPCS4_CODES", "").split(",")
    if c.strip()
]

# ---------------------------------------------------------------------------
# Generation volume parameters
# ---------------------------------------------------------------------------

# Number of synthetic patients to generate per run
# Env var: N_PATIENTS
N_PATIENTS: int = int(os.environ.get("N_PATIENTS", "5"))

# Number of journey events to generate per patient (approximate)
# Actual number depends on LOS and admission type
N_EVENTS_PER_PATIENT: int = int(os.environ.get("N_EVENTS_PER_PATIENT", "8"))

# Minimum and maximum length of stay (days) when not determined by codes
MIN_LOS_DAYS: int = int(os.environ.get("MIN_LOS_DAYS", "1"))
MAX_LOS_DAYS: int = int(os.environ.get("MAX_LOS_DAYS", "14"))

# Default admission date if not specified (ISO format YYYY-MM-DD)
# Defaults to today's date at runtime
import datetime as _dt  # noqa: E402
DEFAULT_ADMISSION_DATE: str = os.environ.get(
    "ADMISSION_DATE", _dt.date.today().isoformat()
)

# ---------------------------------------------------------------------------
# Output configuration
# ---------------------------------------------------------------------------

# Output directory for generated CSV files
# Env var: OUTPUT_DIR
OUTPUT_DIR: str = os.environ.get("OUTPUT_DIR", "data/output")

# Output filenames
OUTPUT_PATIENTS_FILE: str = "synthetic_patients.csv"
OUTPUT_ADMISSIONS_FILE: str = "synthetic_admissions.csv"
OUTPUT_NOTES_FILE: str = "synthetic_clinical_notes.csv"
OUTPUT_JOURNEYS_FILE: str = "synthetic_journeys.csv"
OUTPUT_SUMMARY_FILE: str = "generation_summary.json"

# ---------------------------------------------------------------------------
# Augmentation parameters
# ---------------------------------------------------------------------------

# Whether to apply abbreviation augmentation to generated notes
APPLY_ABBREVIATIONS: bool = (
    os.environ.get("APPLY_ABBREVIATIONS", "true").lower() == "true"
)

# Whether to apply typo augmentation to generated notes
APPLY_TYPOS: bool = (
    os.environ.get("APPLY_TYPOS", "false").lower() == "true"
)

# Typo rate: proportion of words to corrupt (0.0 - 1.0)
TYPO_RATE: float = float(os.environ.get("TYPO_RATE", "0.02"))

# ---------------------------------------------------------------------------
# Admission type distribution (when generating without specific codes)
# ---------------------------------------------------------------------------

# Proportion of admissions that should be emergency vs elective
# (1.0 = all emergency, 0.0 = all elective)
EMERGENCY_ADMISSION_RATE: float = float(
    os.environ.get("EMERGENCY_ADMISSION_RATE", "0.6")
)

# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

# Maximum number of concurrent LLM calls (for async generation)
MAX_CONCURRENT_CALLS: int = int(os.environ.get("MAX_CONCURRENT_CALLS", "3"))

# ---------------------------------------------------------------------------
# Convenience dict for passing all params around
# ---------------------------------------------------------------------------

PARAMS: dict = {
    "llm_provider": LLM_PROVIDER,
    "model": MODEL,
    "temperature": TEMPERATURE,
    "max_llm_attempts": MAX_LLM_ATTEMPTS,
    "icd10_codes": ICD10_CODES,
    "opcs4_codes": OPCS4_CODES,
    "n_patients": N_PATIENTS,
    "n_events_per_patient": N_EVENTS_PER_PATIENT,
    "min_los_days": MIN_LOS_DAYS,
    "max_los_days": MAX_LOS_DAYS,
    "default_admission_date": DEFAULT_ADMISSION_DATE,
    "output_dir": OUTPUT_DIR,
    "output_patients_file": OUTPUT_PATIENTS_FILE,
    "output_admissions_file": OUTPUT_ADMISSIONS_FILE,
    "output_notes_file": OUTPUT_NOTES_FILE,
    "output_journeys_file": OUTPUT_JOURNEYS_FILE,
    "output_summary_file": OUTPUT_SUMMARY_FILE,
    "apply_abbreviations": APPLY_ABBREVIATIONS,
    "apply_typos": APPLY_TYPOS,
    "typo_rate": TYPO_RATE,
    "emergency_admission_rate": EMERGENCY_ADMISSION_RATE,
    "max_concurrent_calls": MAX_CONCURRENT_CALLS,
}
