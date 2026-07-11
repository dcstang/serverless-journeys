"""ICD-10 code handling for synthetic clinical note generation.

ICD-10 (International Classification of Diseases, 10th Revision) codes follow
the format: letter + 2 digits + optional decimal + digit(s), e.g. I21.0, J18.1.

The curated code dictionary lives in code_systems/icd10.json, not here - see
src.codes.loader for the file format. Importing src.codes (which happens
before this module can be imported) discovers that file and registers it
under the key 'icd10'. The functions below are thin wrappers around the
generic registry implementation, kept for convenience and backward
compatibility - equivalent behaviour is available for any registered code
system via src.codes.registry directly.
"""

from __future__ import annotations

from src.codes import registry

CODE_SYSTEM_KEY = "icd10"


def lookup_code(code: str) -> dict | None:
    """Return the metadata dict for an ICD-10 code, or None if not found.

    Performs a case-insensitive lookup and also tries the uppercased code.
    """
    return registry.lookup_code(registry.get_code_system(CODE_SYSTEM_KEY), code)


def get_clinical_context(code: str) -> str:
    """Return a rich text description of the ICD-10 code suitable for LLM prompts.

    Example return value:
        "ICD-10 I21.0 (Acute transmural myocardial infarction of anterior wall):
         This patient is being managed as an emergency case. Specialty: Cardiology.
         Typical length of stay: 4-7 days. ..."
    """
    return registry.get_clinical_context(registry.get_code_system(CODE_SYSTEM_KEY), code)


def infer_specialty(code: str) -> str:
    """Return the most likely clinical specialty for the given ICD-10 code.

    Falls back to a chapter-based heuristic if the code is not in the dictionary.
    """
    return registry.infer_specialty(registry.get_code_system(CODE_SYSTEM_KEY), code)


def parse_codes(codes_str: str) -> list[str]:
    """Parse a comma-separated string of ICD-10 codes into a list.

    Strips whitespace and filters out empty strings.

    Args:
        codes_str: Comma-separated codes, e.g. "I21.0, J18.1, K35.2"

    Returns:
        List of uppercased code strings, e.g. ["I21.0", "J18.1", "K35.2"]
    """
    return registry.parse_codes(codes_str)
