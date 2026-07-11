"""OPCS-4 code handling module.

OPCS-4 (Office of Population Censuses and Surveys Classification of Interventions
and Procedures, version 4) is the UK classification of surgical and interventional
procedures. Codes follow the format: letter + 2 digits + optional decimal + digit,
e.g. K40.1, W37.1, H01.1.

The curated code dictionary lives in code_systems/opcs4.json, not here - see
src.codes.loader for the file format. Importing src.codes (which happens
before this module can be imported) discovers that file and registers it
under the key 'opcs4'. The functions below are thin wrappers around the
generic registry implementation, kept for convenience and backward
compatibility - equivalent behaviour is available for any registered code
system via src.codes.registry directly.
"""

from __future__ import annotations

from src.codes import registry

CODE_SYSTEM_KEY = "opcs4"


def lookup_code(code: str) -> dict | None:
    """Return the metadata dict for a given OPCS-4 code, or None if not found.

    Performs a case-insensitive lookup.

    Args:
        code: OPCS-4 code string e.g. "K40.1" or "k40.1".

    Returns:
        Dictionary with keys description, chapter_name, surgical_specialty,
        elective_or_emergency, typical_los_days, or None.
    """
    return registry.lookup_code(registry.get_code_system(CODE_SYSTEM_KEY), code)


def get_clinical_context(code: str) -> str:
    """Return a rich text description suitable for inclusion in an LLM prompt.

    Args:
        code: OPCS-4 code string.

    Returns:
        A formatted string describing the surgical/procedural context, or a
        generic message if the code is not found.
    """
    return registry.get_clinical_context(registry.get_code_system(CODE_SYSTEM_KEY), code)


def infer_specialty(code: str) -> str:
    """Return the most likely surgical specialty for a given OPCS-4 code.

    Falls back to a chapter-based heuristic if not in the curated dictionary.

    Args:
        code: OPCS-4 code string.

    Returns:
        Surgical specialty string.
    """
    return registry.infer_specialty(registry.get_code_system(CODE_SYSTEM_KEY), code)


def parse_codes(codes_str: str) -> list[str]:
    """Parse a comma-separated string of OPCS-4 codes into a list.

    Handles whitespace, mixed case, and empty strings gracefully.

    Args:
        codes_str: Comma-separated OPCS-4 codes e.g. "K40.1, W37.1, H01.1".

    Returns:
        List of normalised (upper-case, stripped) code strings.
    """
    return registry.parse_codes(codes_str)
