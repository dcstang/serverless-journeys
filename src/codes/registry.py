"""Generic registry for diagnostic and procedure coding standards.

The synthetic generation pipeline should not be hardwired to any one
coding standard. Real NHS/health data uses many: ICD-10 and ICD-11 for
diagnoses, OPCS-4 in the UK, CPT/HCPCS in the US, SNOMED CT for either,
ICD-9-CM in older datasets, and so on.

A CodeSystem is a small, uniform description of one such standard: its
curated code -> metadata dictionary (if any), which field in that metadata
holds the specialty, and a first-letter fallback map for codes outside the
curated set. src/codes/icd10.py and src/codes/opcs4.py build a CodeSystem
from their existing dictionaries and register it here at import time.

To add support for another standard (say, SNOMED CT), build a CodeSystem
the same way in a new module and call register_code_system() - the rest of
the pipeline (prompt generation in src/processing.py, the CLI's
--diagnostic-code-system / --procedure-code-system flags, and the
backward-pass reflection checks) then works with it automatically, with no
further code changes required. A system with an empty `codes` dict is also
valid: every lookup falls back to the generic "not in curated dictionary"
path, so an entirely uncurated standard still works end to end - the LLM
just generates purely from the raw code with no extra clinical context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

CodeKind = Literal["diagnostic", "procedure"]


@dataclass(frozen=True)
class CodeSystem:
    """Describes one diagnostic or procedure coding standard.

    Attributes:
        key: Short identifier used to select this system, e.g. 'icd10'.
        name: Display name used in prompts and messages, e.g. 'ICD-10'.
        kind: Either 'diagnostic' or 'procedure'.
        codes: Curated code -> metadata dict. May be empty for an
            uncurated standard; lookups then always fall back gracefully.
        specialty_field: Key within each code's metadata dict holding the
            clinical/surgical specialty.
        type_field: Key within each code's metadata dict holding an
            admission/procedure type (e.g. 'emergency'/'elective'), or
            None if the standard doesn't distinguish this.
        chapter_map: Fallback specialty by the code's first character,
            used when a code isn't in the curated dictionary.
        default_specialty: Specialty to fall back to when nothing else matches.
    """

    key: str
    name: str
    kind: CodeKind
    codes: dict[str, dict]
    specialty_field: str = "specialty"
    type_field: str | None = None
    chapter_map: dict[str, str] = field(default_factory=dict)
    default_specialty: str = "General Medicine"


_REGISTRY: dict[str, CodeSystem] = {}


def register_code_system(system: CodeSystem) -> None:
    """Register (or replace) a code system under its key."""
    _REGISTRY[system.key.strip().lower()] = system


def get_code_system(key: str) -> CodeSystem:
    """Look up a registered code system by key (case-insensitive).

    Raises:
        KeyError: If no system is registered under that key.
    """
    try:
        return _REGISTRY[key.strip().lower()]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "(none registered)"
        raise KeyError(
            f"Unknown code system '{key}'. Registered systems: {available}"
        ) from None


def list_code_systems(kind: CodeKind | None = None) -> list[str]:
    """List registered code system keys, optionally filtered by kind."""
    return sorted(k for k, s in _REGISTRY.items() if kind is None or s.kind == kind)


def parse_codes(codes_str: str) -> list[str]:
    """Parse a comma-separated code string into a normalised list.

    System-agnostic: the same parsing rules apply regardless of which
    coding standard the codes belong to.

    Args:
        codes_str: Comma-separated codes, e.g. "I21.0, J18.1".

    Returns:
        List of stripped, upper-cased code strings.
    """
    if not codes_str or not codes_str.strip():
        return []
    return [c.strip().upper() for c in codes_str.split(",") if c.strip()]


def lookup_code(system: CodeSystem, code: str) -> dict | None:
    """Return curated metadata for a code within a system, or None.

    Args:
        system: The CodeSystem to look the code up in.
        code: The code string (case-insensitive).

    Returns:
        Metadata dict, or None if not present in the curated dictionary.
    """
    if not code:
        return None
    return system.codes.get(code.strip().upper())


def infer_specialty(system: CodeSystem, code: str) -> str:
    """Return the most likely specialty for a code within a system.

    Falls back to a first-letter chapter heuristic, then the system's
    default specialty, if the code isn't in the curated dictionary.

    Args:
        system: The CodeSystem to consult.
        code: The code string.

    Returns:
        Specialty name string.
    """
    info = lookup_code(system, code)
    if info:
        raw = str(info.get(system.specialty_field, system.default_specialty))
        return raw.split("/")[0].strip()

    if not code:
        return system.default_specialty

    chapter_letter = code.strip().upper()[0]
    return system.chapter_map.get(chapter_letter, system.default_specialty)


def format_code_context(system: CodeSystem, code: str, info: dict) -> str:
    """Format a code's metadata dict into a rich LLM-prompt-ready description.

    Shared by get_clinical_context (curated codes) and code research (see
    src/codes/research.py, which synthesizes an info dict for uncurated
    codes via web search and formats it the same way).

    Args:
        system: The CodeSystem the code belongs to.
        code: The code string.
        info: Metadata dict (same shape as a CodeSystem.codes entry).

    Returns:
        Formatted description string.
    """
    code_upper = code.strip().upper()
    description = info.get("description", code_upper)
    specialty = info.get(system.specialty_field, system.default_specialty)
    los = info.get("typical_los_days")
    type_word = info.get(system.type_field) if system.type_field else None

    parts = [f"{system.name} {code_upper} ({description})."]
    if type_word:
        article = "an" if str(type_word)[0].lower() in "aeiou" else "a"
        parts.append(f"This patient is being managed as {article} {type_word} case.")
    parts.append(f"Specialty: {specialty}.")
    if los:
        low, high = los
        if high <= 1:
            parts.append("Typical length of stay: day case (no overnight stay).")
        else:
            parts.append(f"Typical length of stay: {low}-{high} days.")
    parts.append(
        "Ensure all generated content is clinically appropriate for this code, "
        "including relevant investigations, treatments, and specialist involvement."
    )
    return " ".join(parts)


def get_clinical_context(system: CodeSystem, code: str) -> str:
    """Return a rich text description of a code suitable for LLM prompts.

    Args:
        system: The CodeSystem the code belongs to.
        code: The code string.

    Returns:
        A formatted description, or a generic fallback message when the
        code isn't present in the system's curated dictionary.
    """
    code_upper = code.strip().upper()
    info = lookup_code(system, code_upper)

    if info is None:
        kind_word = "diagnosis" if system.kind == "diagnostic" else "procedure"
        return (
            f"{system.name} {code_upper}: Code not found in reference dictionary. "
            f"Treat this as a {kind_word} code from the {system.name} standard and "
            f"generate clinically appropriate NHS documentation for it, following "
            f"standard UK clinical practice and NHS documentation conventions."
        )

    return format_code_context(system, code_upper, info)
