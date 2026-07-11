"""Loads CodeSystems from data-only JSON files instead of Python source.

This is the mechanism behind "bring your own coding standard" - a
contributor (or a Nebius job, via EXTRA_CODE_SYSTEMS_DIR) adds a JSON file
shaped like the schema below and it becomes a fully working CodeSystem with
no Python code required.

JSON schema (one CodeSystem per file):
    {
      "key": "icd10",                        # required, unique registry key
      "name": "ICD-10",                      # required, display name
      "kind": "diagnostic",                  # required, "diagnostic" | "procedure"
      "specialty_field": "specialty",        # optional, default "specialty"
      "type_field": "admission_type",        # optional, default null
      "default_specialty": "General Medicine", # optional, default "General Medicine"
      "chapter_map": {"A": "Infectious Diseases"}, # optional, default {}
      "codes": {                             # required (may be {})
        "I21.0": {
          "description": "...",
          "specialty": "...",                # must match specialty_field
          "typical_los_days": [4, 7]
        }
      }
    }

Two directories are searched by default (see default_code_system_dirs):
  1. <repo root>/code_systems/ - baked into the container image, the place
     to contribute a new standard via a normal pull request.
  2. $EXTRA_CODE_SYSTEMS_DIR, if set - an operator-supplied directory (e.g.
     a mounted volume in a Nebius job) for private/local standards that
     shouldn't go through the image build at all. Files here can also
     override a built-in system by reusing its key.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from src.codes.registry import CodeSystem, register_code_system

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS = ("key", "name", "kind", "codes")
_VALID_KINDS = ("diagnostic", "procedure")


class CodeSystemFileError(ValueError):
    """Raised when a code-system JSON file is missing or malformed."""


def _code_system_from_dict(data: Any, source: str) -> CodeSystem:
    """Validate and build a CodeSystem from a parsed JSON object.

    Args:
        data: Parsed JSON content.
        source: File path, used only to make error messages actionable.

    Raises:
        CodeSystemFileError: If `data` doesn't match the code-system schema.
    """
    if not isinstance(data, dict):
        raise CodeSystemFileError(f"{source}: top-level JSON value must be an object")

    missing = [f for f in _REQUIRED_FIELDS if f not in data]
    if missing:
        raise CodeSystemFileError(f"{source}: missing required field(s): {', '.join(missing)}")

    kind = data["kind"]
    if kind not in _VALID_KINDS:
        raise CodeSystemFileError(
            f"{source}: 'kind' must be one of {_VALID_KINDS}, got {kind!r}"
        )

    codes = data["codes"]
    if not isinstance(codes, dict):
        raise CodeSystemFileError(f"{source}: 'codes' must be an object")

    chapter_map = data.get("chapter_map", {})
    if not isinstance(chapter_map, dict):
        raise CodeSystemFileError(f"{source}: 'chapter_map' must be an object")

    try:
        return CodeSystem(
            key=str(data["key"]),
            name=str(data["name"]),
            kind=kind,
            codes=codes,
            specialty_field=str(data.get("specialty_field", "specialty")),
            type_field=data.get("type_field"),
            chapter_map=chapter_map,
            default_specialty=str(data.get("default_specialty", "General Medicine")),
        )
    except (TypeError, ValueError) as exc:
        raise CodeSystemFileError(f"{source}: {exc}") from exc


def load_code_system_file(path: str | Path) -> CodeSystem:
    """Parse and validate a single code-system JSON file.

    Args:
        path: Path to a JSON file matching the code-system schema.

    Returns:
        The CodeSystem described by the file (not yet registered).

    Raises:
        CodeSystemFileError: If the file is missing, isn't valid JSON, or
            doesn't match the schema.
    """
    path = Path(path)
    try:
        raw = path.read_text()
    except OSError as exc:
        raise CodeSystemFileError(f"{path}: could not read file: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CodeSystemFileError(f"{path}: invalid JSON: {exc}") from exc

    return _code_system_from_dict(data, str(path))


def discover_code_systems(*dirs: str | Path, register: bool = True) -> list[CodeSystem]:
    """Load every *.json file in the given directories as a CodeSystem.

    Directories are processed in order, so a later directory's file can
    override an earlier one's system by reusing the same key (this is how
    EXTRA_CODE_SYSTEMS_DIR overrides a built-in standard). Missing
    directories are skipped silently. A malformed file is logged as a
    warning and skipped rather than aborting the whole run - one bad
    contributed file shouldn't break every other registered standard.

    Args:
        *dirs: Directories to search for *.json files.
        register: If True (default), also register each loaded system.

    Returns:
        The CodeSystems that were successfully loaded, in discovery order.
    """
    loaded: list[CodeSystem] = []
    for directory in dirs:
        directory = Path(directory)
        if not directory.is_dir():
            continue
        for json_path in sorted(directory.glob("*.json")):
            try:
                system = load_code_system_file(json_path)
            except CodeSystemFileError as exc:
                logger.warning("Skipping invalid code system file: %s", exc)
                continue
            if register:
                register_code_system(system)
            loaded.append(system)
            logger.info(
                "Loaded code system '%s' (%s, %d curated codes) from %s",
                system.key, system.kind, len(system.codes), json_path,
            )
    return loaded


def default_code_system_dirs() -> list[Path]:
    """Return the directories searched by bootstrap_default_code_systems.

    1. <repo root>/code_systems/ - built-in standards shipped in the image.
    2. $EXTRA_CODE_SYSTEMS_DIR, if set - operator-supplied standards (e.g.
       a mounted volume in a serverless job), searched second so its files
       can override a built-in system's key.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    dirs = [repo_root / "code_systems"]
    extra = os.environ.get("EXTRA_CODE_SYSTEMS_DIR")
    if extra:
        dirs.append(Path(extra))
    return dirs


def bootstrap_default_code_systems() -> list[str]:
    """Discover and register every code system from the default directories.

    Called once at import time by src/codes/__init__.py, so any code that
    imports something under src.codes ends up with the built-in (and any
    operator-supplied) standards registered - no explicit setup call needed.

    Returns:
        Keys of the code systems that were registered, in load order.
    """
    systems = discover_code_systems(*default_code_system_dirs())
    return [s.key for s in systems]
