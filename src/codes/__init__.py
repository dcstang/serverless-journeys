"""Diagnostic/procedure code handling.

Importing this package registers every built-in CodeSystem (and any
operator-supplied ones from $EXTRA_CODE_SYSTEMS_DIR) by discovering the
JSON files under code_systems/ - see src.codes.loader for the file format
and src.codes.registry for the CodeSystem abstraction itself.
"""

from __future__ import annotations

from src.codes.loader import bootstrap_default_code_systems

bootstrap_default_code_systems()
