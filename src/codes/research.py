"""
Code research: fills in clinical context for diagnostic/procedure codes
that aren't in a CodeSystem's curated dictionary and whose meaning isn't
obvious from the bare code alone - e.g. many real-world ICD-10/OPCS-4/SNOMED
codes outside the small curated set here.

Searches the web (src/codes/search.py) for the code, then has the LLM
synthesize a grounded summary in the same shape as a curated dictionary
entry, so it flows through the rest of the pipeline (registry.get_clinical_context,
registry.infer_specialty, generate_from_codes) exactly like a curated code.

Opt-in: this adds real latency (a search request + an LLM call) per unique
code, so callers should gate it behind an explicit flag rather than always
researching every uncurated code.
"""

from __future__ import annotations

import logging
from typing import Any

from src.codes import registry, search
from src.codes.registry import CodeSystem

logger = logging.getLogger(__name__)

# Cached per (system key, code) for the lifetime of the process, since a
# single pipeline run may drive many patients off the same code and
# re-researching it for each one would be wasted latency/cost.
_RESEARCH_CACHE: dict[tuple[str, str], dict[str, Any] | None] = {}


def clear_cache() -> None:
    """Clear the in-memory research cache (mainly for tests)."""
    _RESEARCH_CACHE.clear()


def research_code(
    system: CodeSystem,
    code: str,
    model: str | None = None,
    num_search_results: int = 5,
) -> dict[str, Any] | None:
    """Research an uncurated code via web search + LLM synthesis.

    Args:
        system: The CodeSystem the code belongs to.
        code: The code to research.
        model: Optional LLM model override for the synthesis call.
        num_search_results: Number of search results to feed to the LLM.

    Returns:
        A metadata dict shaped like a curated CodeSystem.codes entry (a
        'description' key, system.specialty_field, system.type_field if
        set, and 'typical_los_days'), or None if search/synthesis failed or
        produced low-confidence results.
    """
    cache_key = (system.key, code.strip().upper())
    if cache_key in _RESEARCH_CACHE:
        return _RESEARCH_CACHE[cache_key]

    info = _research_code_uncached(system, code, model, num_search_results)
    _RESEARCH_CACHE[cache_key] = info
    return info


def _research_code_uncached(
    system: CodeSystem,
    code: str,
    model: str | None,
    num_search_results: int,
) -> dict[str, Any] | None:
    from src.processing import call_llm, parse_llm_json  # noqa: PLC0415
    from src.prompts import research_prompts  # noqa: PLC0415

    code_upper = code.strip().upper()
    query = f"{system.name} code {code_upper} clinical meaning diagnosis definition NHS"

    try:
        results = search.search(query, num_results=num_search_results)
    except search.SearchUnavailableError as exc:
        logger.warning("Code research search failed for %s %s: %s", system.name, code_upper, exc)
        return None

    if not results:
        logger.info("No search results found for %s %s", system.name, code_upper)
        return None

    results_str = "\n\n".join(
        f"[{i + 1}] {r['title']}\n{r['snippet']}\n({r['link']})" for i, r in enumerate(results)
    )

    prompt = research_prompts["research_code_prompt"].substitute(
        CODE=code_upper,
        CODE_SYSTEM=system.name,
        CODE_KIND=system.kind,
        SEARCH_RESULTS=results_str,
    )

    try:
        response = call_llm(prompt, model=model, temp=0.2)
        parsed = parse_llm_json(response)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Code research synthesis failed for %s %s: %s", system.name, code_upper, exc)
        return None

    if parsed.get("confidence") == "low":
        logger.info("Code research for %s %s returned low confidence, discarding", system.name, code_upper)
        return None

    try:
        info: dict[str, Any] = {
            "description": parsed["description"],
            system.specialty_field: parsed["specialty"],
            "typical_los_days": (
                int(parsed["typical_los_days_min"]),
                int(parsed["typical_los_days_max"]),
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning(
            "Code research for %s %s returned malformed data: %s", system.name, code_upper, exc
        )
        return None

    if system.type_field:
        info[system.type_field] = parsed.get("type", "emergency")

    logger.info("Researched %s %s: %s", system.name, code_upper, info["description"])
    return info


def get_researched_clinical_context(
    system: CodeSystem,
    code: str,
    model: str | None = None,
    num_search_results: int = 5,
) -> str | None:
    """Research an uncurated code and format it as LLM-prompt-ready context.

    Args:
        system: The CodeSystem the code belongs to.
        code: The code to research.
        model: Optional LLM model override for the synthesis call.
        num_search_results: Number of search results to feed to the LLM.

    Returns:
        Formatted context string (see registry.format_code_context), or
        None if research failed - callers should fall back to
        registry.get_clinical_context's generic message in that case.
    """
    info = research_code(system, code, model=model, num_search_results=num_search_results)
    if info is None:
        return None
    context = registry.format_code_context(system, code, info)
    return f"[Web-researched code - verify clinical accuracy] {context}"
