"""
Static configuration for synthetic clinical note generation.

Contains: event types and per-note-type style instructions.

Adapted from nhsengland/synthetic_clinical_notes.
"""

# ---------------------------------------------------------------------------
# possible_event_types
# List of clinical event types that can appear in a patient journey.
# Mirrors the keys in doc_templates.document_templates.
# ---------------------------------------------------------------------------

possible_event_types: list[str] = [
    "ED event",
    "ED review and hand-over",
    "post take ward round",
    "general ward round",
    "inter-specialty review",
    "operation",
    "pre-op assessment",
    "pre-op consent",
    "pre-op checklist",
    "anaesthetics assessment",
    "post-anaesthesia recovery",
    "nursing",
    "therapy",
    "misc",
]

# ---------------------------------------------------------------------------
# style_instructions
# Per-note-type writing style guidance for the LLM
# ---------------------------------------------------------------------------

style_instructions: dict[str, str] = {
    "ED event": (
        "Write in a concise, rapid-documentation style typical of busy emergency departments. "
        "Use NEWS2 scoring prominently. Triage category must be documented. "
        "Time-stamp all observations. Abbreviated clinical language is acceptable but must "
        "be unambiguous. Use bullet points for investigations and management."
    ),
    "ED review and hand-over": (
        "Write as a structured handover using SBAR format (Situation, Background, Assessment, "
        "Recommendation). Include time-critical information prominently. "
        "Clearly state outstanding decisions and who is responsible."
    ),
    "post take ward round": (
        "Write in formal consultant-level clinical prose. Structure clearly with PMH, "
        "examination, investigations, impression, and plan. "
        "Include problem-based management plan with clear actions. "
        "Document VTE risk assessment and any capacity/consent issues."
    ),
    "general ward round": (
        "Write in the style of daily ward round notes: concise but complete. "
        "Focus on progress, results, and updated plan. "
        "Document discharge planning every day. "
        "Note any deterioration with clear escalation plan."
    ),
    "inter-specialty review": (
        "Write as a formal specialty opinion. Clearly state the clinical question from the "
        "referring team and provide a structured response. "
        "Recommendations should be specific and actionable. "
        "Avoid clinical overlap with referring team's role."
    ),
    "operation": (
        "Write as a formal operation note following the ASGBI standards. "
        "Include all mandatory fields: date/time, team, procedure, anaesthetic, findings, "
        "specimens, complications, and post-op plan. "
        "Use anatomical terminology correctly. "
        "Document implant details with batch numbers."
    ),
    "pre-op assessment": (
        "Write as a structured pre-operative assessment note. "
        "Document ASA grade with justification. "
        "Include full airway assessment. "
        "Address all co-morbidities and their peri-operative implications. "
        "Document fasting and medication instructions clearly."
    ),
    "pre-op consent": (
        "Write as a formal consent note per GMC Good Medical Practice guidance. "
        "Document benefits, risks (common and serious), and alternatives discussed. "
        "Confirm patient's understanding and questions answered. "
        "Capacity assessment documented."
    ),
    "pre-op checklist": (
        "Write as a structured WHO Surgical Safety Checklist. "
        "All items should be explicitly checked and documented. "
        "Antibiotic prophylaxis time documented. "
        "Laterality/site confirmation documented."
    ),
    "anaesthetics assessment": (
        "Write as a formal anaesthetic pre-operative assessment. "
        "ASA grade with detailed justification. "
        "Mallampati and airway assessment prominently documented. "
        "Specific anaesthetic plan with backup plan documented. "
        "PONV risk and prophylaxis plan included."
    ),
    "post-anaesthesia recovery": (
        "Write as structured PACU nursing/anaesthetic notes. "
        "Modified Aldrete scores at arrival and discharge. "
        "Pain and PONV scores at regular intervals. "
        "Escalation threshold clearly stated."
    ),
    "nursing": (
        "Write as clinical nursing notes using NHS nursing documentation conventions. "
        "Use SOAPIE or equivalent structured format. "
        "Document care delivered against care plan. "
        "Escalations and communications documented with name, time, and outcome."
    ),
    "therapy": (
        "Write as professional AHP notes with SMART goals. "
        "Use standardised assessment tools (Barthel, MUST, Waterlow, etc.). "
        "Equipment provision and home exercise programmes documented. "
        "Discharge recommendations must be explicit."
    ),
    "emergency_admission": (
        "Write as a thorough emergency clerking note. "
        "Full systematic history with SOCRATES for pain. "
        "Complete examination with all systems. "
        "Investigations ordered and available results documented. "
        "Clear immediate management plan with escalation criteria."
    ),
    "elective_admission": (
        "Write as a structured elective admission note. "
        "Pre-operative assessment summary. "
        "Consent status confirmed. "
        "Clear peri-operative plan with enhanced recovery pathway."
    ),
    "misc": (
        "Write as a brief, professional clinical note. "
        "Include date, time, clinician details, and clear clinical content. "
        "Ensure note is factual, objective, and relevant."
    ),
}

# Default style instructions for note types not listed above
default_style_instructions: str = (
    "Write in clear, professional UK clinical language following NHS documentation "
    "standards. Use SI units throughout. Clinical content must be accurate and "
    "appropriate for the speciality and event type. "
    "Avoid jargon or abbreviations that could cause ambiguity."
)
