"""
Static configuration for synthetic clinical note generation.

Contains: event types, clinician roles, sections to skip during augmentation,
style instructions, and allergy prevalence data.

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
# doctor_roles
# Consultant and junior doctor roles by specialty
# ---------------------------------------------------------------------------

doctor_roles: dict[str, list[str]] = {
    "General Medicine": [
        "Consultant Physician",
        "SpR General Medicine",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
        "Foundation Year 1 Doctor",
        "Acute Medicine Registrar",
        "Specialty Doctor",
    ],
    "Cardiology": [
        "Consultant Cardiologist",
        "Cardiology Registrar",
        "Cardiology SpR",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
        "Interventional Cardiology Fellow",
    ],
    "Respiratory Medicine": [
        "Consultant Respiratory Physician",
        "Respiratory Registrar",
        "Respiratory SpR",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
    ],
    "General Surgery": [
        "Consultant General Surgeon",
        "Surgical Registrar",
        "Surgical SpR",
        "Core Surgical Trainee (CST)",
        "Foundation Year 2 Doctor",
        "Foundation Year 1 Doctor",
    ],
    "Orthopaedic Surgery": [
        "Consultant Orthopaedic Surgeon",
        "Orthopaedic Registrar",
        "Orthopaedic SpR",
        "Core Surgical Trainee (CST)",
        "Foundation Year 2 Doctor",
    ],
    "Trauma and Orthopaedics": [
        "Consultant Trauma and Orthopaedic Surgeon",
        "T&O Registrar",
        "T&O SpR",
        "Core Surgical Trainee (CST)",
        "Foundation Year 2 Doctor",
    ],
    "Gastroenterology": [
        "Consultant Gastroenterologist",
        "Gastroenterology Registrar",
        "Gastroenterology SpR",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
    ],
    "Neurology": [
        "Consultant Neurologist",
        "Neurology Registrar",
        "Neurology SpR",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
    ],
    "Urology": [
        "Consultant Urological Surgeon",
        "Urology Registrar",
        "Urology SpR",
        "Core Surgical Trainee (CST)",
        "Foundation Year 2 Doctor",
    ],
    "Nephrology": [
        "Consultant Nephrologist",
        "Nephrology Registrar",
        "Nephrology SpR",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
    ],
    "Endocrinology": [
        "Consultant Endocrinologist",
        "Endocrinology Registrar",
        "Endocrinology SpR",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
    ],
    "Oncology": [
        "Consultant Medical Oncologist",
        "Oncology Registrar",
        "Oncology SpR",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
    ],
    "Stroke Medicine": [
        "Consultant Stroke Physician",
        "Stroke Registrar",
        "Stroke SpR",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
    ],
    "Psychiatry": [
        "Consultant Psychiatrist",
        "Psychiatry Registrar (ST4-6)",
        "Psychiatry Core Trainee (CT1-3)",
        "Foundation Year 2 Doctor",
        "Specialty Doctor",
    ],
    "Emergency Medicine": [
        "Consultant in Emergency Medicine",
        "Emergency Medicine Registrar",
        "Emergency Medicine SpR",
        "Emergency Medicine CT1-2",
        "Foundation Year 2 Doctor",
    ],
    "Anaesthetics": [
        "Consultant Anaesthetist",
        "Anaesthetics Registrar",
        "Anaesthetics SpR",
        "Core Anaesthetics Trainee (CT1-2)",
    ],
    "Intensive Care": [
        "Consultant in Intensive Care Medicine",
        "ICM Registrar",
        "ICM SpR",
        "Foundation Year 2 Doctor",
    ],
    "Geriatric Medicine": [
        "Consultant Geriatrician",
        "Geriatrics Registrar",
        "Geriatrics SpR",
        "Core Medical Trainee (CMT)",
        "Foundation Year 2 Doctor",
    ],
    "Colorectal Surgery": [
        "Consultant Colorectal Surgeon",
        "Colorectal Registrar",
        "Colorectal SpR",
        "Core Surgical Trainee (CST)",
        "Foundation Year 2 Doctor",
    ],
    "Vascular Surgery": [
        "Consultant Vascular Surgeon",
        "Vascular Registrar",
        "Vascular SpR",
        "Core Surgical Trainee (CST)",
        "Foundation Year 2 Doctor",
    ],
    "Haematology": [
        "Consultant Haematologist",
        "Haematology Registrar",
        "Haematology SpR",
        "Core Medical Trainee (CMT)",
    ],
    "Rheumatology": [
        "Consultant Rheumatologist",
        "Rheumatology Registrar",
        "Rheumatology SpR",
        "Core Medical Trainee (CMT)",
    ],
    "Infectious Diseases": [
        "Consultant in Infectious Diseases",
        "Infectious Diseases Registrar",
        "Infectious Diseases SpR",
        "Core Medical Trainee (CMT)",
    ],
}

# Flat list of all possible doctor roles for convenience
all_doctor_roles: list[str] = sorted(
    {role for roles in doctor_roles.values() for role in roles}
)

# ---------------------------------------------------------------------------
# therapist_roles
# Allied health professional and nursing roles
# ---------------------------------------------------------------------------

therapist_roles: list[str] = [
    # Nursing
    "Staff Nurse",
    "Senior Staff Nurse",
    "Charge Nurse / Ward Sister",
    "Clinical Nurse Specialist",
    "Advanced Nurse Practitioner",
    "Nurse Consultant",
    # Physiotherapy
    "Band 5 Physiotherapist",
    "Band 6 Physiotherapist",
    "Band 7 Physiotherapist",
    "Senior Physiotherapist",
    "Physiotherapy Clinical Specialist",
    # Occupational Therapy
    "Band 5 Occupational Therapist",
    "Band 6 Occupational Therapist",
    "Band 7 Occupational Therapist",
    "Senior Occupational Therapist",
    # Speech and Language Therapy
    "Speech and Language Therapist",
    "Senior Speech and Language Therapist",
    # Dietetics
    "Dietitian",
    "Senior Dietitian",
    "Specialist Dietitian",
    # Pharmacy
    "Clinical Pharmacist",
    "Medicines Management Pharmacist",
    # Social Work
    "Social Worker",
    "Senior Social Worker",
    # Other AHPs
    "Radiographer",
    "Diagnostic Radiographer",
    "Operating Department Practitioner (ODP)",
    "Healthcare Assistant",
]

# ---------------------------------------------------------------------------
# sections_to_ignore_typos
# Note sections where typos should NOT be introduced (structured/coded data)
# ---------------------------------------------------------------------------

sections_to_ignore_typos: list[str] = [
    "patient_demographics",
    "allergies",
    "medications_on_admission",
    "current_medications",
    "vital_signs_and_observations",
    "investigations",
    "investigations_reviewed",
    "procedure_details",
    "surgical_team",
    "anaesthetic_details",
    "pre-op checklist",
    "patient_identification",
    "consent_and_procedure",
    "fasting_status",
    "theatre_safety_items",
    "asa_grade_and_risk",
    "asa_grade_and_risk_assessment",
]

# ---------------------------------------------------------------------------
# sections_to_ignore_abbreviations
# Note sections that should remain in full text (not abbreviated)
# ---------------------------------------------------------------------------

sections_to_ignore_abbreviations: list[str] = [
    "patient_demographics",
    "consent_decision",
    "patient_questions_and_understanding",
    "handover_information",
    "discharge_from_recovery",
    "pre-op checklist",
    "patient_identification",
    "consent_and_procedure",
    "theatre_safety_items",
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

# ---------------------------------------------------------------------------
# allergy_prevalence
# Approximate UK prevalence rates for common drug allergies
# Used when generating random patient allergy profiles
# ---------------------------------------------------------------------------

allergy_prevalence: dict[str, float] = {
    "NKDA": 0.75,                           # 75% of patients have no known drug allergies
    "Penicillin - rash": 0.05,
    "Penicillin - anaphylaxis": 0.01,
    "Amoxicillin - rash": 0.03,
    "Co-amoxiclav - rash": 0.02,
    "Aspirin - bronchospasm": 0.02,
    "NSAIDs - gastrointestinal intolerance": 0.03,
    "NSAIDs - bronchospasm": 0.01,
    "Codeine - nausea and vomiting": 0.02,
    "Morphine - nausea and vomiting": 0.02,
    "Tramadol - hallucinations": 0.005,
    "Metronidazole - nausea": 0.01,
    "Sulfonamides - rash": 0.01,
    "Erythromycin - gastrointestinal intolerance": 0.02,
    "Clarithromycin - gastrointestinal intolerance": 0.01,
    "Trimethoprim - rash": 0.005,
    "Ciprofloxacin - tendinopathy": 0.005,
    "ACE inhibitors - angio-oedema": 0.01,
    "ACE inhibitors - cough": 0.05,
    "Statins - myalgia": 0.03,
    "Metformin - gastrointestinal intolerance": 0.02,
    "Latex - contact urticaria": 0.01,
    "Iodinated contrast - anaphylactoid": 0.005,
    "Chlorhexidine - anaphylaxis": 0.002,
    "Ibuprofen - rash": 0.01,
    "Carbamazepine - Stevens-Johnson syndrome": 0.001,
    "Vancomycin - red man syndrome": 0.01,
    "Gentamicin - nephrotoxicity": 0.005,
}

# ---------------------------------------------------------------------------
# ward_names
# Fictitious but realistic NHS ward names by specialty
# ---------------------------------------------------------------------------

ward_names: dict[str, list[str]] = {
    "Cardiology": [
        "Coronary Care Unit (CCU)",
        "Cardiac Ward (Ward 7)",
        "Cardiology Ward (Ward 9)",
    ],
    "Respiratory Medicine": [
        "Respiratory Ward (Ward 4)",
        "Chest Unit (Ward 12)",
    ],
    "General Medicine": [
        "Medical Assessment Unit (MAU)",
        "Acute Medical Unit (AMU)",
        "General Medical Ward (Ward 6)",
        "General Medical Ward (Ward 8)",
    ],
    "General Surgery": [
        "Surgical Admissions Unit (SAU)",
        "General Surgical Ward (Ward 3)",
        "Day Surgery Unit",
    ],
    "Orthopaedic Surgery": [
        "Trauma Ward (Ward 11)",
        "Elective Orthopaedics Ward (Ward 14)",
        "Orthopaedic Day Unit",
    ],
    "Trauma and Orthopaedics": [
        "Trauma Ward (Ward 11)",
        "Elective Orthopaedics Ward (Ward 14)",
    ],
    "Stroke Medicine": [
        "Stroke Unit (Ward 5)",
        "Hyper-Acute Stroke Unit (HASU)",
    ],
    "Intensive Care": [
        "Intensive Care Unit (ICU)",
        "High Dependency Unit (HDU)",
    ],
    "Neurology": [
        "Neurology Ward (Ward 10)",
        "Neurorehabilitation Unit",
    ],
    "Psychiatry": [
        "Acute Psychiatric Ward",
        "Psychiatric Intensive Care Unit (PICU)",
    ],
    "Emergency Medicine": [
        "Emergency Department",
        "Clinical Decision Unit (CDU)",
        "Emergency Assessment Area",
    ],
}

default_ward: str = "General Medical Ward (Ward 6)"

# ---------------------------------------------------------------------------
# consultant_names
# Fictitious but realistic NHS consultant names (for use in note generation)
# ---------------------------------------------------------------------------

consultant_names: list[str] = [
    "Dr Sarah Mitchell",
    "Dr James Patel",
    "Dr Fiona Thompson",
    "Dr Andrew Nguyen",
    "Dr Rachel Davies",
    "Dr Michael Okonkwo",
    "Dr Claire Robertson",
    "Dr David Hassan",
    "Dr Priya Sharma",
    "Dr Thomas Williams",
    "Mr Jonathan Clarke",      # Surgical consultants use 'Mr/Mrs/Ms'
    "Mr Aisha Osei",
    "Mr Robert Griffiths",
    "Mrs Nadia Fernandez",
    "Mr Christopher Blake",
    "Ms Helen Watkins",
    "Prof Elizabeth Morgan",   # Academic consultants
    "Prof Rajan Krishnamurthy",
]
