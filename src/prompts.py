"""
Prompt templates for synthetic clinical note generation.

All prompts use Python's string.Template with $VARIABLE substitution.
Categories:
  - patient_prompts: patient and admission generation
  - journey_prompts: patient journey / event sequence generation
  - note_prompts: clinical note generation
  - processing_prompts: post-processing (cleaning, abbreviations)
  - evaluation_prompts: quality evaluation metrics

Adapted from nhsengland/synthetic_clinical_notes with additions for
ICD-10 and OPCS-4 code-driven generation.
"""

from string import Template

# ---------------------------------------------------------------------------
# patient_prompts
# Generate patient demographics and admission summaries
# ---------------------------------------------------------------------------

patient_prompts: dict[str, Template] = {

    "generate_patient_prompt": Template(
        "You are generating a realistic synthetic NHS patient record for use in "
        "clinical note generation training data. The patient must be entirely fictitious "
        "with no resemblance to any real person.\n\n"
        "Generate a patient with the following details as a JSON object:\n"
        "- full_name: realistic UK name (first name and surname)\n"
        "- title: Mr/Mrs/Miss/Ms/Dr/Prof\n"
        "- first_name: first name\n"
        "- surname: surname\n"
        "- date_of_birth: ISO format (YYYY-MM-DD), age should be between 18 and 95\n"
        "- age: integer\n"
        "- sex: Male or Female\n"
        "- gender_identity: same as sex unless specified otherwise\n"
        "- nhs_number: 10-digit NHS number (format: XXX XXX XXXX)\n"
        "- mrn: 7-digit hospital MRN\n"
        "- address_line_1: UK street address\n"
        "- address_line_2: optional second line or empty string\n"
        "- address_city: UK city or town\n"
        "- postcode: valid UK postcode format\n"
        "- phone_number: UK mobile or landline\n"
        "- registered_gp: GP practice name and address\n"
        "- next_of_kin_name: full name with relationship in brackets\n"
        "- next_of_kin_number: UK phone number\n"
        "- allergies: list of drug allergies as strings (e.g. 'Penicillin - anaphylaxis'), "
        "or ['NKDA']\n"
        "- past_medical_history: list of 2-5 relevant past medical conditions\n"
        "- medications: list of current medications (name, dose, frequency)\n"
        "- smoking_status: Never smoker / Ex-smoker (X pack years) / Current smoker (X/day)\n"
        "- alcohol_units_per_week: integer\n"
        "- occupation: current or most recent occupation\n"
        "- living_situation: lives alone / with spouse / with family / care home\n\n"
        "Return ONLY valid JSON with these exact keys. No explanatory text."
    ),

    "emergency_admission_prompt": Template(
        "You are an NHS clinician generating a realistic synthetic emergency admission "
        "summary for clinical note training data.\n\n"
        "PATIENT DETAILS:\n$PATIENT_DETAILS\n\n"
        "DIAGNOSIS: $DIAGNOSIS\n"
        "CHIEF COMPLAINT: $CHIEF_COMPLAINT\n"
        "ADMITTING CONSULTANT: $CONSULTANT\n"
        "SPECIALTY: $SPECIALTY\n"
        "ADMISSION DATE: $ADMISSION_DATE\n"
        "ADMISSION TIME: $ADMISSION_TIME\n\n"
        "Generate a realistic emergency admission clerking note as a JSON object with "
        "these keys:\n"
        "- admission_type: 'emergency'\n"
        "- admission_method: e.g. 'Emergency department' / 'GP referral' / '999 ambulance'\n"
        "- chief_complaint: presenting complaint\n"
        "- history_of_presenting_complaint: detailed HPC in clinical prose\n"
        "- past_medical_history: relevant PMH\n"
        "- medications: current medications\n"
        "- allergies: known drug allergies\n"
        "- systems_review: brief systematic review\n"
        "- examination_findings: detailed examination including vitals and NEWS2 score\n"
        "- investigations: initial investigations ordered and any available results\n"
        "- working_diagnosis: primary diagnosis and differentials\n"
        "- management_plan: immediate management plan\n"
        "- admitting_consultant: $CONSULTANT\n"
        "- specialty: $SPECIALTY\n"
        "- ward: appropriate ward for specialty\n"
        "- estimated_los_days: estimated length of stay as integer\n\n"
        "Use realistic UK clinical terminology and NHS documentation conventions. "
        "Measurements in SI units (mmHg, mmol/L, g/dL). "
        "Return ONLY valid JSON."
    ),

    "elective_admission_prompt": Template(
        "You are an NHS clinician generating a realistic synthetic elective admission "
        "summary for clinical note training data.\n\n"
        "PATIENT DETAILS:\n$PATIENT_DETAILS\n\n"
        "PLANNED PROCEDURE: $PROCEDURE\n"
        "ADMITTING CONSULTANT: $CONSULTANT\n"
        "SPECIALTY: $SPECIALTY\n"
        "ADMISSION DATE: $ADMISSION_DATE\n"
        "ADMISSION TIME: $ADMISSION_TIME\n\n"
        "Generate a realistic elective admission clerking note as a JSON object with "
        "these keys:\n"
        "- admission_type: 'elective'\n"
        "- admission_method: 'Elective admission'\n"
        "- planned_procedure: full procedure name\n"
        "- indication: clinical indication for the procedure\n"
        "- past_medical_history: relevant PMH and surgical history\n"
        "- medications: current medications including peri-operative adjustments\n"
        "- allergies: known drug allergies\n"
        "- examination_findings: pre-operative examination including vitals\n"
        "- pre_op_investigations: pre-operative investigations reviewed\n"
        "- asa_grade: ASA classification (I-IV) with brief justification\n"
        "- consent_status: obtained/pending\n"
        "- management_plan: admission and peri-operative plan\n"
        "- admitting_consultant: $CONSULTANT\n"
        "- specialty: $SPECIALTY\n"
        "- ward: appropriate ward for specialty\n"
        "- estimated_los_days: estimated length of stay as integer\n\n"
        "Use realistic UK clinical terminology and NHS documentation conventions. "
        "Return ONLY valid JSON."
    ),

    "length_of_stay_prompt": Template(
        "You are an NHS clinical informatics specialist. Based on the following admission "
        "details, estimate a realistic length of stay in days for this patient.\n\n"
        "ADMISSION DETAILS:\n$ADMISSION_DETAILS\n\n"
        "Consider: diagnosis severity, co-morbidities, age, social circumstances, "
        "and typical NHS LOS benchmarks for this condition.\n\n"
        "Return ONLY a JSON object with:\n"
        "- estimated_los_days: integer (estimated length of stay)\n"
        "- los_rationale: brief explanation (1-2 sentences)\n"
        "- discharge_barriers: list of potential factors that could prolong LOS\n\n"
        "Return ONLY valid JSON."
    ),

    "code_driven_admission_prompt": Template(
        "You are an experienced NHS consultant generating a realistic synthetic patient "
        "admission summary driven by structured clinical codes. Codes may come from any "
        "diagnostic or procedure coding standard (e.g. ICD-10, ICD-11, SNOMED CT, OPCS-4, "
        "CPT) - treat the system name given below as authoritative for how to interpret "
        "the codes. This will be used as training data for clinical note generation models.\n\n"
        "PATIENT DETAILS:\n$PATIENT_DETAILS\n\n"
        "DIAGNOSES ($DIAGNOSTIC_CODE_SYSTEM):\n$DIAGNOSES_CONTEXT\n\n"
        "PROCEDURES ($PROCEDURE_CODE_SYSTEM):\n$PROCEDURES_CONTEXT\n\n"
        "ADMISSION DATE: $ADMISSION_DATE\n"
        "ADMISSION TIME: $ADMISSION_TIME\n\n"
        "Generate a clinically coherent NHS admission record that incorporates all of "
        "the above diagnoses and procedures (ignore any section marked 'None specified'). "
        "Ensure:\n"
        "1. A logical clinical narrative that links the diagnoses and procedures\n"
        "2. Appropriate priority ordering (primary diagnosis/procedure first)\n"
        "3. Chief complaint, history, investigations, and medications entirely consistent "
        "with the diagnoses and procedures above\n"
        "4. Management plan that addresses each diagnosis and planned procedure\n"
        "5. Realistic multidisciplinary team involvement\n"
        "6. NHS documentation standards throughout\n\n"
        "Return a JSON object with keys:\n"
        "- admission_type: 'emergency' or 'elective'\n"
        "- primary_diagnosis: most clinically significant diagnosis (empty string if none given)\n"
        "- secondary_diagnoses: list of other diagnoses\n"
        "- primary_procedure: primary planned procedure if applicable (empty string if none given)\n"
        "- secondary_procedures: list of additional procedures if applicable\n"
        "- chief_complaint: presenting complaint\n"
        "- history_of_presenting_complaint: detailed history\n"
        "- past_medical_history: relevant PMH\n"
        "- medications: full medication list\n"
        "- allergies: drug allergies\n"
        "- examination_findings: full clinical examination\n"
        "- investigations: relevant investigations with results\n"
        "- working_diagnosis: confirmed diagnosis/diagnoses and differentials\n"
        "- management_plan: comprehensive management plan\n"
        "- specialty: primary specialty\n"
        "- ward: appropriate ward\n"
        "- estimated_los_days: realistic LOS integer\n\n"
        "Return ONLY valid JSON."
    ),
}

# ---------------------------------------------------------------------------
# journey_prompts
# Generate the sequence of events in a patient's hospital journey
# ---------------------------------------------------------------------------

journey_prompts: dict[str, Template] = {

    "simple_patient_journey_prompt": Template(
        "You are an NHS clinical informatics specialist generating a realistic sequence "
        "of clinical events for a synthetic patient's hospital journey.\n\n"
        "PATIENT DETAILS:\n$PATIENT_DETAILS\n\n"
        "ADMISSION DETAILS:\n$ADMISSION_DETAILS\n\n"
        "ADMISSION DATE: $ADMISSION_DATE\n"
        "DISCHARGE DATE: $DISCHARGE_DATE\n\n"
        "POSSIBLE EVENT TYPES:\n$POSSIBLE_EVENT_TYPES\n\n"
        "Generate a realistic, chronologically ordered sequence of clinical events for "
        "this patient's hospital stay. Each event should be appropriate to the patient's "
        "diagnosis, admission type, and clinical trajectory.\n\n"
        "Guidelines:\n"
        "- For emergency admissions: start with ED event(s), then post take ward round, "
        "then daily ward rounds\n"
        "- For elective surgical admissions: include pre-op assessment, consent, operation, "
        "recovery, and post-op ward rounds\n"
        "- Include nursing notes at least once daily\n"
        "- Include therapy reviews if LOS > 3 days\n"
        "- Include inter-specialty reviews if relevant to diagnosis\n"
        "- Events should show a logical clinical trajectory (admission -> treatment -> "
        "improvement -> discharge)\n\n"
        "Return a JSON array of event objects. Each event object must have:\n"
        "- event_type: one of the possible event types listed above\n"
        "- event_date: ISO date (YYYY-MM-DD) within admission-discharge range\n"
        "- event_time: HH:MM (24-hour)\n"
        "- event_order: integer starting from 1\n"
        "- brief_description: one sentence describing what happens at this event\n"
        "- clinician_type: type of clinician performing this event\n\n"
        "Return ONLY a valid JSON array."
    ),

    "continue_journey_prompt": Template(
        "You are continuing to generate a realistic NHS patient journey. "
        "The journey so far is shown below, along with the remaining events that "
        "need detailed descriptions.\n\n"
        "JOURNEY SO FAR:\n$JOURNEY_SO_FAR\n\n"
        "REMAINING EVENTS TO DETAIL:\n$REMAINING_EVENTS\n\n"
        "For each remaining event, provide a more detailed brief_description and "
        "any additional context that will help generate the clinical note. "
        "Ensure clinical consistency and logical progression from the previous events.\n\n"
        "Return ONLY a valid JSON array of the remaining events with the same structure "
        "as the journey so far, with updated brief_description fields."
    ),

    "generate_event_details_prompt": Template(
        "You are an NHS clinician generating detailed event information for a specific "
        "point in a patient's hospital journey. This will be used to generate a "
        "realistic clinical note.\n\n"
        "PATIENT DETAILS:\n$PATIENT_DETAILS\n\n"
        "ADMISSION DETAILS:\n$ADMISSION_DETAILS\n\n"
        "CURRENT EVENT TYPE: $EVENT_TYPE\n"
        "EVENT DATE: $EVENT_DATE\n\n"
        "JOURNEY SO FAR (for clinical context):\n$JOURNEY_SO_FAR\n\n"
        "AVAILABLE DOCTOR ROLES:\n$DOCTOR_ROLES\n\n"
        "AVAILABLE THERAPIST ROLES:\n$THERAPIST_ROLES\n\n"
        "STAFF NAMES TO USE:\n$STAFF_NAMES\n\n"
        "Generate detailed event information as a JSON object with:\n"
        "- event_type: '$EVENT_TYPE'\n"
        "- event_date: '$EVENT_DATE'\n"
        "- event_time: HH:MM (24-hour format, realistic for this event type)\n"
        "- clinician_name: choose from staff names provided\n"
        "- clinician_role: appropriate role from doctor/therapist roles\n"
        "- clinical_status: patient's clinical status at this point (stable/improving/"
        "deteriorating/critical)\n"
        "- key_findings: list of 3-5 key clinical findings relevant to this event\n"
        "- key_actions: list of 3-5 key actions taken or planned\n"
        "- vital_signs: dict of current vital signs (HR, BP, RR, SpO2, Temp)\n"
        "- news2_score: integer NEWS2 score\n"
        "- brief_clinical_summary: 2-3 sentence summary of this event\n\n"
        "Ensure clinical coherence with the journey so far. "
        "Return ONLY valid JSON."
    ),

    "validate_simple_journey_prompt": Template(
        "You are an NHS clinical quality reviewer. Review the following synthetic patient "
        "journey for clinical accuracy, consistency, and realism.\n\n"
        "PATIENT JOURNEY:\n$JOURNEY\n\n"
        "ADMISSION DETAILS:\n$ADMISSION_DETAILS\n\n"
        "PATIENT DETAILS:\n$PATIENT_DETAILS\n\n"
        "Assess the journey for:\n"
        "1. Clinical accuracy: Are events appropriate for the diagnosis?\n"
        "2. Chronological consistency: Do events follow a logical timeline?\n"
        "3. Clinical trajectory: Is the patient pathway realistic (admission to discharge)?\n"
        "4. NHS appropriateness: Are NHS processes followed correctly?\n"
        "5. Completeness: Are there missing events that should be present?\n\n"
        "Return a JSON object with:\n"
        "- is_valid: boolean\n"
        "- overall_quality: integer 1-10\n"
        "- issues_found: list of specific issues (empty list if none)\n"
        "- suggested_corrections: list of corrections to apply (empty list if none)\n"
        "- clinical_accuracy_score: integer 1-10\n"
        "- nhs_appropriateness_score: integer 1-10\n\n"
        "Return ONLY valid JSON."
    ),
}

# ---------------------------------------------------------------------------
# note_prompts
# Generate individual clinical notes for specific events
# ---------------------------------------------------------------------------

note_prompts: dict[str, Template] = {

    "clinical_note_prompt": Template(
        "You are an experienced NHS clinician writing a realistic clinical note for "
        "synthetic training data. The note must be clinically accurate, use appropriate "
        "UK medical terminology, and follow NHS documentation standards.\n\n"
        "PATIENT DETAILS:\n$PATIENT_DETAILS\n\n"
        "ADMISSION DETAILS (diagnosis/procedure driving this admission):\n"
        "$ADMISSION_DETAILS\n\n"
        "EVENT DETAILS:\n$EVENT_DETAILS\n\n"
        "PREVIOUS EVENTS IN JOURNEY (for context):\n$PREVIOUS_EVENTS\n\n"
        "NOTE TEMPLATE (sections to include):\n$NOTE_TEMPLATE\n\n"
        "STYLE INSTRUCTIONS:\n$STYLE_INSTRUCTIONS\n\n"
        "INCLUDE DETAILED EXAMINATION: $ADD_EXAMINATION\n\n"
        "RED FLAGS TO CONSIDER: $RED_FLAGS\n\n"
        "Write a complete, realistic clinical note for this event. Requirements:\n"
        "1. Write in clinical prose appropriate to the note type\n"
        "2. Include all sections from the note template\n"
        "3. Ensure clinical consistency with previous events in the journey\n"
        "4. Use UK units: mmHg, mmol/L, g/dL, degrees Celsius\n"
        "5. Reference the patient by name and use their specific details\n"
        "6. Include realistic vital signs appropriate to the clinical state\n"
        "7. Use standard UK drug names (not brand names unless specifically required)\n"
        "8. Apply style instructions for clinical authenticity\n"
        "9. If red flags apply, ensure they are addressed in the note\n"
        "10. Ensure clinical content is consistent with the admission diagnosis/procedure above\n\n"
        "Format: structured clinical note with clear section headers matching the template. "
        "Do NOT include any patient identifiers beyond those already in patient details. "
        "Write the note text only - no JSON wrapper."
    ),

    "simple_clinical_note_prompt": Template(
        "You are an NHS clinician. Write a concise, realistic clinical note for "
        "the following patient event.\n\n"
        "PATIENT: $PATIENT_DETAILS\n\n"
        "EVENT: $EVENT_DETAILS\n\n"
        "Write a realistic clinical note of 150-400 words. "
        "Include: date/time, clinician role, chief complaint or reason for note, "
        "key findings, and management plan. "
        "Use UK clinical conventions and terminology. "
        "Write the note text only."
    ),

    "validate_responses_prompt": Template(
        "You are an NHS clinical quality assurance reviewer evaluating a synthetic "
        "clinical note for accuracy and quality.\n\n"
        "CLINICAL NOTE TO EVALUATE:\n$NOTE\n\n"
        "REFERENCE MATERIAL (admission details, events, patient context):\n$REFERENCE_MATERIAL\n\n"
        "Evaluate the note against these criteria:\n"
        "1. Clinical accuracy: Are findings, diagnoses, and treatments clinically appropriate?\n"
        "2. Factual consistency: Is the note consistent with the reference material?\n"
        "3. Completeness: Does the note cover the required clinical content?\n"
        "4. Professional language: Is the language appropriate for NHS clinical documentation?\n"
        "5. NHS standards: Does it follow UK clinical documentation conventions?\n\n"
        "Return a JSON object with:\n"
        "- is_acceptable: boolean (true if note is clinically usable)\n"
        "- quality_score: integer 1-10\n"
        "- clinical_accuracy: integer 1-10\n"
        "- consistency_with_context: integer 1-10\n"
        "- issues: list of specific issues found (empty if none)\n"
        "- suggested_improvements: list of improvements (empty if none)\n\n"
        "Return ONLY valid JSON."
    ),

    "determine_examination_prompt": Template(
        "You are an NHS consultant. Based on the event type and patient details below, "
        "determine whether a full, focused, or no clinical examination should be documented "
        "in this clinical note.\n\n"
        "EVENT TYPE: $EVENT_TYPE\n\n"
        "PATIENT DETAILS:\n$PATIENT_DETAILS\n\n"
        "Return a JSON object with:\n"
        "- include_examination: boolean\n"
        "- examination_type: 'full_systems' / 'focused' / 'targeted' / 'none'\n"
        "- systems_to_examine: list of body systems to include (empty if none)\n"
        "- rationale: brief explanation\n\n"
        "Return ONLY valid JSON."
    ),

    "red_flags_prompt": Template(
        "You are an NHS consultant. Based on this patient's admission details, "
        "identify any clinical red flags or safety-critical considerations that "
        "should be documented in their clinical notes.\n\n"
        "ADMISSION DETAILS:\n$ADMISSION_DETAILS\n\n"
        "PATIENT DETAILS:\n$PATIENT_DETAILS\n\n"
        "Return a JSON object with:\n"
        "- has_red_flags: boolean\n"
        "- red_flags: list of specific red flag conditions or risks (e.g. "
        "'High-risk airway', 'Anticoagulated patient', 'Recent MI - STEMI protocol', "
        "'Neutropenic sepsis', 'Safeguarding concern', 'Capacity concerns')\n"
        "- safety_alerts: list of safety alerts to include in notes\n"
        "- escalation_criteria: list of clinical criteria requiring urgent escalation\n\n"
        "Return ONLY valid JSON."
    ),
}

# ---------------------------------------------------------------------------
# processing_prompts
# Post-processing utilities: cleaning and abbreviation augmentation
# ---------------------------------------------------------------------------

processing_prompts: dict[str, Template] = {

    "clean_outputs_prompt": Template(
        "You are a clinical text processing specialist. Clean the following clinical "
        "text according to the specified cleaning type.\n\n"
        "CLEANING TYPE: $CLEANING_TYPE\n\n"
        "TEXT TO CLEAN:\n$VALUE\n\n"
        "Cleaning instructions by type:\n"
        "- 'remove_identifiers': Remove or replace any potential real patient identifiers "
        "(real NHS numbers, real phone numbers, real addresses) with fictional equivalents. "
        "Keep the clinical content intact.\n"
        "- 'standardise_units': Convert all units to SI units used in UK clinical practice "
        "(mmHg, mmol/L, g/dL, kg, cm, Celsius).\n"
        "- 'fix_formatting': Fix formatting issues: consistent section headers, "
        "proper punctuation, remove duplicate whitespace.\n"
        "- 'clinical_language': Ensure clinical language is appropriate for NHS documentation. "
        "Replace lay terms with clinical terms where appropriate.\n"
        "- 'remove_hallucinations': Remove factually incorrect clinical statements, "
        "fictional drug names, or impossible clinical values.\n\n"
        "Return the cleaned text only. No explanatory comments."
    ),

    "add_abbreviations_prompt": Template(
        "You are a clinical text specialist. Rewrite the following clinical note to "
        "naturally incorporate common NHS clinical abbreviations as a clinician would "
        "use them when writing quickly.\n\n"
        "ORIGINAL TEXT:\n$TEXT\n\n"
        "Guidelines for abbreviation:\n"
        "- Use abbreviations naturally, not uniformly (mix abbreviated and full forms)\n"
        "- Common acceptable abbreviations: SOB (shortness of breath), "
        "CP (chest pain), Hx (history), Dx (diagnosis), Rx (treatment/prescription), "
        "OE (on examination), O/E (on examination), Ix (investigations), "
        "Mx (management), NAD (no abnormality detected), HS (heart sounds), "
        "BS (breath sounds), HS1+2+0 (normal heart sounds), "
        "AVPU (Alert, Voice, Pain, Unresponsive), "
        "JVP (jugular venous pressure), PR (per rectum), "
        "PO (per os/by mouth), IV (intravenous), SC (subcutaneous), "
        "IM (intramuscular), BD (twice daily), TDS (three times daily), "
        "QDS (four times daily), OD (once daily), PRN (as required), "
        "STAT (immediately), NBM (nil by mouth), "
        "FBC (full blood count), U&E (urea and electrolytes), "
        "LFTs (liver function tests), TFTs (thyroid function tests), "
        "CRP (C-reactive protein), ECG (electrocardiogram), "
        "CXR (chest X-ray), USS (ultrasound scan), CT (computed tomography), "
        "MRI (magnetic resonance imaging), "
        "HTN (hypertension), DM (diabetes mellitus), "
        "IHD (ischaemic heart disease), CCF (congestive cardiac failure), "
        "COPD (chronic obstructive pulmonary disease), "
        "AF (atrial fibrillation), PE (pulmonary embolism), "
        "DVT (deep vein thrombosis), MI (myocardial infarction), "
        "CVA (cerebrovascular accident/stroke), TIA (transient ischaemic attack)\n"
        "- Do not abbreviate patient name, ward, or procedure names\n"
        "- Maintain clinical accuracy\n\n"
        "Return the abbreviated text only."
    ),
}

# ---------------------------------------------------------------------------
# evaluation_prompts
# Quality metrics for generated clinical notes
# ---------------------------------------------------------------------------

evaluation_prompts: dict[str, Template] = {

    "calculate_fluency_prompt": Template(
        "You are evaluating the linguistic fluency of a synthetic clinical note "
        "for NHS training data quality assessment.\n\n"
        "CLINICAL NOTE:\n$NOTE\n\n"
        "Assess fluency on these dimensions:\n"
        "1. Grammatical correctness: Is the text grammatically correct for clinical prose?\n"
        "2. Clinical register: Is the language at an appropriate professional level?\n"
        "3. Readability: Is the note clear and easy to follow for a clinician?\n"
        "4. Coherence: Does the note flow logically from section to section?\n"
        "5. Authenticity: Does it read like a genuine NHS clinical note?\n\n"
        "Return a JSON object with:\n"
        "- fluency_score: float 0.0-1.0 (0=incomprehensible, 1=perfectly fluent)\n"
        "- grammar_score: float 0.0-1.0\n"
        "- clinical_register_score: float 0.0-1.0\n"
        "- readability_score: float 0.0-1.0\n"
        "- authenticity_score: float 0.0-1.0\n"
        "- fluency_issues: list of specific fluency issues found\n\n"
        "Return ONLY valid JSON."
    ),

    "calculate_groundedness_prompt": Template(
        "You are evaluating whether a synthetic clinical note is grounded in the "
        "reference material (patient and admission context) provided.\n\n"
        "CLINICAL NOTE:\n$NOTE\n\n"
        "REFERENCE MATERIAL:\n$REFERENCE\n\n"
        "Assess groundedness: does every factual claim in the note have support in "
        "the reference material, or is it a clinically reasonable inference?\n\n"
        "Check for:\n"
        "1. Fabricated patient details (wrong age, sex, diagnosis)\n"
        "2. Contradictory findings (test results inconsistent with diagnosis)\n"
        "3. Hallucinated medications not in the patient's drug list\n"
        "4. Timeline inconsistencies\n"
        "5. Clinically implausible values (impossible vital signs, lab results)\n\n"
        "Return a JSON object with:\n"
        "- groundedness_score: float 0.0-1.0\n"
        "- is_grounded: boolean (true if score >= 0.7)\n"
        "- ungrounded_claims: list of specific claims not supported by reference\n"
        "- contradictions: list of direct contradictions with reference material\n"
        "- hallucinations: list of fabricated facts\n\n"
        "Return ONLY valid JSON."
    ),

    "calculate_relevance_prompt": Template(
        "You are evaluating the clinical relevance of a synthetic clinical note "
        "to its stated purpose and context.\n\n"
        "CLINICAL NOTE:\n$NOTE\n\n"
        "REFERENCE MATERIAL (expected content and context):\n$REFERENCE\n\n"
        "Assess relevance: does the note contain clinically relevant content for "
        "the event type, patient, and diagnosis specified in the reference?\n\n"
        "Check for:\n"
        "1. Appropriate focus: Does the note focus on the right clinical issues?\n"
        "2. Event type match: Is the content appropriate for this type of clinical note?\n"
        "3. Diagnosis relevance: Is content specific to the patient's diagnosis?\n"
        "4. Clinical completeness: Are key required elements present?\n"
        "5. Absence of irrelevant content: Is there content inappropriate for this note type?\n\n"
        "Return a JSON object with:\n"
        "- relevance_score: float 0.0-1.0\n"
        "- is_relevant: boolean (true if score >= 0.7)\n"
        "- missing_elements: list of clinically important elements absent from note\n"
        "- irrelevant_content: list of inappropriate content found\n"
        "- diagnosis_specificity_score: float 0.0-1.0\n\n"
        "Return ONLY valid JSON."
    ),
}
