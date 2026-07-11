"""
Prompt templates for synthetic clinical note generation.

All prompts use Python's string.Template with $VARIABLE substitution.
Categories:
  - patient_prompts: patient and admission generation
  - journey_prompts: patient journey / event sequence generation
  - note_prompts: clinical note generation
  - processing_prompts: post-processing (cleaning, abbreviations)
  - evaluation_prompts: quality evaluation metrics
  - research_prompts: web-search-grounded code research (src/codes/research.py)
  - correction_prompts: targeted backward-pass correction (main.py)

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
        "- estimated_los_days: estimated length of stay as integer, realistically between "
        "$MIN_LOS_DAYS and $MAX_LOS_DAYS days for this presentation\n\n"
        "Use realistic UK clinical terminology and NHS documentation conventions. "
        "Measurements in SI units (mmHg, mmol/L, g/dL). "
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
        "TARGET NUMBER OF EVENTS: approximately $TARGET_EVENT_COUNT\n\n"
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
        "improvement -> discharge)\n"
        "- Use the target number of events above as a guide, not a hard rule: a short "
        "admission-date-to-discharge-date window or a simple presentation may reasonably "
        "need fewer events, and a long or complex admission may need more. Clinical realism "
        "given the actual LOS and admission type takes priority over hitting the target "
        "exactly.\n\n"
        "Return a JSON array of event objects. Each event object must have:\n"
        "- event_type: one of the possible event types listed above\n"
        "- event_date: ISO date (YYYY-MM-DD) within admission-discharge range\n"
        "- event_time: HH:MM (24-hour)\n"
        "- event_order: integer starting from 1\n"
        "- brief_description: one sentence describing what happens at this event\n"
        "- clinician_type: type of clinician performing this event\n\n"
        "Return ONLY a valid JSON array."
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

}

# ---------------------------------------------------------------------------
# processing_prompts
# Post-processing utilities: cleaning and abbreviation augmentation
# ---------------------------------------------------------------------------

processing_prompts: dict[str, Template] = {

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

# ---------------------------------------------------------------------------
# research_prompts
# Web-search-grounded research for codes with no curated dictionary entry
# ---------------------------------------------------------------------------

research_prompts: dict[str, Template] = {

    "research_code_prompt": Template(
        "You are a clinical coding specialist. Based on the web search results below, "
        "produce a concise, clinically accurate summary of the following code so it can "
        "be used to generate realistic NHS clinical documentation.\n\n"
        "CODE: $CODE\n"
        "CODE SYSTEM: $CODE_SYSTEM ($CODE_KIND code)\n\n"
        "WEB SEARCH RESULTS:\n$SEARCH_RESULTS\n\n"
        "Using ONLY information grounded in the search results above (do not invent "
        "clinical facts not supported by them), return a JSON object with:\n"
        "- description: concise clinical name/description of this code (one line)\n"
        "- specialty: the most relevant NHS clinical/surgical specialty\n"
        "- type: 'emergency' or 'elective' - typical acuity for this diagnosis/procedure\n"
        "- typical_los_days_min: integer, typical minimum NHS length of stay in days\n"
        "- typical_los_days_max: integer, typical maximum NHS length of stay in days\n"
        "- confidence: 'high' or 'low' - use 'low' if the search results were too sparse, "
        "off-topic, or ambiguous to confidently determine the above\n\n"
        "Return ONLY valid JSON."
    ),
}

# ---------------------------------------------------------------------------
# correction_prompts
# Targeted backward-pass correction when a driving code isn't reflected in
# already-generated content
# ---------------------------------------------------------------------------

correction_prompts: dict[str, Template] = {

    "correct_admission_prompt": Template(
        "You are an NHS clinician revising a synthetic admission record. The record "
        "below was meant to be grounded in a specific diagnosis/procedure, but a "
        "review found that it does not clearly reflect it. Revise the record so it "
        "explicitly and clinically incorporates the code below, changing as little "
        "else as possible.\n\n"
        "CODE TO INCORPORATE: $CODE\n"
        "CLINICAL CONTEXT:\n$CODE_CONTEXT\n\n"
        "CURRENT ADMISSION RECORD (JSON):\n$CURRENT_ADMISSION\n\n"
        "Return the revised admission record as a JSON object with the same keys as "
        "the current record above, updated so that chief_complaint, "
        "history_of_presenting_complaint / indication, working_diagnosis / "
        "planned_procedure, and management_plan (whichever of these keys are present) "
        "clearly and specifically reflect the code above. "
        "Return ONLY valid JSON."
    ),

    "correct_note_prompt": Template(
        "You are an NHS clinician revising a synthetic clinical note. The note below "
        "was meant to be consistent with a specific diagnosis/procedure, but a review "
        "found that it does not clearly reflect it. Revise the note so it explicitly "
        "and clinically incorporates the code below, changing as little else as "
        "possible and preserving the note's original structure and style.\n\n"
        "CODE TO INCORPORATE: $CODE\n"
        "CLINICAL CONTEXT:\n$CODE_CONTEXT\n\n"
        "CURRENT NOTE TEXT:\n$CURRENT_NOTE\n\n"
        "Return the revised note text only - no JSON wrapper, no explanatory comments."
    ),
}
