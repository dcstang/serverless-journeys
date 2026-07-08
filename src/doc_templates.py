"""
Document templates for synthetic clinical note generation.

Each template defines the structure and field guidance for a specific type of
clinical note or document used in NHS clinical documentation. Field descriptions
are written as guidance for the LLM generating the note content.

Adapted from nhsengland/synthetic_clinical_notes.
"""

# ---------------------------------------------------------------------------
# document_templates
# Maps note type name -> dict of section_name -> field guidance string
# ---------------------------------------------------------------------------

document_templates: dict[str, dict[str, str]] = {

    "patient_details": {
        "patient_demographics": (
            "Full name, date of birth, NHS number, MRN, address, GP practice. "
            "Include contact telephone number and next of kin details with relationship."
        ),
        "allergies_and_adverse_reactions": (
            "List all known drug allergies and adverse reactions with reaction type "
            "(e.g. anaphylaxis, rash, nausea). State NKDA (no known drug allergies) "
            "if applicable. Include food and environmental allergies if relevant."
        ),
        "past_medical_history": (
            "Chronological list of significant past medical and surgical history. "
            "Include dates of diagnoses, operations, and hospitalisations where known. "
            "Note relevant family history for hereditary conditions."
        ),
        "current_medications": (
            "Complete medication reconciliation list. For each medication include: "
            "drug name (generic), dose, frequency, route, and indication. "
            "Note any recent changes. Include OTC medications and supplements."
        ),
        "social_history": (
            "Occupation (current/previous), smoking status (pack years if ex/current smoker), "
            "alcohol consumption (units/week), recreational drug use, living situation "
            "(house/flat/alone/with family/care home), functional status and mobility aids, "
            "caring responsibilities."
        ),
    },

    "emergency_admission": {
        "chief_complaint": (
            "Primary reason for emergency attendance in the patient's own words where possible. "
            "Include onset, duration, and severity."
        ),
        "history_of_presenting_complaint": (
            "Detailed chronological account of the presenting illness. Use SOCRATES for pain: "
            "Site, Onset, Character, Radiation, Associated symptoms, Timing, Exacerbating/"
            "relieving factors, Severity (0-10). Include relevant systemic enquiry. "
            "Document pertinent positives and negatives."
        ),
        "past_medical_and_surgical_history": (
            "Relevant past medical conditions, previous admissions for same presentation, "
            "previous surgical procedures with approximate dates."
        ),
        "medications_on_admission": (
            "Full medication list as per patient/carer/GP records. Include dose, frequency, "
            "and route. Note any medications withheld and reason."
        ),
        "allergies": (
            "Known drug allergies with type of reaction. State NKDA if appropriate."
        ),
        "systems_review": (
            "Brief targeted systems review covering cardiovascular, respiratory, "
            "gastrointestinal, neurological, and other relevant systems."
        ),
        "examination_findings": (
            "General: Appearance, distress level, BMI category. "
            "Vital signs: HR, BP, RR, SpO2 (on air/O2), Temperature, GCS/AVPU. "
            "Cardiovascular: JVP, heart sounds, peripheral pulses, oedema. "
            "Respiratory: Air entry, added sounds, percussion. "
            "Abdomen: Inspection, palpation (tenderness, guarding, rigidity, organomegaly), "
            "bowel sounds. Neurological: GCS, focal deficits, cranial nerves as relevant."
        ),
        "investigations": (
            "Results of bloods (FBC, U&E, LFTs, CRP, troponin, lactate, glucose as relevant), "
            "urinalysis, ECG findings, imaging (CXR, CT, USS) with brief interpretation."
        ),
        "working_diagnosis_and_assessment": (
            "Primary working diagnosis with differential diagnoses listed. "
            "Assessment of severity/acuity. CURB-65, NEWS2, or other relevant scoring."
        ),
        "management_plan": (
            "Immediate management: Resuscitation, airway, IV access, fluids, analgesia, "
            "antibiotics, specific treatments. "
            "Investigations ordered. Referrals made. "
            "Monitoring plan and escalation criteria. "
            "Goals of care discussion if relevant. DNAR status."
        ),
    },

    "elective_admission": {
        "reason_for_admission": (
            "Planned procedure or investigation. Referring consultant and specialty. "
            "Date added to waiting list and wait time. Urgency (routine/urgent/soon)."
        ),
        "fitness_for_surgery_summary": (
            "Summary of pre-operative assessment findings. ASA grade. "
            "Anaesthetic risk classification. Relevant co-morbidities affecting surgical risk."
        ),
        "past_medical_and_surgical_history": (
            "Relevant medical conditions and previous surgical procedures. "
            "Previous anaesthetic issues. Family history of anaesthetic problems (MH)."
        ),
        "medications_on_admission": (
            "Full medication list. Note: anticoagulants and antiplatelets withheld/bridged "
            "as per protocol. Diabetic medications adjusted per fasting protocol. "
            "Regular medications to continue. New medications prescribed."
        ),
        "allergies": (
            "Drug allergies with reaction type. Latex allergy status. NKDA if appropriate."
        ),
        "examination_findings": (
            "General: Appearance, BMI. Vital signs: HR, BP, RR, SpO2, Temperature. "
            "Targeted examination relevant to planned procedure and anaesthetic assessment. "
            "Airway assessment: Mallampati score, mouth opening, neck mobility, dentition."
        ),
        "investigations_reviewed": (
            "Pre-operative investigations reviewed: FBC, U&E, clotting, group & save/crossmatch, "
            "ECG (interpretation), CXR (interpretation), echocardiography if relevant, "
            "lung function tests if relevant. Outstanding investigations."
        ),
        "consent": (
            "Procedure discussed with patient. Risks, benefits, and alternatives explained. "
            "Patient understands and consents. Capacity confirmed. Consent form completed."
        ),
        "management_plan": (
            "Scheduled procedure and theatre list position. Anaesthetic type planned. "
            "VTE prophylaxis plan. MRSA screen result. "
            "Specific pre-operative preparations (bowel prep, skin prep, fasting times). "
            "Post-operative care level (ward/HDU/ITU). Enhanced recovery pathway."
        ),
    },

    "ED event": {
        "triage_and_initial_assessment": (
            "Manchester Triage System category (1-5) with colour and time target. "
            "Chief complaint as presented at triage. "
            "Initial observations: HR, BP, RR, SpO2, Temperature, BM, pain score (0-10). "
            "NEWS2 score. Ambulance pre-alert information if applicable."
        ),
        "investigations_and_results": (
            "Investigations requested and results: bloods (specify which), urine dip, "
            "ECG findings, imaging (specify modality and brief report). "
            "Point of care testing results (VBG, lactate, troponin, BNP if performed)."
        ),
        "interventions_and_treatment": (
            "IV access established: gauge and site. "
            "Fluids administered: type, volume, rate. "
            "Medications given: drug, dose, route, time. "
            "Oxygen therapy: flow rate, delivery device, target SpO2. "
            "Other interventions: catheterisation, splinting, wound care, etc."
        ),
        "nursing_notes": (
            "Patient position and comfort. Pressure area assessment (Waterlow score). "
            "Fluid balance. Patient communication and anxieties addressed. "
            "Family/next of kin updated. Safeguarding concerns noted if any. "
            "Isolation precautions if applicable."
        ),
    },

    "ED review and hand-over": {
        "patient_summary": (
            "Brief patient identifier: name, age, presenting complaint. "
            "Time of arrival and total time in department so far."
        ),
        "clinical_course_in_ed": (
            "Summary of assessment, investigations performed and key results, "
            "treatments given and response. Trajectory: improving/stable/deteriorating."
        ),
        "working_diagnosis": (
            "Current working diagnosis or differential list in order of likelihood. "
            "Relevant risk stratification score results."
        ),
        "management_to_date": (
            "Treatments administered, IV fluids, medications given, procedures performed. "
            "Specialist team reviews: who was called, when, and their recommendation."
        ),
        "outstanding_tasks_and_plan": (
            "Investigations awaited: specify what and estimated time. "
            "Decisions pending: admission/discharge/further observation. "
            "If admitting: specialty, bed request status, receiving team notified. "
            "If discharging: safety-netting advice, GP letter, follow-up arranged. "
            "Escalation criteria and threshold for senior review."
        ),
        "handover_information": (
            "Handover to: (name, grade, team). Time of handover. "
            "Key concerns and must-not-miss diagnoses. "
            "Patient and family aware of plan."
        ),
    },

    "post take ward round": {
        "referral_and_admission_details": (
            "Source of referral (ED/GP/outpatient/transfer). Referring clinician and reason. "
            "Time and date of admission. Admitting team and registrar/consultant."
        ),
        "clinical_history_summary": (
            "Concise presenting complaint and history of presenting illness. "
            "Relevant past medical history, medications, allergies, and social history."
        ),
        "examination_findings": (
            "Full systematic examination findings on post-take ward round. "
            "Vital signs with NEWS2 score. Focused examination relevant to presentation. "
            "Note any abnormal findings in detail."
        ),
        "investigations_reviewed": (
            "All investigation results reviewed: bloods, imaging, ECG, microbiology. "
            "Interpretation of results in clinical context. "
            "Results pending and their clinical urgency."
        ),
        "clinical_impression": (
            "Post-take diagnosis or differential diagnoses. Severity assessment. "
            "Relevant scoring tools (e.g. CRB-65, GRACE, Child-Pugh). "
            "Comparison with admission findings and trajectory."
        ),
        "management_plan": (
            "Active problem list with specific management for each. "
            "Drug chart review and prescribing. VTE assessment completed. "
            "Level of care: ward/HDU/ITU decision and rationale. "
            "Monitoring requirements. "
            "Specialty review requests with urgency. "
            "Estimated length of stay and discharge planning initiated. "
            "Goals of care and DNAR if appropriate. Patient and family updated."
        ),
    },

    "general ward round": {
        "clinical_update": (
            "Date and time of ward round. Patient's subjective report overnight/since last review. "
            "Nursing staff update. Events since last round (procedures, results, incidents)."
        ),
        "observations_and_monitoring": (
            "Current observations: HR, BP, RR, SpO2, Temperature, urine output, pain score. "
            "NEWS2 trend. Fluid balance over 24 hours. Weight if relevant."
        ),
        "examination_findings": (
            "Focused examination relevant to active problems. "
            "Key positive and negative findings. Comparison with previous examination."
        ),
        "results_review": (
            "New investigation results available: interpret in clinical context. "
            "Microbiology results and antibiotic stewardship review. "
            "Outstanding results."
        ),
        "progress_and_assessment": (
            "Assessment of progress against management plan. "
            "Is patient improving/static/deteriorating? "
            "Active problem list review. Identify new problems."
        ),
        "updated_management_plan": (
            "Changes to treatment: medications added/stopped/adjusted with clear rationale. "
            "IV to oral switch where appropriate. Review lines and catheters. "
            "New investigation requests. Specialist review update. "
            "Allied health professional review (physio, OT, dietitian, SALT, pharmacy)."
        ),
        "discharge_planning": (
            "Estimated date of discharge (EDD). Discharge criteria for this patient. "
            "Criteria to reside: clinical/social/functional barriers. "
            "Social work referral if needed. Package of care assessment. "
            "TTOs (to take out medications) planning. Follow-up arrangements. "
            "GP and community team notifications planned."
        ),
    },

    "inter-specialty review": {
        "referral_details": (
            "Referring team: consultant, specialty, contact details. "
            "Reviewing team: specialty, reviewing clinician name and grade. "
            "Date and time of review. Urgency (routine/urgent/emergency)."
        ),
        "reason_for_referral": (
            "Specific clinical question or reason for review stated clearly. "
            "Relevant background: diagnosis, current management, specific concern."
        ),
        "patient_summary": (
            "Brief summary of patient: age, sex, presenting diagnosis, "
            "relevant PMH, current medications and observations."
        ),
        "assessment": (
            "Reviewing specialty's clinical assessment. "
            "History elicited by reviewing team. Examination findings. "
            "Review of relevant investigation results. "
            "Impression and working diagnosis from reviewing specialty's perspective."
        ),
        "recommendations": (
            "Specific management recommendations. "
            "Additional investigations requested by reviewing team. "
            "Medications prescribed or recommended. "
            "Follow-up plan: outpatient referral, repeat review, discharge instructions. "
            "Handback to referring team with clear ongoing responsibilities. "
            "Urgent review criteria specified."
        ),
    },

    "operation": {
        "procedure_details": (
            "Date and start/finish time. Theatre number and hospital site. "
            "Procedure performed (full name as per OPCS-4). "
            "Intended vs performed procedure (note any deviations and reason). "
            "Surgical approach: open/laparoscopic/endoscopic/robotic. "
            "Laterality if applicable."
        ),
        "surgical_team": (
            "Operating surgeon (grade and name). Assistant(s). "
            "Anaesthetist. Scrub nurse/ODP. "
            "Any other personnel present (students, observers – consent obtained)."
        ),
        "anaesthetic_details": (
            "Anaesthetic type: GA/spinal/epidural/regional/LA+sedation. "
            "Intubation/airway management. "
            "Notable events during induction. Intra-operative monitoring."
        ),
        "operative_findings": (
            "Intra-operative findings in detail. "
            "Anatomy, pathology encountered, tissue appearance. "
            "Specimens taken: description and destination (histology/microbiology). "
            "Swab and instrument count confirmed correct."
        ),
        "operative_technique": (
            "Step-by-step description of technique used. "
            "Implants/prostheses used: manufacturer, product name, size, batch/lot number. "
            "Suture materials used. Drains placed: type, size, location, output at end."
        ),
        "complications": (
            "Intra-operative complications (specify: bleeding, injury to adjacent structures, "
            "conversion from laparoscopic to open, etc.). "
            "Estimated blood loss (mL). Blood products given."
        ),
        "post_operative_plan": (
            "Immediate post-operative instructions: ward/HDU/ITU destination. "
            "Analgesia plan. IV fluids. Diet. Mobilisation. "
            "Drain management. Wound care. "
            "VTE prophylaxis commenced/continued. "
            "Specific monitoring required. "
            "Follow-up: clinic date, histology review, suture removal."
        ),
    },

    "pre-op assessment": {
        "reason_for_pre_op_assessment": (
            "Planned operation. Operating surgeon and specialty. "
            "Provisional theatre date. Urgency grade."
        ),
        "medical_history_and_comorbidities": (
            "Full medical history. Specific co-morbidities relevant to anaesthetic risk. "
            "Cardiovascular: IHD, CCF, arrhythmias, pacemaker. "
            "Respiratory: COPD, asthma, OSA, smoking. "
            "Metabolic: diabetes, obesity (BMI), thyroid disease. "
            "Renal/hepatic: CKD, liver disease. "
            "Previous anaesthetic history and any complications."
        ),
        "medications_and_allergies": (
            "Complete medication reconciliation. "
            "Anticoagulants: type, indication, INR if on warfarin, plan for peri-operative management. "
            "Antiplatelets: type, indication, plan. "
            "Diabetic medications: adjustment plan for fasting. "
            "Allergies with reaction type."
        ),
        "functional_capacity": (
            "Exercise tolerance in metabolic equivalents (METs). "
            "Can patient climb one flight of stairs or walk on level ground without symptoms? "
            "NYHA/MRC dyspnoea classification. "
            "Activities of daily living."
        ),
        "examination_and_investigations": (
            "BP, HR, weight, height, BMI, SpO2 on air. "
            "Cardiovascular and respiratory examination findings. "
            "Airway assessment: Mallampati score, inter-incisor distance, thyromental distance, "
            "neck mobility, jaw protrusion (ULBT). Dentures/crowns. "
            "Pre-operative investigations: FBC, U&E, creatinine, clotting, group and save, "
            "ECG (report), CXR (report), echo (report if done), PFTs (report if done)."
        ),
        "asa_grade_and_risk_assessment": (
            "ASA physical status classification (I-VI) with justification. "
            "Specific risk scores: RCRI for cardiac risk, P-POSSUM if applicable. "
            "Venous thromboembolism risk assessment (Caprini score). "
            "Anaesthetic risk discussed with patient."
        ),
        "pre_op_instructions": (
            "Fasting instructions: nil by mouth from (time) for solids, clear fluids allowed until (time). "
            "Medications to take on morning of surgery (with sip of water). "
            "Medications to withhold. Bowel preparation if required. "
            "MRSA decolonisation if required. Skin preparation instructions. "
            "Consent status. Further investigations or specialist reviews required before surgery."
        ),
    },

    "pre-op consent": {
        "procedure_and_indication": (
            "Full name of procedure as per OPCS-4. "
            "Clinical indication: diagnosis and why this procedure is recommended. "
            "Expected benefit of procedure."
        ),
        "intended_benefits": (
            "Specific goals of this operation for this patient. "
            "Expected functional and symptomatic improvement. "
            "Prognosis with and without intervention."
        ),
        "serious_risks": (
            "Generic surgical risks: bleeding requiring transfusion (approximate rate), "
            "infection (wound/deep), anaesthetic risks (DVT/PE, allergy, aspiration), "
            "damage to adjacent structures, conversion to open procedure (if laparoscopic). "
            "Procedure-specific serious risks: list with approximate incidence. "
            "Risk of death if applicable."
        ),
        "frequent_side_effects": (
            "Common post-operative symptoms: pain, nausea, constipation, fatigue. "
            "Procedure-specific common side effects with approximate incidence."
        ),
        "alternatives_discussed": (
            "Non-surgical alternatives discussed: conservative management, "
            "other procedures, watchful waiting. "
            "Patient's preference documented."
        ),
        "patient_questions_and_understanding": (
            "Questions raised by patient/family and answers given. "
            "Patient's stated understanding of procedure and risks. "
            "Interpreter used if applicable. Patient leaflet given."
        ),
        "consent_decision": (
            "Patient has capacity (Mental Capacity Act 2005 assessment if any doubt). "
            "Patient consents/declines. Consent form number and type (1/2/3/4). "
            "Signature obtained. Date. Witness if required."
        ),
    },

    "pre-op checklist": {
        "patient_identification": (
            "Patient ID confirmed with two identifiers: full name + DOB. "
            "NHS number verified. Wristband checked. "
            "Allergy band applied if appropriate."
        ),
        "consent_and_procedure": (
            "Valid consent form present and signed. Procedure site marked if applicable. "
            "Laterality confirmed with patient. "
            "Operation site marked (indelible pen) and confirmed with surgeon."
        ),
        "fasting_status": (
            "Fasting status confirmed: last solid food time, last clear fluid time. "
            "Fasting instructions followed. Chewing gum removed."
        ),
        "medications_and_investigations": (
            "Pre-operative medications given as prescribed (e.g. antihypertensives, "
            "anticoagulant bridge). Pre-operative investigations reviewed and results available. "
            "Group and screen/crossmatch in date. Blood products available if required."
        ),
        "implants_and_prostheses": (
            "Implant details confirmed available and correct (for joint replacement, etc.). "
            "MRSA screen result. "
            "Antibiotic prophylaxis prescribed and administered (time documented)."
        ),
        "theatre_safety_items": (
            "WHO Surgical Safety Checklist: Sign In completed. "
            "Patient identification confirmed with anaesthetic team. "
            "Known allergies confirmed. "
            "Anaesthetic machine and medication checks complete. "
            "Pulse oximeter functioning."
        ),
    },

    "anaesthetics assessment": {
        "pre_anaesthetic_assessment_summary": (
            "Date and purpose of assessment (pre-op / pre-procedure). "
            "Planned procedure and urgency. Previous assessments reviewed."
        ),
        "anaesthetic_history": (
            "Previous anaesthetics: type, complications (PONV, difficult intubation, "
            "awareness, allergy, MH, suxamethonium apnoea). "
            "Family history of anaesthetic complications. "
            "Previous ITU admissions."
        ),
        "airway_assessment": (
            "Mallampati classification (I-IV) at rest and with tongue out. "
            "Inter-incisor distance (cm). Thyromental distance (cm). "
            "Neck mobility and range of movement. "
            "Upper lip bite test (ULBT) class. "
            "Dentition: loose teeth, caps, bridges, dentures (upper/lower/partial). "
            "Predicted difficulty: easy/moderate/difficult/CICO risk. "
            "Anticipated airway plan documented (plan A/B/C/D)."
        ),
        "asa_grade_and_risk": (
            "ASA classification (I-VI) with clear justification. "
            "Specific risk scores used: RCRI, ACS-NSQIP, SORT if applicable. "
            "Surgical complexity: minor/intermediate/major/major+. "
            "Overall anaesthetic risk: low/medium/high/very high. "
            "Risk discussed with patient and documented."
        ),
        "anaesthetic_plan": (
            "Preferred anaesthetic technique: GA (TIVA/volatile)/spinal/epidural/CSE/"
            "peripheral nerve block/LA+sedation. Rationale for choice. "
            "Airway plan: LMA/ETT/awake fibreoptic intubation/video laryngoscopy. "
            "Monitoring plan: standard/arterial line/CVC/TOE/BIS/neuromonitoring. "
            "Analgesic plan: multimodal, regional techniques, opioid-sparing. "
            "Post-operative destination: ward/HDU/ITU with rationale."
        ),
        "specific_considerations": (
            "Any specific anaesthetic considerations: "
            "PONV prophylaxis plan (Apfel score). "
            "Temperature management. "
            "Blood conservation strategy (cell salvage, TXA). "
            "Positioning risks. "
            "Drug interactions and dose adjustments. "
            "Latex precautions if required."
        ),
    },

    "post-anaesthesia recovery": {
        "arrival_in_recovery": (
            "Time of arrival in post-anaesthesia care unit (PACU). "
            "Procedure performed and anaesthetic technique used. "
            "Airway on arrival: intubated/LMA/breathing spontaneously. "
            "Handover from anaesthetist: key intra-operative events, analgesia given, "
            "specific instructions."
        ),
        "recovery_observations": (
            "Aldrete/Modified Aldrete Score on arrival and at intervals. "
            "Observations: HR, BP (target range specified), SpO2, RR, Temperature, "
            "pain score (0-10), sedation score (RASS or numerical). "
            "Oxygen requirements. Airway status and management. "
            "Fluid balance: IV fluids given, blood products, urine output, drain output."
        ),
        "post_operative_analgesia": (
            "Pain assessment using validated scale (NRS 0-10 or VRS). "
            "Analgesics administered: name, dose, route, time. "
            "Breakthrough analgesia given. Patient-controlled analgesia (PCA) commenced if applicable. "
            "Epidural/regional block functioning as expected: dermatomal level if spinal. "
            "Pain goal achieved before discharge from recovery."
        ),
        "post_operative_nausea_and_vomiting": (
            "PONV assessment: presence, severity, antiemetics given. "
            "Apfel score reviewed. "
            "Episodes of vomiting: frequency, volume. "
            "Response to antiemetic treatment."
        ),
        "complications_and_events": (
            "Any post-operative complications: "
            "Haemodynamic instability: cause and management. "
            "Respiratory: desaturation, laryngospasm, bronchospasm. "
            "Neurological: prolonged emergence, agitation. "
            "Surgical: bleeding from wound, drain output. "
            "Escalation actions taken: anaesthetist called, senior review."
        ),
        "discharge_from_recovery": (
            "Time of discharge from PACU. Modified Aldrete Score at discharge (target ≥9). "
            "Destination: ward/HDU/ITU. "
            "Handover information given to ward nurse. "
            "Specific post-op instructions communicated: analgesia, observations frequency, "
            "drain management, catheter management, diet, mobilisation."
        ),
    },

    "nursing": {
        "vital_signs_and_observations": (
            "Time of observations. "
            "HR (rhythm noted), BP (both arms if indicated), RR, SpO2 (on air or O2 – specify), "
            "Temperature, GCS or AVPU score, Blood glucose if diabetic or unwell. "
            "NEWS2 score and escalation action taken. Pain score (0-10 NRS)."
        ),
        "medications_administered": (
            "Medications given: drug name, dose, route, time, administered by. "
            "Controlled drugs: co-signature obtained. "
            "IV medications: infusion rate, site check. "
            "PRN medications given: indication documented. "
            "Any medications refused or omitted: reason documented."
        ),
        "fluid_balance": (
            "Input: oral fluids, IV fluids (type and rate), nasogastric feeds, blood products. "
            "Output: urine (volume and colour), wound drain, stoma, vomit. "
            "Running 24-hour balance. Fluid balance target if prescribed."
        ),
        "care_provided": (
            "Personal care: hygiene, mouth care, pressure area care. "
            "Repositioning frequency and pressure ulcer assessment (Braden/Waterlow score). "
            "Falls risk assessment (score and actions taken). "
            "Lines and catheters: site inspected, dressing changed if indicated, "
            "catheter care, urinary catheter drainage noted. "
            "Wound inspection: appearance, dressing status, exudate type and amount."
        ),
        "patient_condition_and_response": (
            "Patient's reported symptoms and concerns. "
            "Orientation and mental state. "
            "Communication needs (hearing/language/cognitive). "
            "Response to treatments given. "
            "Eating and drinking: dietary intake, supplements given. "
            "Sleep and rest periods."
        ),
        "plan_and_escalation": (
            "Tasks completed from nursing care plan. "
            "Outstanding tasks for next shift. "
            "Escalations made: clinician notified (name, time, concern, response). "
            "Patient and family communication. "
            "Discharge planning contributions."
        ),
    },

    "therapy": {
        "referral_and_background": (
            "Therapy type: Physiotherapy / Occupational Therapy / Speech and Language Therapy / "
            "Dietetics / Pharmacy / Social Work. "
            "Referral date and reason. Referrer name and team. "
            "Patient diagnosis and co-morbidities relevant to therapy."
        ),
        "assessment_findings": (
            "Objective assessment relevant to therapy discipline. "
            "Physiotherapy: range of movement, muscle power (MRC grading), balance, gait. "
            "Occupational therapy: functional independence (Barthel/FIM), cognitive function, "
            "home environment. "
            "Speech and language: swallowing assessment (FEES/videofluoroscopy), "
            "communication aids needed. "
            "Dietetics: nutritional screening (MUST score), dietary intake, weight changes. "
            "Pre-morbid function and baseline comparison."
        ),
        "interventions": (
            "Treatment provided this session: exercises, mobilisation, "
            "equipment provision, education, dietary advice, communication aids. "
            "Duration of session. Patient cooperation and participation. "
            "Equipment provided: walking aids, splints, adaptive equipment."
        ),
        "goals": (
            "Short-term goals (24-48 hours): specific, measurable, achievable. "
            "Long-term goals (discharge target): functional level, living situation. "
            "Patient and carer goals incorporated. "
            "Discharge destination aim: home/rehab/care facility."
        ),
        "progress": (
            "Progress against previous session goals. "
            "Response to intervention: improving/static/declining. "
            "Barriers to progress: pain, fatigue, cognition, motivation, social factors."
        ),
        "recommendations_and_plan": (
            "Frequency and duration of further therapy input. "
            "Home therapy programme given. "
            "Equipment to order. "
            "Referrals to community therapy services. "
            "Recommendations for discharge planning: timing, support needed, adaptations. "
            "Carer training required."
        ),
    },

    "misc": {
        "clinical_note": (
            "Free text clinical note. Include date, time, clinician name and grade. "
            "Note should be legible, professional, factual, and clinically relevant. "
            "Include reason for note, key findings or events, actions taken, and plan."
        ),
    },
}

# ---------------------------------------------------------------------------
# template_sections_to_combine
# Maps note type -> list of section keys to combine into a single note body
# Used when generating a single flowing document from multiple template sections
# ---------------------------------------------------------------------------

template_sections_to_combine: dict[str, list[str]] = {
    "patient_details": [
        "patient_demographics",
        "allergies_and_adverse_reactions",
        "past_medical_history",
        "current_medications",
        "social_history",
    ],
    "emergency_admission": [
        "chief_complaint",
        "history_of_presenting_complaint",
        "past_medical_and_surgical_history",
        "medications_on_admission",
        "allergies",
        "systems_review",
        "examination_findings",
        "investigations",
        "working_diagnosis_and_assessment",
        "management_plan",
    ],
    "elective_admission": [
        "reason_for_admission",
        "fitness_for_surgery_summary",
        "past_medical_and_surgical_history",
        "medications_on_admission",
        "allergies",
        "examination_findings",
        "investigations_reviewed",
        "consent",
        "management_plan",
    ],
    "ED event": [
        "triage_and_initial_assessment",
        "investigations_and_results",
        "interventions_and_treatment",
        "nursing_notes",
    ],
    "ED review and hand-over": [
        "patient_summary",
        "clinical_course_in_ed",
        "working_diagnosis",
        "management_to_date",
        "outstanding_tasks_and_plan",
        "handover_information",
    ],
    "post take ward round": [
        "referral_and_admission_details",
        "clinical_history_summary",
        "examination_findings",
        "investigations_reviewed",
        "clinical_impression",
        "management_plan",
    ],
    "general ward round": [
        "clinical_update",
        "observations_and_monitoring",
        "examination_findings",
        "results_review",
        "progress_and_assessment",
        "updated_management_plan",
        "discharge_planning",
    ],
    "inter-specialty review": [
        "referral_details",
        "reason_for_referral",
        "patient_summary",
        "assessment",
        "recommendations",
    ],
    "operation": [
        "procedure_details",
        "surgical_team",
        "anaesthetic_details",
        "operative_findings",
        "operative_technique",
        "complications",
        "post_operative_plan",
    ],
    "pre-op assessment": [
        "reason_for_pre_op_assessment",
        "medical_history_and_comorbidities",
        "medications_and_allergies",
        "functional_capacity",
        "examination_and_investigations",
        "asa_grade_and_risk_assessment",
        "pre_op_instructions",
    ],
    "pre-op consent": [
        "procedure_and_indication",
        "intended_benefits",
        "serious_risks",
        "frequent_side_effects",
        "alternatives_discussed",
        "patient_questions_and_understanding",
        "consent_decision",
    ],
    "pre-op checklist": [
        "patient_identification",
        "consent_and_procedure",
        "fasting_status",
        "medications_and_investigations",
        "implants_and_prostheses",
        "theatre_safety_items",
    ],
    "anaesthetics assessment": [
        "pre_anaesthetic_assessment_summary",
        "anaesthetic_history",
        "airway_assessment",
        "asa_grade_and_risk",
        "anaesthetic_plan",
        "specific_considerations",
    ],
    "post-anaesthesia recovery": [
        "arrival_in_recovery",
        "recovery_observations",
        "post_operative_analgesia",
        "post_operative_nausea_and_vomiting",
        "complications_and_events",
        "discharge_from_recovery",
    ],
    "nursing": [
        "vital_signs_and_observations",
        "medications_administered",
        "fluid_balance",
        "care_provided",
        "patient_condition_and_response",
        "plan_and_escalation",
    ],
    "therapy": [
        "referral_and_background",
        "assessment_findings",
        "interventions",
        "goals",
        "progress",
        "recommendations_and_plan",
    ],
    "misc": [
        "clinical_note",
    ],
}
