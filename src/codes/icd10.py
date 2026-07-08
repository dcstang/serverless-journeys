"""
ICD-10 code handling for synthetic clinical note generation.

ICD-10 (International Classification of Diseases, 10th Revision) codes follow
the format: letter + 2 digits + optional decimal + digit(s), e.g. I21.0, J18.1.

This module builds a src.codes.registry.CodeSystem from ICD10_CODES and
registers it under the key 'icd10'. The functions below are thin wrappers
around the generic registry implementation, kept for convenience and
backward compatibility - equivalent behaviour is available for any
registered code system via src.codes.registry directly.
"""

from __future__ import annotations

from src.codes import registry

# ---------------------------------------------------------------------------
# Curated ICD-10 code dictionary
# Keys: ICD-10 code string
# Values: dict with description, chapter_name, specialty, admission_type,
#         typical_los_days (tuple: min, max)
# ---------------------------------------------------------------------------
ICD10_CODES: dict[str, dict] = {
    # Chapter I – Circulatory System
    "I21.0": {
        "description": "Acute transmural myocardial infarction of anterior wall (STEMI)",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Cardiology",
        "admission_type": "emergency",
        "typical_los_days": (4, 7),
    },
    "I21.1": {
        "description": "Acute transmural myocardial infarction of inferior wall (STEMI)",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Cardiology",
        "admission_type": "emergency",
        "typical_los_days": (3, 6),
    },
    "I21.4": {
        "description": "Acute subendocardial myocardial infarction (NSTEMI)",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Cardiology",
        "admission_type": "emergency",
        "typical_los_days": (3, 5),
    },
    "I48.0": {
        "description": "Paroxysmal atrial fibrillation",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Cardiology",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    "I48.1": {
        "description": "Persistent atrial fibrillation",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Cardiology",
        "admission_type": "elective",
        "typical_los_days": (1, 2),
    },
    "I50.0": {
        "description": "Congestive heart failure",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Cardiology",
        "admission_type": "emergency",
        "typical_los_days": (5, 10),
    },
    "I63.9": {
        "description": "Cerebral infarction, unspecified (ischaemic stroke)",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Stroke Medicine",
        "admission_type": "emergency",
        "typical_los_days": (5, 14),
    },
    "I64": {
        "description": "Stroke, not specified as haemorrhage or infarction",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Stroke Medicine",
        "admission_type": "emergency",
        "typical_los_days": (5, 14),
    },
    "I10": {
        "description": "Essential (primary) hypertension",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "General Medicine",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    "I26.9": {
        "description": "Pulmonary embolism without mention of acute cor pulmonale",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Respiratory Medicine",
        "admission_type": "emergency",
        "typical_los_days": (3, 7),
    },
    # Chapter J – Respiratory System
    "J18.1": {
        "description": "Lobar pneumonia, unspecified organism",
        "chapter_name": "Diseases of the respiratory system",
        "specialty": "Respiratory Medicine",
        "admission_type": "emergency",
        "typical_los_days": (4, 8),
    },
    "J44.1": {
        "description": "Chronic obstructive pulmonary disease with acute exacerbation",
        "chapter_name": "Diseases of the respiratory system",
        "specialty": "Respiratory Medicine",
        "admission_type": "emergency",
        "typical_los_days": (4, 8),
    },
    "J45.9": {
        "description": "Asthma, unspecified",
        "chapter_name": "Diseases of the respiratory system",
        "specialty": "Respiratory Medicine",
        "admission_type": "emergency",
        "typical_los_days": (2, 4),
    },
    "J96.0": {
        "description": "Acute respiratory failure, unspecified",
        "chapter_name": "Diseases of the respiratory system",
        "specialty": "Respiratory Medicine",
        "admission_type": "emergency",
        "typical_los_days": (5, 14),
    },
    # Chapter K – Digestive System
    "K35.2": {
        "description": "Acute appendicitis with generalised peritonitis",
        "chapter_name": "Diseases of the digestive system",
        "specialty": "General Surgery",
        "admission_type": "emergency",
        "typical_los_days": (4, 7),
    },
    "K35.8": {
        "description": "Acute appendicitis with other complications",
        "chapter_name": "Diseases of the digestive system",
        "specialty": "General Surgery",
        "admission_type": "emergency",
        "typical_los_days": (3, 5),
    },
    "K80.0": {
        "description": "Calculus of gallbladder with acute cholecystitis",
        "chapter_name": "Diseases of the digestive system",
        "specialty": "General Surgery",
        "admission_type": "emergency",
        "typical_los_days": (3, 6),
    },
    "K85.9": {
        "description": "Acute pancreatitis, unspecified",
        "chapter_name": "Diseases of the digestive system",
        "specialty": "General Surgery",
        "admission_type": "emergency",
        "typical_los_days": (5, 10),
    },
    "K92.1": {
        "description": "Melaena (upper gastrointestinal bleeding)",
        "chapter_name": "Diseases of the digestive system",
        "specialty": "Gastroenterology",
        "admission_type": "emergency",
        "typical_los_days": (3, 6),
    },
    "K57.3": {
        "description": "Diverticular disease of large intestine without perforation or abscess",
        "chapter_name": "Diseases of the digestive system",
        "specialty": "General Surgery",
        "admission_type": "emergency",
        "typical_los_days": (3, 6),
    },
    # Chapter N – Genitourinary System
    "N20.0": {
        "description": "Calculus of kidney (nephrolithiasis)",
        "chapter_name": "Diseases of the genitourinary system",
        "specialty": "Urology",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    "N39.0": {
        "description": "Urinary tract infection, site not specified",
        "chapter_name": "Diseases of the genitourinary system",
        "specialty": "General Medicine",
        "admission_type": "emergency",
        "typical_los_days": (2, 5),
    },
    "N18.3": {
        "description": "Chronic kidney disease, stage 3",
        "chapter_name": "Diseases of the genitourinary system",
        "specialty": "Nephrology",
        "admission_type": "emergency",
        "typical_los_days": (3, 7),
    },
    "N17.9": {
        "description": "Acute kidney injury, unspecified",
        "chapter_name": "Diseases of the genitourinary system",
        "specialty": "Nephrology",
        "admission_type": "emergency",
        "typical_los_days": (4, 8),
    },
    # Chapter S/T – Injury, Poisoning
    "S72.0": {
        "description": "Fracture of neck of femur (hip fracture)",
        "chapter_name": "Injury, poisoning and certain other consequences of external causes",
        "specialty": "Trauma and Orthopaedics",
        "admission_type": "emergency",
        "typical_los_days": (7, 14),
    },
    "S52.5": {
        "description": "Fracture of lower end of radius (Colles fracture)",
        "chapter_name": "Injury, poisoning and certain other consequences of external causes",
        "specialty": "Trauma and Orthopaedics",
        "admission_type": "emergency",
        "typical_los_days": (1, 2),
    },
    "S06.0": {
        "description": "Concussion",
        "chapter_name": "Injury, poisoning and certain other consequences of external causes",
        "specialty": "Emergency Medicine",
        "admission_type": "emergency",
        "typical_los_days": (1, 2),
    },
    # Chapter C – Neoplasms
    "C34.1": {
        "description": "Malignant neoplasm of upper lobe, bronchus or lung",
        "chapter_name": "Neoplasms",
        "specialty": "Respiratory Medicine / Oncology",
        "admission_type": "elective",
        "typical_los_days": (3, 7),
    },
    "C18.9": {
        "description": "Malignant neoplasm of colon, unspecified",
        "chapter_name": "Neoplasms",
        "specialty": "Colorectal Surgery / Oncology",
        "admission_type": "elective",
        "typical_los_days": (5, 10),
    },
    "C61": {
        "description": "Malignant neoplasm of prostate",
        "chapter_name": "Neoplasms",
        "specialty": "Urology / Oncology",
        "admission_type": "elective",
        "typical_los_days": (2, 4),
    },
    "C50.9": {
        "description": "Malignant neoplasm of breast, unspecified",
        "chapter_name": "Neoplasms",
        "specialty": "Breast Surgery / Oncology",
        "admission_type": "elective",
        "typical_los_days": (2, 5),
    },
    # Chapter E – Endocrine and Metabolic
    "E11.9": {
        "description": "Type 2 diabetes mellitus without complications",
        "chapter_name": "Endocrine, nutritional and metabolic diseases",
        "specialty": "Endocrinology / General Medicine",
        "admission_type": "emergency",
        "typical_los_days": (2, 5),
    },
    "E10.1": {
        "description": "Type 1 diabetes mellitus with ketoacidosis",
        "chapter_name": "Endocrine, nutritional and metabolic diseases",
        "specialty": "Endocrinology",
        "admission_type": "emergency",
        "typical_los_days": (3, 5),
    },
    "E86.0": {
        "description": "Dehydration",
        "chapter_name": "Endocrine, nutritional and metabolic diseases",
        "specialty": "General Medicine",
        "admission_type": "emergency",
        "typical_los_days": (2, 4),
    },
    # Chapter G – Nervous System
    "G35": {
        "description": "Multiple sclerosis",
        "chapter_name": "Diseases of the nervous system",
        "specialty": "Neurology",
        "admission_type": "emergency",
        "typical_los_days": (3, 7),
    },
    "G43.9": {
        "description": "Migraine, unspecified",
        "chapter_name": "Diseases of the nervous system",
        "specialty": "Neurology",
        "admission_type": "emergency",
        "typical_los_days": (1, 2),
    },
    "G40.9": {
        "description": "Epilepsy, unspecified",
        "chapter_name": "Diseases of the nervous system",
        "specialty": "Neurology",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    # Chapter M – Musculoskeletal
    "M16.1": {
        "description": "Primary osteoarthritis of hip",
        "chapter_name": "Diseases of the musculoskeletal system and connective tissue",
        "specialty": "Trauma and Orthopaedics",
        "admission_type": "elective",
        "typical_los_days": (3, 5),
    },
    "M17.1": {
        "description": "Primary osteoarthritis of knee",
        "chapter_name": "Diseases of the musculoskeletal system and connective tissue",
        "specialty": "Trauma and Orthopaedics",
        "admission_type": "elective",
        "typical_los_days": (3, 5),
    },
    "M54.5": {
        "description": "Low back pain",
        "chapter_name": "Diseases of the musculoskeletal system and connective tissue",
        "specialty": "Orthopaedics / Pain Medicine",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    # Chapter F – Mental and Behavioural Disorders
    "F32.1": {
        "description": "Moderate depressive episode",
        "chapter_name": "Mental and behavioural disorders",
        "specialty": "Psychiatry",
        "admission_type": "emergency",
        "typical_los_days": (7, 21),
    },
    "F20.9": {
        "description": "Schizophrenia, unspecified",
        "chapter_name": "Mental and behavioural disorders",
        "specialty": "Psychiatry",
        "admission_type": "emergency",
        "typical_los_days": (14, 28),
    },
    # Chapter A/B – Infectious Diseases
    "A41.9": {
        "description": "Sepsis, unspecified organism",
        "chapter_name": "Certain infectious and parasitic diseases",
        "specialty": "General Medicine / Infectious Diseases",
        "admission_type": "emergency",
        "typical_los_days": (5, 14),
    },
    "A40.0": {
        "description": "Sepsis due to Streptococcus",
        "chapter_name": "Certain infectious and parasitic diseases",
        "specialty": "General Medicine / Infectious Diseases",
        "admission_type": "emergency",
        "typical_los_days": (5, 10),
    },
    "B34.9": {
        "description": "Viral infection, unspecified",
        "chapter_name": "Certain infectious and parasitic diseases",
        "specialty": "General Medicine",
        "admission_type": "emergency",
        "typical_los_days": (2, 5),
    },
    # Additional common codes
    "R07.4": {
        "description": "Chest pain, unspecified",
        "chapter_name": "Symptoms, signs and abnormal clinical and laboratory findings",
        "specialty": "Cardiology / Emergency Medicine",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    "R55": {
        "description": "Syncope and collapse",
        "chapter_name": "Symptoms, signs and abnormal clinical and laboratory findings",
        "specialty": "Cardiology / General Medicine",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    "Z96.6": {
        "description": "Presence of orthopaedic joint implants",
        "chapter_name": "Factors influencing health status and contact with health services",
        "specialty": "Trauma and Orthopaedics",
        "admission_type": "elective",
        "typical_los_days": (3, 5),
    },
    # More codes to reach ~80
    "I35.0": {
        "description": "Nonrheumatic aortic valve stenosis",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "Cardiology / Cardiac Surgery",
        "admission_type": "elective",
        "typical_los_days": (5, 8),
    },
    "I84.9": {
        "description": "Haemorrhoids, unspecified",
        "chapter_name": "Diseases of the circulatory system",
        "specialty": "General Surgery / Colorectal Surgery",
        "admission_type": "elective",
        "typical_los_days": (1, 2),
    },
    "J06.9": {
        "description": "Acute upper respiratory infection, unspecified",
        "chapter_name": "Diseases of the respiratory system",
        "specialty": "General Medicine",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    "K25.3": {
        "description": "Gastric ulcer, acute without haemorrhage or perforation",
        "chapter_name": "Diseases of the digestive system",
        "specialty": "Gastroenterology",
        "admission_type": "emergency",
        "typical_los_days": (3, 5),
    },
    "K40.9": {
        "description": "Unilateral inguinal hernia, without obstruction or gangrene",
        "chapter_name": "Diseases of the digestive system",
        "specialty": "General Surgery",
        "admission_type": "elective",
        "typical_los_days": (1, 2),
    },
    "K56.6": {
        "description": "Ileus, unspecified (intestinal obstruction)",
        "chapter_name": "Diseases of the digestive system",
        "specialty": "General Surgery",
        "admission_type": "emergency",
        "typical_los_days": (5, 10),
    },
    "N40": {
        "description": "Benign prostatic hyperplasia",
        "chapter_name": "Diseases of the genitourinary system",
        "specialty": "Urology",
        "admission_type": "elective",
        "typical_los_days": (2, 4),
    },
    "N81.1": {
        "description": "Cystocele",
        "chapter_name": "Diseases of the genitourinary system",
        "specialty": "Urology / Gynaecology",
        "admission_type": "elective",
        "typical_los_days": (2, 3),
    },
    "S82.0": {
        "description": "Fracture of patella",
        "chapter_name": "Injury, poisoning and certain other consequences of external causes",
        "specialty": "Trauma and Orthopaedics",
        "admission_type": "emergency",
        "typical_los_days": (2, 4),
    },
    "T39.1": {
        "description": "Poisoning by 4-aminophenol derivatives (paracetamol overdose)",
        "chapter_name": "Injury, poisoning and certain other consequences of external causes",
        "specialty": "Emergency Medicine / General Medicine",
        "admission_type": "emergency",
        "typical_los_days": (2, 4),
    },
    "C20": {
        "description": "Malignant neoplasm of rectum",
        "chapter_name": "Neoplasms",
        "specialty": "Colorectal Surgery / Oncology",
        "admission_type": "elective",
        "typical_los_days": (5, 10),
    },
    "C16.9": {
        "description": "Malignant neoplasm of stomach, unspecified",
        "chapter_name": "Neoplasms",
        "specialty": "Upper GI Surgery / Oncology",
        "admission_type": "elective",
        "typical_los_days": (7, 14),
    },
    "E03.9": {
        "description": "Hypothyroidism, unspecified",
        "chapter_name": "Endocrine, nutritional and metabolic diseases",
        "specialty": "Endocrinology / General Medicine",
        "admission_type": "elective",
        "typical_los_days": (2, 4),
    },
    "G20": {
        "description": "Parkinson's disease",
        "chapter_name": "Diseases of the nervous system",
        "specialty": "Neurology",
        "admission_type": "emergency",
        "typical_los_days": (4, 8),
    },
    "G30.9": {
        "description": "Alzheimer's disease, unspecified",
        "chapter_name": "Diseases of the nervous system",
        "specialty": "Geriatric Medicine / Neurology",
        "admission_type": "emergency",
        "typical_los_days": (7, 14),
    },
    "M05.9": {
        "description": "Seropositive rheumatoid arthritis, unspecified",
        "chapter_name": "Diseases of the musculoskeletal system and connective tissue",
        "specialty": "Rheumatology",
        "admission_type": "elective",
        "typical_los_days": (2, 5),
    },
    "M80.0": {
        "description": "Age-related osteoporosis with current pathological fracture",
        "chapter_name": "Diseases of the musculoskeletal system and connective tissue",
        "specialty": "Trauma and Orthopaedics / Rheumatology",
        "admission_type": "emergency",
        "typical_los_days": (4, 8),
    },
    "F41.1": {
        "description": "Generalised anxiety disorder",
        "chapter_name": "Mental and behavioural disorders",
        "specialty": "Psychiatry",
        "admission_type": "elective",
        "typical_los_days": (7, 14),
    },
    "A09": {
        "description": "Other and unspecified infectious gastroenteritis and colitis",
        "chapter_name": "Certain infectious and parasitic diseases",
        "specialty": "General Medicine / Gastroenterology",
        "admission_type": "emergency",
        "typical_los_days": (2, 4),
    },
    "Z51.1": {
        "description": "Chemotherapy session for neoplasm",
        "chapter_name": "Factors influencing health status and contact with health services",
        "specialty": "Oncology",
        "admission_type": "elective",
        "typical_los_days": (1, 2),
    },
    "R00.0": {
        "description": "Tachycardia, unspecified",
        "chapter_name": "Symptoms, signs and abnormal clinical and laboratory findings",
        "specialty": "Cardiology / Emergency Medicine",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    "R10.4": {
        "description": "Other and unspecified abdominal pain",
        "chapter_name": "Symptoms, signs and abnormal clinical and laboratory findings",
        "specialty": "General Surgery / Emergency Medicine",
        "admission_type": "emergency",
        "typical_los_days": (1, 3),
    },
    "R50.9": {
        "description": "Fever, unspecified",
        "chapter_name": "Symptoms, signs and abnormal clinical and laboratory findings",
        "specialty": "General Medicine",
        "admission_type": "emergency",
        "typical_los_days": (2, 5),
    },
    "L03.1": {
        "description": "Cellulitis of other parts of limb",
        "chapter_name": "Diseases of the skin and subcutaneous tissue",
        "specialty": "General Medicine / Dermatology",
        "admission_type": "emergency",
        "typical_los_days": (3, 6),
    },
    "H35.3": {
        "description": "Degeneration of macula and posterior pole (macular degeneration)",
        "chapter_name": "Diseases of the eye and adnexa",
        "specialty": "Ophthalmology",
        "admission_type": "elective",
        "typical_los_days": (1, 1),
    },
    "O80": {
        "description": "Single spontaneous delivery",
        "chapter_name": "Pregnancy, childbirth and the puerperium",
        "specialty": "Obstetrics",
        "admission_type": "emergency",
        "typical_los_days": (2, 3),
    },
    "Q21.1": {
        "description": "Atrial septal defect",
        "chapter_name": "Congenital malformations, deformations and chromosomal abnormalities",
        "specialty": "Cardiology / Cardiac Surgery",
        "admission_type": "elective",
        "typical_los_days": (3, 7),
    },
}


# ---------------------------------------------------------------------------
# Chapter-letter fallback specialty map, used for codes outside ICD10_CODES
# ---------------------------------------------------------------------------
_CHAPTER_SPECIALTY_MAP: dict[str, str] = {
    "A": "Infectious Diseases",
    "B": "Infectious Diseases",
    "C": "Oncology",
    "D": "Haematology",
    "E": "Endocrinology",
    "F": "Psychiatry",
    "G": "Neurology",
    "H": "Ophthalmology / ENT",
    "I": "Cardiology",
    "J": "Respiratory Medicine",
    "K": "Gastroenterology / General Surgery",
    "L": "Dermatology",
    "M": "Rheumatology / Orthopaedics",
    "N": "Nephrology / Urology",
    "O": "Obstetrics",
    "P": "Paediatrics / Neonatology",
    "Q": "Genetics / Paediatrics",
    "R": "Emergency Medicine",
    "S": "Trauma and Orthopaedics",
    "T": "Emergency Medicine",
    "Z": "General Medicine",
}

CODE_SYSTEM = registry.CodeSystem(
    key="icd10",
    name="ICD-10",
    kind="diagnostic",
    codes=ICD10_CODES,
    specialty_field="specialty",
    type_field="admission_type",
    chapter_map=_CHAPTER_SPECIALTY_MAP,
    default_specialty="General Medicine",
)
registry.register_code_system(CODE_SYSTEM)


def lookup_code(code: str) -> dict | None:
    """Return the metadata dict for an ICD-10 code, or None if not found.

    Performs a case-insensitive lookup and also tries the uppercased code.
    """
    return registry.lookup_code(CODE_SYSTEM, code)


def get_clinical_context(code: str) -> str:
    """Return a rich text description of the ICD-10 code suitable for LLM prompts.

    Example return value:
        "ICD-10 I21.0 (Acute transmural myocardial infarction of anterior wall):
         This patient is being managed as an emergency case. Specialty: Cardiology.
         Typical length of stay: 4-7 days. ..."
    """
    return registry.get_clinical_context(CODE_SYSTEM, code)


def infer_specialty(code: str) -> str:
    """Return the most likely clinical specialty for the given ICD-10 code.

    Falls back to a chapter-based heuristic if the code is not in the dictionary.
    """
    return registry.infer_specialty(CODE_SYSTEM, code)


def parse_codes(codes_str: str) -> list[str]:
    """Parse a comma-separated string of ICD-10 codes into a list.

    Strips whitespace and filters out empty strings.

    Args:
        codes_str: Comma-separated codes, e.g. "I21.0, J18.1, K35.2"

    Returns:
        List of uppercased code strings, e.g. ["I21.0", "J18.1", "K35.2"]
    """
    return registry.parse_codes(codes_str)
