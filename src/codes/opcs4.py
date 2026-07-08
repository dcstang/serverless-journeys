"""
OPCS-4 code handling module.

OPCS-4 (Office of Population Censuses and Surveys Classification of Interventions
and Procedures, version 4) is the UK classification of surgical and interventional
procedures. Codes follow the format: letter + 2 digits + optional decimal + digit,
e.g. K40.1, W37.1, H01.1.

This module builds a src.codes.registry.CodeSystem from OPCS4_CODES and
registers it under the key 'opcs4'. The functions below are thin wrappers
around the generic registry implementation, kept for convenience and
backward compatibility - equivalent behaviour is available for any
registered code system via src.codes.registry directly.
"""

from __future__ import annotations

from src.codes import registry

# ---------------------------------------------------------------------------
# Curated OPCS-4 code dictionary
# Keys: OPCS-4 code string
# Values: dict with description, chapter_name, surgical_specialty,
#         elective_or_emergency, typical_los_days (tuple: min, max)
# ---------------------------------------------------------------------------

OPCS4_CODES: dict[str, dict] = {
    # Chapter A – Nervous System Operations
    "A07.1": {
        "description": "Primary excision of lesion of brain",
        "chapter_name": "Operations on nervous system",
        "surgical_specialty": "Neurosurgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (7, 14),
    },
    "A07.2": {
        "description": "Excision of metastasis of brain",
        "chapter_name": "Operations on nervous system",
        "surgical_specialty": "Neurosurgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (7, 14),
    },
    "A35.1": {
        "description": "Carpal tunnel decompression",
        "chapter_name": "Operations on nervous system",
        "surgical_specialty": "Neurosurgery / Plastic Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "A45.1": {
        "description": "Anterior cervical discectomy and fusion",
        "chapter_name": "Operations on nervous system",
        "surgical_specialty": "Neurosurgery / Spinal Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 4),
    },

    # Chapter B – Endocrine System Operations
    "B27.1": {
        "description": "Total thyroidectomy",
        "chapter_name": "Operations on endocrine system",
        "surgical_specialty": "Endocrine Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 3),
    },
    "B27.2": {
        "description": "Subtotal thyroidectomy",
        "chapter_name": "Operations on endocrine system",
        "surgical_specialty": "Endocrine Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 3),
    },
    "B31.1": {
        "description": "Total parathyroidectomy",
        "chapter_name": "Operations on endocrine system",
        "surgical_specialty": "Endocrine Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 3),
    },

    # Chapter E – Operations on Respiratory System
    "E20.1": {
        "description": "Lobectomy of lung",
        "chapter_name": "Operations on respiratory system",
        "surgical_specialty": "Cardiothoracic Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (5, 10),
    },
    "E20.3": {
        "description": "Pneumonectomy",
        "chapter_name": "Operations on respiratory system",
        "surgical_specialty": "Cardiothoracic Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (7, 14),
    },
    "E52.1": {
        "description": "Fibreoptic bronchoscopy and biopsy",
        "chapter_name": "Operations on respiratory system",
        "surgical_specialty": "Respiratory Medicine",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },

    # Chapter G – Operations on Upper Gastrointestinal Tract
    "G18.1": {
        "description": "Diagnostic gastroscopy",
        "chapter_name": "Operations on upper gastrointestinal tract",
        "surgical_specialty": "Gastroenterology",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "G27.1": {
        "description": "Total gastrectomy",
        "chapter_name": "Operations on upper gastrointestinal tract",
        "surgical_specialty": "Upper GI Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (7, 14),
    },
    "G27.3": {
        "description": "Partial gastrectomy",
        "chapter_name": "Operations on upper gastrointestinal tract",
        "surgical_specialty": "Upper GI Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (7, 12),
    },
    "G44.1": {
        "description": "Laparoscopic Nissen fundoplication",
        "chapter_name": "Operations on upper gastrointestinal tract",
        "surgical_specialty": "Upper GI Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 4),
    },

    # Chapter H – Operations on Lower Gastrointestinal Tract
    "H01.1": {
        "description": "Emergency excision of rectum",
        "chapter_name": "Operations on lower gastrointestinal tract",
        "surgical_specialty": "Colorectal Surgery",
        "elective_or_emergency": "emergency",
        "typical_los_days": (7, 14),
    },
    "H05.1": {
        "description": "Total colectomy and ileorectal anastomosis",
        "chapter_name": "Operations on lower gastrointestinal tract",
        "surgical_specialty": "Colorectal Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (7, 12),
    },
    "H13.1": {
        "description": "Laparoscopic right hemicolectomy",
        "chapter_name": "Operations on lower gastrointestinal tract",
        "surgical_specialty": "Colorectal Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (4, 7),
    },
    "H20.1": {
        "description": "Colonoscopy and polypectomy",
        "chapter_name": "Operations on lower gastrointestinal tract",
        "surgical_specialty": "Gastroenterology / Colorectal Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "H22.1": {
        "description": "Laparoscopic appendicectomy",
        "chapter_name": "Operations on lower gastrointestinal tract",
        "surgical_specialty": "General Surgery",
        "elective_or_emergency": "emergency",
        "typical_los_days": (2, 4),
    },
    "H41.1": {
        "description": "Primary repair of inguinal hernia",
        "chapter_name": "Operations on lower gastrointestinal tract",
        "surgical_specialty": "General Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },

    # Chapter J – Other Abdominal Operations
    "J12.1": {
        "description": "Open cholecystectomy",
        "chapter_name": "Other abdominal operations",
        "surgical_specialty": "General Surgery",
        "elective_or_emergency": "emergency",
        "typical_los_days": (4, 7),
    },
    "J18.1": {
        "description": "Laparoscopic cholecystectomy",
        "chapter_name": "Other abdominal operations",
        "surgical_specialty": "General Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (1, 2),
    },
    "J38.1": {
        "description": "Whipple's procedure (pancreaticoduodenectomy)",
        "chapter_name": "Other abdominal operations",
        "surgical_specialty": "Hepatopancreaticobiliary Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (10, 21),
    },

    # Chapter K – Operations on Heart
    "K40.1": {
        "description": "Coronary artery bypass grafting using saphenous vein graft",
        "chapter_name": "Operations on heart",
        "surgical_specialty": "Cardiac Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (7, 10),
    },
    "K46.0": {
        "description": "Percutaneous transluminal coronary angioplasty (PTCA) and stenting",
        "chapter_name": "Operations on heart",
        "surgical_specialty": "Interventional Cardiology",
        "elective_or_emergency": "emergency",
        "typical_los_days": (2, 3),
    },
    "K49.1": {
        "description": "Insertion of pacemaker",
        "chapter_name": "Operations on heart",
        "surgical_specialty": "Cardiology",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 3),
    },
    "K57.1": {
        "description": "Aortic valve replacement",
        "chapter_name": "Operations on heart",
        "surgical_specialty": "Cardiac Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (7, 10),
    },
    "K65.1": {
        "description": "Right heart catheterisation",
        "chapter_name": "Operations on heart",
        "surgical_specialty": "Cardiology",
        "elective_or_emergency": "elective",
        "typical_los_days": (1, 2),
    },

    # Chapter L – Operations on Arteries and Veins
    "L29.1": {
        "description": "Carotid endarterectomy",
        "chapter_name": "Operations on arteries and veins",
        "surgical_specialty": "Vascular Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 4),
    },
    "L19.1": {
        "description": "Open repair of abdominal aortic aneurysm",
        "chapter_name": "Operations on arteries and veins",
        "surgical_specialty": "Vascular Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (7, 12),
    },
    "L22.1": {
        "description": "Endovascular repair of abdominal aortic aneurysm (EVAR)",
        "chapter_name": "Operations on arteries and veins",
        "surgical_specialty": "Vascular Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (3, 5),
    },

    # Chapter M – Urinary System Operations
    "M34.1": {
        "description": "Percutaneous nephrolithotomy (PCNL)",
        "chapter_name": "Operations on urinary tract",
        "surgical_specialty": "Urology",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 4),
    },
    "M45.1": {
        "description": "Transurethral resection of prostate (TURP)",
        "chapter_name": "Operations on urinary tract",
        "surgical_specialty": "Urology",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 4),
    },
    "M48.1": {
        "description": "Cystoscopy and biopsy of bladder",
        "chapter_name": "Operations on urinary tract",
        "surgical_specialty": "Urology",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "M52.1": {
        "description": "Ureteroscopy and lithotripsy",
        "chapter_name": "Operations on urinary tract",
        "surgical_specialty": "Urology",
        "elective_or_emergency": "elective",
        "typical_los_days": (1, 2),
    },
    "N20.1": {
        "description": "Radical nephrectomy",
        "chapter_name": "Operations on urinary tract",
        "surgical_specialty": "Urology / Oncology",
        "elective_or_emergency": "elective",
        "typical_los_days": (3, 6),
    },

    # Chapter T – Operations on Soft Tissue
    "T19.1": {
        "description": "Primary repair of rotator cuff",
        "chapter_name": "Operations on soft tissue",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "elective",
        "typical_los_days": (1, 2),
    },

    # Chapter V – Operations on Bones
    "V38.1": {
        "description": "Open reduction and internal fixation of fracture of shaft of femur",
        "chapter_name": "Operations on bones",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "emergency",
        "typical_los_days": (5, 10),
    },
    "V40.1": {
        "description": "Intramedullary nailing of femoral shaft fracture",
        "chapter_name": "Operations on bones",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "emergency",
        "typical_los_days": (5, 10),
    },
    "V22.1": {
        "description": "Dynamic hip screw fixation for femoral neck fracture",
        "chapter_name": "Operations on bones",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "emergency",
        "typical_los_days": (7, 14),
    },
    "V55.1": {
        "description": "Open reduction and internal fixation of distal radius fracture",
        "chapter_name": "Operations on bones",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "emergency",
        "typical_los_days": (1, 2),
    },

    # Chapter W – Operations on Joints
    "W37.1": {
        "description": "Primary total hip replacement",
        "chapter_name": "Operations on joints",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "elective",
        "typical_los_days": (3, 5),
    },
    "W40.1": {
        "description": "Primary total knee replacement",
        "chapter_name": "Operations on joints",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "elective",
        "typical_los_days": (3, 5),
    },
    "W82.1": {
        "description": "Arthroscopy of knee",
        "chapter_name": "Operations on joints",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "W83.1": {
        "description": "Arthroscopic meniscectomy",
        "chapter_name": "Operations on joints",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "W78.1": {
        "description": "Revision total hip replacement",
        "chapter_name": "Operations on joints",
        "surgical_specialty": "Trauma and Orthopaedics",
        "elective_or_emergency": "elective",
        "typical_los_days": (5, 8),
    },

    # Chapter X – Miscellaneous / Diagnostic procedures
    "Y50.2": {
        "description": "Insertion of prosthesis NEC",
        "chapter_name": "Miscellaneous operations",
        "surgical_specialty": "Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (1, 3),
    },
    "X40.1": {
        "description": "Magnetic resonance imaging of head",
        "chapter_name": "Diagnostic imaging procedures",
        "surgical_specialty": "Radiology",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "X42.1": {
        "description": "CT scan of thorax",
        "chapter_name": "Diagnostic imaging procedures",
        "surgical_specialty": "Radiology",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },

    # Additional common procedures
    "O01.1": {
        "description": "Lower segment caesarean section",
        "chapter_name": "Operations on gravid uterus",
        "surgical_specialty": "Obstetrics",
        "elective_or_emergency": "emergency",
        "typical_los_days": (3, 4),
    },
    "S60.1": {
        "description": "Mastectomy",
        "chapter_name": "Operations on breast",
        "surgical_specialty": "Breast Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (2, 3),
    },
    "S63.1": {
        "description": "Wide local excision of breast",
        "chapter_name": "Operations on breast",
        "surgical_specialty": "Breast Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (1, 2),
    },
    "C75.1": {
        "description": "Trabeculectomy for glaucoma",
        "chapter_name": "Operations on eye",
        "surgical_specialty": "Ophthalmology",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "C71.1": {
        "description": "Extracapsular extraction of lens with insertion of prosthetic lens (cataract surgery)",
        "chapter_name": "Operations on eye",
        "surgical_specialty": "Ophthalmology",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "F34.1": {
        "description": "Extraction of tooth",
        "chapter_name": "Operations on mouth",
        "surgical_specialty": "Oral and Maxillofacial Surgery / Dentistry",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "D15.1": {
        "description": "Myringoplasty",
        "chapter_name": "Operations on ear",
        "surgical_specialty": "Ear, Nose and Throat Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (0, 1),
    },
    "D16.1": {
        "description": "Mastoidectomy",
        "chapter_name": "Operations on ear",
        "surgical_specialty": "Ear, Nose and Throat Surgery",
        "elective_or_emergency": "elective",
        "typical_los_days": (1, 2),
    },
}


# ---------------------------------------------------------------------------
# Chapter-letter fallback surgical specialty map, used for codes outside
# OPCS4_CODES
# ---------------------------------------------------------------------------
_CHAPTER_SPECIALTY_MAP: dict[str, str] = {
    "A": "Neurosurgery",
    "B": "Endocrine Surgery",
    "C": "Ophthalmology",
    "D": "Ear, Nose and Throat Surgery",
    "E": "Cardiothoracic Surgery",
    "F": "Oral and Maxillofacial Surgery",
    "G": "Upper GI Surgery",
    "H": "Colorectal Surgery",
    "J": "General Surgery",
    "K": "Cardiac Surgery",
    "L": "Vascular Surgery",
    "M": "Urology",
    "N": "Urology",
    "O": "Obstetrics",
    "P": "Urology",
    "Q": "Gynaecology",
    "S": "Breast Surgery",
    "T": "Plastic Surgery",
    "V": "Trauma and Orthopaedics",
    "W": "Trauma and Orthopaedics",
    "X": "Radiology",
    "Y": "Surgery",
    "Z": "Surgery",
}

CODE_SYSTEM = registry.CodeSystem(
    key="opcs4",
    name="OPCS-4",
    kind="procedure",
    codes=OPCS4_CODES,
    specialty_field="surgical_specialty",
    type_field="elective_or_emergency",
    chapter_map=_CHAPTER_SPECIALTY_MAP,
    default_specialty="General Surgery",
)
registry.register_code_system(CODE_SYSTEM)


def lookup_code(code: str) -> dict | None:
    """Return the metadata dict for a given OPCS-4 code, or None if not found.

    Performs a case-insensitive lookup.

    Args:
        code: OPCS-4 code string e.g. "K40.1" or "k40.1".

    Returns:
        Dictionary with keys description, chapter_name, surgical_specialty,
        elective_or_emergency, typical_los_days, or None.
    """
    return registry.lookup_code(CODE_SYSTEM, code)


def get_clinical_context(code: str) -> str:
    """Return a rich text description suitable for inclusion in an LLM prompt.

    Args:
        code: OPCS-4 code string.

    Returns:
        A formatted string describing the surgical/procedural context, or a
        generic message if the code is not found.
    """
    return registry.get_clinical_context(CODE_SYSTEM, code)


def infer_specialty(code: str) -> str:
    """Return the most likely surgical specialty for a given OPCS-4 code.

    Falls back to a chapter-based heuristic if not in the curated dictionary.

    Args:
        code: OPCS-4 code string.

    Returns:
        Surgical specialty string.
    """
    return registry.infer_specialty(CODE_SYSTEM, code)


def parse_codes(codes_str: str) -> list[str]:
    """Parse a comma-separated string of OPCS-4 codes into a list.

    Handles whitespace, mixed case, and empty strings gracefully.

    Args:
        codes_str: Comma-separated OPCS-4 codes e.g. "K40.1, W37.1, H01.1".

    Returns:
        List of normalised (upper-case, stripped) code strings.
    """
    return registry.parse_codes(codes_str)
