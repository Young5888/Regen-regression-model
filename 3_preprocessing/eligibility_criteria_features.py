"""
Parse eligibility `criteria` text into numeric features (ClinicalTrials.gov-style).

Used by preprocess (materialize columns on clean studies) and exploration.
"""
from __future__ import annotations

import pandas as pd

CRITERIA_TEXT_FEATURE_COLUMNS = [
    "eligibility_criteria_char_len",
    "eligibility_n_inclusion_tildes",
    "eligibility_n_exclusion_tildes",
    "eligibility_has_burden_procedure",
]

BURDEN_KEYWORDS = [
    "biopsy",
    "mri",
    "ecg",
    "ekg",
    "washout",
    "endoscopy",
    "colonoscopy",
    "bronchoscopy",
    "lumbar puncture",
    "spinal tap",
    "pet scan",
    "ct scan",
    "cardiac catheterization",
]


def count_inclusion_tildes(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    if "Inclusion Criteria:" not in text:
        return 0
    try:
        before_excl = text.split("Exclusion Criteria:")[0]
        inclusion_part = before_excl.split("Inclusion Criteria:")[1]
        return inclusion_part.count("~")
    except (IndexError, ValueError):
        return 0


def count_exclusion_tildes(text: str) -> int:
    if not isinstance(text, str) or not text.strip():
        return 0
    if "Exclusion Criteria:" not in text:
        return 0
    exclusion_part = text.split("Exclusion Criteria:")[1]
    return exclusion_part.count("~")


def has_burden_keyword(text: str) -> bool:
    if not isinstance(text, str) or not text.strip():
        return False
    t = text.lower()
    return any(kw in t for kw in BURDEN_KEYWORDS)


def compute_criteria_features_for_eligibilities(elig: pd.DataFrame) -> pd.DataFrame:
    """
    Input: columns nct_id, criteria (one row per nct_id recommended).
    Output: nct_id + CRITERIA_TEXT_FEATURE_COLUMNS (int).
    """
    if "nct_id" not in elig.columns or "criteria" not in elig.columns:
        raise ValueError("elig must contain nct_id and criteria")
    t = elig["criteria"].fillna("").astype(str)
    out = pd.DataFrame(
        {
            "nct_id": elig["nct_id"].values,
            "eligibility_criteria_char_len": t.str.len().astype("int64"),
            "eligibility_n_inclusion_tildes": t.map(count_inclusion_tildes).astype("int64"),
            "eligibility_n_exclusion_tildes": t.map(count_exclusion_tildes).astype("int64"),
            "eligibility_has_burden_procedure": t.map(lambda x: int(has_burden_keyword(x))).astype("int64"),
        }
    )
    return out
