"""
Join studies with sponsors, select features, train/val/test split,
and run regression to predict duration_days.

Restricted to COMPLETED trials only (actual duration, not estimated).
"""
import re
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CLEAN_DATA = PROJECT_ROOT / "clean_data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Features: target_duration (97% null) and number_of_groups (100% null) excluded
# category (132 unique) outperforms therapeutic_area (16 unique): R² 0.317 vs 0.285
# Phase: one-hot (R² 0.317) > phase flags (0.315) > no phase (0.280)
# downcase_mesh_term: ablation R² 0.319 vs 0.317 baseline — small gain, included
# intervention_type: ablation R² 0.320 vs 0.319 baseline — included
# eligibility: gender, minimum_age, maximum_age, adult, child, older_adult (ablation-tested)
FEATURE_COLUMNS = [
    "phase",
    "enrollment",
    "n_sponsors",
    "number_of_arms",
    "start_year",
    "category",
    "downcase_mesh_term",
    "intervention_type",
]
ELIGIBILITY_COLUMNS = ["gender", "minimum_age", "maximum_age", "adult", "child", "older_adult"]
SITE_FOOTPRINT_FEATURES = [
    "number_of_facilities",
    "number_of_countries",
    "us_only",
    "number_of_us_states",
    "has_single_facility",
    "facility_density",
]
DESIGN_FEATURES = [
    "randomized",
    "intervention_model",
    "masking_depth_score",
    "primary_purpose",
    "design_complexity_composite",
]
ARM_INTERVENTION_FEATURES = [
    "number_of_interventions",
    "intervention_type_diversity",
    "mono_therapy",
    "has_placebo",
    "has_active_comparator",
    "n_mesh_intervention_terms",
]
TARGET_COLUMN = "duration_days"

# Best-performing columns from ablation studies (see MODEL.md)
KEPT_ELIGIBILITY = ["gender", "minimum_age", "maximum_age", "adult", "child", "older_adult"]
KEPT_SITE_FOOTPRINT = ["number_of_facilities", "number_of_countries", "us_only", "has_single_facility"]
KEPT_DESIGN = ["randomized", "intervention_model", "masking_depth_score", "primary_purpose", "design_complexity_composite"]
KEPT_ARM_INTERVENTION = [
    "number_of_interventions",
    "intervention_type_diversity",
    "mono_therapy",
    "has_placebo",
    "has_active_comparator",
    "n_mesh_intervention_terms",
]
DESIGN_OUTCOMES_FEATURES = [
    "max_planned_followup_days",
    "n_primary_outcomes",
    "n_secondary_outcomes",
    "n_outcomes",
    "has_survival_endpoint",
    "has_safety_endpoint",
    "endpoint_complexity_score",
]
KEPT_DESIGN_OUTCOMES = [
    "max_planned_followup_days",
    "n_primary_outcomes",
    "n_secondary_outcomes",
    "n_outcomes",
    "has_survival_endpoint",
    "has_safety_endpoint",
    "endpoint_complexity_score",
]

RAW_DATA = PROJECT_ROOT / "raw_data"


def _parse_time_frame_days(tf: str) -> float | None:
    """Parse time_frame string to days. Returns None if unparseable."""
    if pd.isna(tf) or not isinstance(tf, str) or not tf.strip():
        return None
    tf = tf.strip().lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(day|week|month|year)s?", tf)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    if unit == "day":
        return val
    if unit == "week":
        return val * 7
    if unit == "month":
        return val * 30.44
    if unit == "year":
        return val * 365.25
    return None


def _has_endpoint_keywords(text: str, keywords: list[str]) -> bool:
    if pd.isna(text) or not isinstance(text, str):
        return False
    t = text.lower()
    return any(k in t for k in keywords)


def load_and_join(
    eligibility_columns: list[str] | None = None,
    site_footprint_columns: list[str] | None = None,
    design_columns: list[str] | None = None,
    arm_intervention_columns: list[str] | None = None,
    design_outcomes_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Load clean studies, sponsors, and categorized_output; join on nct_id.
    If eligibility_columns is provided, join eligibilities table for those columns.
    """
    studies = pd.read_csv(CLEAN_DATA / "studies.csv", low_memory=False)
    sponsors = pd.read_csv(CLEAN_DATA / "sponsors.csv", low_memory=False)

    # Restrict to COMPLETED trials only (actual duration)
    studies = studies[studies["overall_status"] == "COMPLETED"].copy()

    # Aggregate sponsors: count per nct_id
    sponsor_counts = sponsors.groupby("nct_id").size().reset_index(name="n_sponsors")
    df = studies.merge(sponsor_counts, on="nct_id", how="left")
    df["n_sponsors"] = df["n_sponsors"].fillna(0).astype(int)

    # Join category from categorized_output (take highest-confidence per trial)
    categorized = pd.read_csv(RAW_DATA / "categorized_output.csv", low_memory=False)
    cat_agg = (
        categorized.sort_values("confidence", ascending=False)
        .groupby("nct_id")[["category"]]
        .first()
        .reset_index()
    )
    df = df.merge(cat_agg, on="nct_id", how="left")
    df["category"] = df["category"].fillna("Other_Unclassified")

    # Join downcase_mesh_term from browse_conditions (first per trial)
    bc_path = RAW_DATA / "browse_conditions.csv"
    if bc_path.exists():
        bc = pd.read_csv(bc_path, low_memory=False)
        mesh_col = "downcase_mesh_term" if "downcase_mesh_term" in bc.columns else "mesh_term"
        if mesh_col in bc.columns:
            mesh_agg = bc.groupby("nct_id")[mesh_col].first().reset_index()
            mesh_agg.columns = ["nct_id", "downcase_mesh_term"]
            df = df.merge(mesh_agg, on="nct_id", how="left")
            df["downcase_mesh_term"] = df["downcase_mesh_term"].fillna("unknown")

    # Join intervention_type from interventions (mode per trial)
    int_path = RAW_DATA / "interventions.csv"
    if int_path.exists():
        interventions = pd.read_csv(int_path, low_memory=False)
        if "intervention_type" in interventions.columns:
            int_agg = (
                interventions.groupby("nct_id")["intervention_type"]
                .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
                .reset_index()
            )
            df = df.merge(int_agg, on="nct_id", how="left")
            df["intervention_type"] = df["intervention_type"].fillna("UNKNOWN")

    # Join eligibilities (first row per nct_id)
    elig_path = RAW_DATA / "eligibilities.csv"
    if elig_path.exists() and eligibility_columns:
        elig = pd.read_csv(elig_path, low_memory=False)
        cols_to_join = ["nct_id"] + [c for c in eligibility_columns if c in elig.columns]
        if len(cols_to_join) > 1:
            elig_agg = elig[cols_to_join].groupby("nct_id").first().reset_index()
            df = df.merge(elig_agg, on="nct_id", how="left")

    # Join site footprint (calculated_values, facilities, countries)
    if site_footprint_columns:
        cv_path = RAW_DATA / "calculated_values.csv"
        if cv_path.exists():
            cv = pd.read_csv(cv_path, low_memory=False)
            cv_cols = ["nct_id", "number_of_facilities", "has_us_facility", "has_single_facility"]
            cv_cols = [c for c in cv_cols if c in cv.columns]
            cv_agg = cv[cv_cols].groupby("nct_id").first().reset_index()
            df = df.merge(cv_agg, on="nct_id", how="left")

        countries_path = RAW_DATA / "countries.csv"
        if countries_path.exists():
            countries = pd.read_csv(countries_path, low_memory=False)
            # Exclude removed countries (removed=True means no longer associated)
            if "removed" in countries.columns:
                countries_active = countries[~countries["removed"].fillna(False).astype(bool)]
            else:
                countries_active = countries
            n_countries = countries_active.groupby("nct_id").size().reset_index(name="number_of_countries")
            df = df.merge(n_countries, on="nct_id", how="left")
            # US-only: 1 if exactly 1 country and it's US
            if "name" in countries.columns:
                us_only = (
                    countries_active.groupby("nct_id")["name"]
                    .apply(lambda x: 1 if (len(x) == 1 and "united states" in str(x.iloc[0]).lower()) else 0)
                    .reset_index(name="us_only")
                )
                df = df.merge(us_only, on="nct_id", how="left")

        fac_path = RAW_DATA / "facilities.csv"
        if fac_path.exists() and "number_of_us_states" in site_footprint_columns:
            fac = pd.read_csv(fac_path, low_memory=False)
            us_fac = fac[fac["country"].str.upper().str.contains("UNITED STATES", na=False)]
            n_us_states = us_fac.groupby("nct_id")["state"].nunique().reset_index(name="number_of_us_states")
            df = df.merge(n_us_states, on="nct_id", how="left")

        # Derived: facility_density = number_of_facilities / enrollment
        if "facility_density" in site_footprint_columns and "number_of_facilities" in df.columns and "enrollment" in df.columns:
            enroll = pd.to_numeric(df["enrollment"], errors="coerce").fillna(1)
            df["facility_density"] = df["number_of_facilities"].fillna(0) / enroll.replace(0, 1)

    # Join designs (one row per nct_id)
    if design_columns:
        designs_path = RAW_DATA / "designs.csv"
        if designs_path.exists():
            designs = pd.read_csv(designs_path, low_memory=False)
            design_cols = ["nct_id", "allocation", "intervention_model", "primary_purpose", "masking",
                          "subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked"]
            design_cols = [c for c in design_cols if c in designs.columns]
            design_agg = designs[design_cols].groupby("nct_id").first().reset_index()
            df = df.merge(design_agg, on="nct_id", how="left")

            # Derived: randomized (1 if RANDOMIZED)
            if "randomized" in design_columns and "allocation" in df.columns:
                df["randomized"] = (df["allocation"].str.upper() == "RANDOMIZED").astype(int)

            # Derived: masking_depth_score (NONE=0, SINGLE=1, DOUBLE=2, TRIPLE=3, QUADRUPLE=4)
            if "masking_depth_score" in design_columns and "masking" in df.columns:
                mask_map = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
                df["masking_depth_score"] = df["masking"].str.upper().map(mask_map).fillna(0)
                # Add role flags: +0.25 per masked role (max 1 extra)
                for role in ["subject_masked", "caregiver_masked", "investigator_masked", "outcomes_assessor_masked"]:
                    if role in df.columns:
                        df["masking_depth_score"] += df[role].apply(
                            lambda x: 0.25 if x in (True, "true", "True", 1) else 0
                        )

            # Derived: design_complexity_composite (randomized + multi-arm + normalized masking)
            if "design_complexity_composite" in design_columns:
                r = (df["allocation"].str.upper() == "RANDOMIZED").astype(int) if "allocation" in df.columns else 0
                m = df["masking_depth_score"].fillna(0) if "masking_depth_score" in df.columns else 0
                if "masking_depth_score" not in df.columns and "masking" in df.columns:
                    mask_map = {"NONE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "QUADRUPLE": 4}
                    m = df["masking"].str.upper().map(mask_map).fillna(0)
                arms = pd.to_numeric(df["number_of_arms"], errors="coerce").fillna(1)
                multi = (arms > 1).astype(int)
                df["design_complexity_composite"] = r + multi + (m / 5)

    # Join arm/intervention complexity (interventions, design_groups, browse_interventions)
    if arm_intervention_columns:
        int_path = RAW_DATA / "interventions.csv"
        if int_path.exists():
            interventions = pd.read_csv(int_path, low_memory=False)
            if "intervention_type" in interventions.columns:
                n_int = interventions.groupby("nct_id").size().reset_index(name="number_of_interventions")
                df = df.merge(n_int, on="nct_id", how="left")
                n_types = interventions.groupby("nct_id")["intervention_type"].nunique().reset_index(name="intervention_type_diversity")
                df = df.merge(n_types, on="nct_id", how="left")
                df["mono_therapy"] = (df["intervention_type_diversity"].fillna(0) == 1).astype(int)

        dg_path = RAW_DATA / "design_groups.csv"
        if dg_path.exists():
            dg = pd.read_csv(dg_path, low_memory=False)
            if "group_type" in dg.columns:
                dg["_gt"] = dg["group_type"].fillna("").astype(str).str.upper()
                dg["_title"] = dg.get("title", pd.Series([""] * len(dg))).fillna("").astype(str).str.upper()
                dg["_combined"] = dg["_gt"] + " " + dg["_title"]
                has_placebo = dg.groupby("nct_id")["_combined"].apply(lambda x: 1 if x.str.contains("PLACEBO", na=False).any() else 0).reset_index(name="has_placebo")
                df = df.merge(has_placebo, on="nct_id", how="left")
                has_ac = dg.groupby("nct_id")["_combined"].apply(lambda x: 1 if x.str.contains("ACTIVE.COMPARATOR|ACTIVE_COMPARATOR|COMPARATOR", na=False, regex=True).any() else 0).reset_index(name="has_active_comparator")
                df = df.merge(has_ac, on="nct_id", how="left")

        bi_path = RAW_DATA / "browse_interventions.csv"
        if bi_path.exists():
            bi = pd.read_csv(bi_path, low_memory=False)
            mesh_col = "downcase_mesh_term" if "downcase_mesh_term" in bi.columns else "mesh_term"
            if mesh_col in bi.columns:
                n_mesh = bi.groupby("nct_id")[mesh_col].nunique().reset_index(name="n_mesh_intervention_terms")
                df = df.merge(n_mesh, on="nct_id", how="left")

    # Join design_outcomes (per-trial aggregates)
    if design_outcomes_columns:
        do_path = RAW_DATA / "design_outcomes.csv"
        if do_path.exists():
            nct_ids = set(df["nct_id"].unique())
            usecols = ["nct_id", "outcome_type", "measure", "time_frame"]
            if "description" in pd.read_csv(do_path, nrows=0).columns:
                usecols.append("description")
            chunks = []
            for chunk in pd.read_csv(do_path, chunksize=200_000, low_memory=False, usecols=usecols):
                chunk = chunk[chunk["nct_id"].isin(nct_ids)]
                if len(chunk) > 0:
                    chunks.append(chunk)
            do = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=usecols + ["_tf_days"])
            if len(do) == 0:
                do = None
            else:
                do["_tf_days"] = do["time_frame"].apply(_parse_time_frame_days)
                SURVIVAL_KW = ["survival", "os", "pfs", "dfs", "overall survival", "progression-free survival"]
                SAFETY_KW = ["safety", "adverse", "ae", "sae", "toxicity", "tolerability"]
                meas = do["measure"] if "measure" in do.columns else pd.Series([""] * len(do))
                desc = do["description"] if "description" in do.columns else pd.Series([""] * len(do))
                do["_has_survival"] = meas.apply(lambda x: _has_endpoint_keywords(x, SURVIVAL_KW)) | desc.apply(lambda x: _has_endpoint_keywords(x, SURVIVAL_KW))
                do["_has_safety"] = meas.apply(lambda x: _has_endpoint_keywords(x, SAFETY_KW)) | desc.apply(lambda x: _has_endpoint_keywords(x, SAFETY_KW))
                n_outcomes = do.groupby("nct_id").size().reset_index(name="n_outcomes")
                max_tf = do.groupby("nct_id")["_tf_days"].max().reset_index(name="max_planned_followup_days")
                agg = n_outcomes.merge(max_tf, on="nct_id", how="left")
                if "outcome_type" in do.columns:
                    n_prim = do.groupby("nct_id")["outcome_type"].apply(lambda x: (x.fillna("").str.upper() == "PRIMARY").sum()).reset_index(name="n_primary_outcomes")
                    n_sec = do.groupby("nct_id")["outcome_type"].apply(lambda x: (x.fillna("").str.upper() == "SECONDARY").sum()).reset_index(name="n_secondary_outcomes")
                    agg = agg.merge(n_prim, on="nct_id", how="left").merge(n_sec, on="nct_id", how="left")
                else:
                    agg["n_primary_outcomes"] = 0
                    agg["n_secondary_outcomes"] = 0
                has_surv = do.groupby("nct_id")["_has_survival"].max().reset_index(name="has_survival_endpoint")
                has_safe = do.groupby("nct_id")["_has_safety"].max().reset_index(name="has_safety_endpoint")
                agg = agg.merge(has_surv, on="nct_id", how="left").merge(has_safe, on="nct_id", how="left")
                agg["endpoint_complexity_score"] = (
                    agg["n_outcomes"].fillna(0) * 0.5
                    + agg["n_primary_outcomes"].fillna(0) * 0.3
                    + agg["n_secondary_outcomes"].fillna(0) * 0.2
                    + agg["has_survival_endpoint"].fillna(0) * 2
                    + agg["has_safety_endpoint"].fillna(0) * 1
                )
                df = df.merge(agg, on="nct_id", how="left")

    return df


def prepare_features(
    df: pd.DataFrame,
    eligibility_columns: list[str] | None = None,
    site_footprint_columns: list[str] | None = None,
    design_columns: list[str] | None = None,
    arm_intervention_columns: list[str] | None = None,
    design_outcomes_columns: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Prepare X and y. Handle missing values and encode categoricals.
    If eligibility_columns is provided, include those eligibility features.
    Returns X, y, and a dict with encoder/scaler for inference.
    """
    df = df.copy()

    # Extract start_year from start_date
    df["start_year"] = pd.to_datetime(df["start_date"], errors="coerce").dt.year

    # Numeric: enrollment, number_of_arms (4.7% null), start_year
    for col in ["enrollment", "number_of_arms", "start_year"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            fill_val = median_val if pd.notna(median_val) else (2015 if col == "start_year" else 0)
            df[col] = df[col].fillna(fill_val)
        else:
            df[col] = 0

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].values

    # Encode phase (one-hot; ablation: one-hot > phase flags > no phase)
    phase_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    phase_encoded = phase_encoder.fit_transform(df[["phase"]])

    # Encode category (one-hot)
    cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_encoded = cat_encoder.fit_transform(df[["category"]])

    # Encode downcase_mesh_term (one-hot, top 50 to limit features)
    mesh_parts = []
    mesh_encoder = None
    if "downcase_mesh_term" in df.columns:
        top_mesh = df["downcase_mesh_term"].value_counts().head(50).index.tolist()
        df["mesh_trimmed"] = df["downcase_mesh_term"].where(
            df["downcase_mesh_term"].isin(top_mesh), "other"
        )
        mesh_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        mesh_parts = [mesh_encoder.fit_transform(df[["mesh_trimmed"]])]

    # Encode intervention_type (one-hot, top 15 by count — same pattern as mesh_term)
    int_parts = []
    int_encoder = None
    if "intervention_type" in df.columns:
        top_int = df["intervention_type"].value_counts().head(15).index.tolist()
        df["intervention_trimmed"] = df["intervention_type"].where(
            df["intervention_type"].isin(top_int), "other"
        )
        int_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        int_parts = [int_encoder.fit_transform(df[["intervention_trimmed"]])]

    # Eligibility features
    elig_parts = []
    elig_encoders = {}
    elig_feature_names = []
    if eligibility_columns:
        for col in eligibility_columns:
            if col not in df.columns:
                continue
            if col == "gender":
                df["gender_fill"] = df["gender"].fillna("ALL").astype(str)
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                elig_parts.append(enc.fit_transform(df[["gender_fill"]]))
                elig_encoders["gender"] = enc
                elig_feature_names.extend(enc.get_feature_names_out(["gender_fill"]))
            elif col in ("minimum_age", "maximum_age"):
                # Parse "18 Years", "65 Years" etc. to numeric
                raw = df[col].astype(str).str.extract(r"(\d+)", expand=False)
                vals = pd.to_numeric(raw, errors="coerce")
                median_val = vals.median()
                if pd.isna(median_val):
                    median_val = 18.0 if col == "minimum_age" else 65.0
                elig_parts.append(np.column_stack([vals.fillna(median_val).values]))
                elig_feature_names.append(col)
            elif col in ("adult", "child", "older_adult"):
                vals = df[col].fillna(False)
                if vals.dtype == bool or vals.dtype == "object":
                    vals = vals.map(lambda x: 1 if x in (True, "true", "True", "YES", "Yes", 1) else 0)
                elig_parts.append(np.column_stack([vals.values.astype(float)]))
                elig_feature_names.append(col)

    # Site footprint features
    site_parts = []
    site_feature_names = []
    if site_footprint_columns:
        for col in site_footprint_columns:
            if col not in df.columns:
                continue
            if col == "has_single_facility":
                vals = df[col].apply(lambda x: 1 if x in (True, "true", "True", "YES", "Yes", 1) else 0)
                site_parts.append(np.column_stack([vals.values.astype(float)]))
                site_feature_names.append(col)
            elif col in ("number_of_facilities", "number_of_countries", "number_of_us_states"):
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                site_parts.append(np.column_stack([vals.values]))
                site_feature_names.append(col)
            elif col == "us_only":
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                site_parts.append(np.column_stack([vals.values]))
                site_feature_names.append(col)
            elif col == "facility_density":
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                site_parts.append(np.column_stack([vals.values]))
                site_feature_names.append(col)

    # Design features
    design_parts = []
    design_feature_names = []
    if design_columns:
        for col in design_columns:
            if col not in df.columns:
                continue
            if col == "randomized":
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                design_parts.append(np.column_stack([vals.values]))
                design_feature_names.append(col)
            elif col == "intervention_model":
                df["intervention_model_fill"] = df["intervention_model"].fillna("UNKNOWN").astype(str)
                top_mod = df["intervention_model_fill"].value_counts().head(6).index.tolist()
                df["intervention_model_trimmed"] = df["intervention_model_fill"].where(
                    df["intervention_model_fill"].isin(top_mod), "other"
                )
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                design_parts.append(enc.fit_transform(df[["intervention_model_trimmed"]]))
                design_feature_names.extend(enc.get_feature_names_out(["intervention_model_trimmed"]))
            elif col == "primary_purpose":
                df["primary_purpose_fill"] = df["primary_purpose"].fillna("OTHER").astype(str)
                top_pp = df["primary_purpose_fill"].value_counts().head(6).index.tolist()
                df["primary_purpose_trimmed"] = df["primary_purpose_fill"].where(
                    df["primary_purpose_fill"].isin(top_pp), "other"
                )
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                design_parts.append(enc.fit_transform(df[["primary_purpose_trimmed"]]))
                design_feature_names.extend(enc.get_feature_names_out(["primary_purpose_trimmed"]))
            elif col in ("masking_depth_score", "design_complexity_composite"):
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                design_parts.append(np.column_stack([vals.values]))
                design_feature_names.append(col)

    # Design outcomes features
    do_parts = []
    do_feature_names = []
    if design_outcomes_columns:
        for col in design_outcomes_columns:
            if col not in df.columns:
                continue
            if col in ("has_survival_endpoint", "has_safety_endpoint"):
                vals = df[col].fillna(0)
                if vals.dtype == bool or vals.dtype == "object":
                    vals = vals.apply(lambda x: 1 if x in (True, "true", "True", 1) else 0)
                do_parts.append(np.column_stack([vals.values.astype(float)]))
                do_feature_names.append(col)
            else:
                vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
                do_parts.append(np.column_stack([vals.values]))
                do_feature_names.append(col)

    # Arm/intervention features
    arm_parts = []
    arm_feature_names = []
    if arm_intervention_columns:
        for col in arm_intervention_columns:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
            arm_parts.append(np.column_stack([vals.values]))
            arm_feature_names.append(col)

    # Numeric features
    X_numeric = np.column_stack([
        df["enrollment"].values,
        df["n_sponsors"].values,
        df["number_of_arms"].values,
        df["start_year"].values,
    ])
    X = np.hstack([phase_encoded, cat_encoded] + mesh_parts + int_parts + elig_parts + site_parts + design_parts + do_parts + arm_parts + [X_numeric])

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    artifacts = {
        "phase_encoder": phase_encoder,
        "cat_encoder": cat_encoder,
        "mesh_encoder": mesh_encoder,
        "int_encoder": int_encoder,
        "elig_encoders": elig_encoders,
        "elig_feature_names": elig_feature_names,
        "site_feature_names": site_feature_names,
        "design_feature_names": design_feature_names,
        "do_feature_names": do_feature_names,
        "arm_feature_names": arm_feature_names,
        "scaler": scaler,
    }
    return X, y, artifacts


def main() -> None:
    df = load_and_join(
        eligibility_columns=KEPT_ELIGIBILITY,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
    )
    X, y, artifacts = prepare_features(
        df,
        eligibility_columns=KEPT_ELIGIBILITY,
        site_footprint_columns=KEPT_SITE_FOOTPRINT,
        design_columns=KEPT_DESIGN,
        arm_intervention_columns=KEPT_ARM_INTERVENTION,
        design_outcomes_columns=KEPT_DESIGN_OUTCOMES,
    )

    # Train 60% / Val 20% / Test 20%
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # Train Ridge regression (regularized, robust to multicollinearity)
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate on train, val, test
    def eval_set(name: str, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = model.predict(X)
        return {
            "set": name,
            "rmse": np.sqrt(mean_squared_error(y, y_pred)),
            "mae": mean_absolute_error(y, y_pred),
            "r2": r2_score(y, y_pred),
        }

    results = [
        eval_set("train", X_train, y_train),
        eval_set("val", X_val, y_val),
        eval_set("test", X_test, y_test),
    ]

    # Report: metrics only
    lines = []
    lines.append("-" * 40)
    lines.append("METRICS")
    lines.append("-" * 40)
    for r in results:
        lines.append(f"  {r['set']:5}: RMSE={r['rmse']:,.0f} days  MAE={r['mae']:,.0f} days  R²={r['r2']:.4f}")
    lines.append("-" * 40)

    report = "\n".join(lines)
    print(report)
    (RESULTS_DIR / "regression_report.txt").write_text(report)


if __name__ == "__main__":
    main()
