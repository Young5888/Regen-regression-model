# Duration Regression Model

## Overview
This project builds a regression model to predict clinical trial duration (in days) using structured data from ClinicalTrials.gov.

The model is trained only on **completed, industry-sponsored interventional trials**, and uses engineered features capturing study design, eligibility criteria, site footprint, and outcome structure.

To account for structural differences across trial phases, we train **separate models for each phase group**:
- PHASE1
- PHASE1/PHASE2
- PHASE2
- PHASE2/PHASE3
- PHASE3

---

## Model Choice
We use `HistGradientBoostingRegressor` with:
- `max_iter = 200`
- `random_state = 42`

Additionally, the model is wrapped in `TransformedTargetRegressor`:
- Target is transformed using `log1p`
- Predictions are inverse-transformed using `expm1`

### Why this approach?
- Captures **non-linear relationships** in structured tabular data  
- Handles **missing values (NaN)** natively  
- Does not require feature scaling  
- Log-transform stabilizes variance and improves regression performance  

---

## Target
- `duration_days` = time from study start to primary completion  

Preprocessing ensures:
- Only **completed trials**
- Only **14 ≤ duration_days ≤ 3650**
- Removes implausible short and extreme-duration trials  

---

## Data Filtering
The dataset is filtered to improve reliability:

- Only **INTERVENTIONAL** studies  
- Only trials with **INDUSTRY sponsors**  
- Excludes **WITHDRAWN** studies  
- Valid date range: **1980–2027**  
- Removes trials with missing or invalid dates  

---

## Features

### Core features
- `phase`
- `enrollment`
- `n_sponsors`
- `number_of_arms`
- `start_year`
- `category` (one-hot encoded)
- `downcase_mesh_term` (top 50 encoded)
- `intervention_type` (top 15 encoded)

---

### Eligibility features
- `gender`, `minimum_age`, `maximum_age`
- `adult`, `child`, `older_adult`

### Eligibility text features (derived from criteria)
- `eligibility_criteria_char_len`
- `eligibility_n_inclusion_tildes`
- `eligibility_n_exclusion_tildes`
- `eligibility_has_burden_procedure`

These capture **complexity of eligibility criteria** :contentReference[oaicite:3]{index=3}  

---

### Site footprint
- `number_of_facilities`
- `number_of_countries`
- `us_only`
- `has_single_facility`
- `facility_density`

---

### Design features
- `randomized`
- `intervention_model`
- `primary_purpose`
- `masking_depth_score`
- `design_complexity_composite`

Composite features combine multiple signals (e.g., masking + arms)

---

### Arm / intervention complexity
- `number_of_interventions`
- `intervention_type_diversity`
- `mono_therapy`
- `has_placebo`
- `has_active_comparator`
- `n_mesh_intervention_terms`

---

### Outcome complexity
- `max_planned_followup_days`
- `n_primary_outcomes`, `n_secondary_outcomes`
- `has_survival_endpoint`
- `has_safety_endpoint`
- `endpoint_complexity_score`

Derived from outcome descriptions and time frames

---

## Training
- One model per phase group  
- No `StandardScaler` (tree-based model)  
- Missing values preserved as NaN  

### Data split
- 60% training  
- 20% validation  
- 20% test  
- `random_state = 42`

---

## Metrics
Evaluation is performed using:

- RMSE (Root Mean Squared Error)  
- MAE (Mean Absolute Error)  
- R² (coefficient of determination)  

Metrics are reported **per phase**

---

## Assumptions
- Completed trials reflect true realized durations  
- Log transformation improves model stability  
- Phase-specific models capture structural differences  

---

## Limitations
- Excludes ongoing and terminated trials → survivorship bias  
- Does not capture external delays (regulatory, funding)  
- High-cardinality features may introduce sparsity  
- Performance varies by phase depending on data size  

