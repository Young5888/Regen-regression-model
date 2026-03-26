# Duration Regression Model

`sklearn.ensemble.HistGradientBoostingRegressor` predicting trial duration (days) for **COMPLETED** trials only. Preprocessing (encoding, scaling) is unchanged from the prior Ridge pipeline; the booster captures non-linear effects and interactions in the feature space.

## Target
- `duration_days` — time from start to primary completion

## Features (ablation-tested, best-performing subset)

### Core features (always included)
- `phase` — trial phase (one-hot)
- `enrollment` — planned enrollment
- `n_sponsors` — number of sponsors
- `number_of_arms` — number of arms
- `start_year` — trial start year
- `category` — therapeutic category (one-hot, 132 levels)
- `downcase_mesh_term` — MeSH condition terms (one-hot)
- `intervention_type` — intervention types (one-hot)

### Eligibility (kept from ablation)
- `gender`, `minimum_age`, `maximum_age`, `adult`, `child`, `older_adult`

### Site footprint (kept from ablation)
- `number_of_facilities`, `number_of_countries`, `us_only`, `has_single_facility`

### Design (kept from ablation)
- `randomized`, `intervention_model`, `masking_depth_score`, `primary_purpose`, `design_complexity_composite`

### Arm/intervention (kept from ablation)
- `number_of_interventions`, `intervention_type_diversity`, `mono_therapy`, `has_placebo`, `has_active_comparator`, `n_mesh_intervention_terms`

### Design outcomes (from design_outcomes table)
- `max_planned_followup_days` — max planned follow-up parsed from time_frame
- `n_primary_outcomes`, `n_secondary_outcomes`, `n_outcomes`
- `has_survival_endpoint`, `has_safety_endpoint` — flags from measure/description
- `endpoint_complexity_score` — composite of outcome count and endpoint types

## Metrics (test set)
- RMSE ≈ 551 days
- MAE ≈ 360 days
- R² ≈ 0.3743

## Train/val/test split
60% / 20% / 20%, random_state=42
