# Regeneron Capstone — Clinical trial duration model

Industry-sponsored clinical trials from ClinicalTrials.gov, preprocessed into `clean_data/`, then modeled with **HistGradientBoostingRegressor** ensembles to predict **`duration_days`** (primary completion minus start) for **completed** trials only.

**Modeling setup (see `4_regression/train_regression.py`):**

- **Dedicated models** fit on **PHASE1**, **PHASE2**, and **PHASE3** only.
- **Early joint model** trained on **PHASE1 + PHASE1/PHASE2 + PHASE2**; used to score **PHASE1/PHASE2** trials.
- **Late joint model** trained on **PHASE2 + PHASE2/PHASE3 + PHASE3**; used to score **PHASE2/PHASE3** trials.

Metrics are still reported **per phase label** (all five labels in `PHASE_REPORT_ORDER`). High-level feature and preprocessing rules are in **`MODEL.md`**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

Place raw extracts under **`raw_data/`** (e.g. from `1_scripts/` BigQuery downloads), then:

```bash
python 3_preprocessing/preprocess.py                 # builds clean_data/
python 4_regression/train_regression.py              # writes results/regression_report.txt
```

**Full pipeline** (downloads, EDA, preprocess, train):

```bash
python main.py                    # runs 1_scripts downloads + exploration + preprocess + train
python main.py --skip-download    # same, when raw_data/ is already populated
```

**Prediction deviation analysis** (optional; per-trial actual vs predicted %):

```bash
python 5_deviation/baseline_deviation.py
```

---

## Results

After `python 4_regression/train_regression.py`, open **`results/regression_report.txt`** for train / val / test **RMSE**, **MAE**, and **R²** per phase label, plus joint-model tables where applicable.

**Design choices that help:** no feature scaling, **missing numerics left as NaN** for the histogram gradient booster, **log1p** on the target inside `TransformedTargetRegressor`, and **`max_iter=200`** per fitted ensemble. Example numbers from a prior run: Phase 1 test **R² ≈ 0.60**; Phase 2 and 3 **≈ 0.42–0.43**. Re-run training to refresh metrics after any data or code change.

---

## Repository layout (high level)

| Path | Role |
|------|------|
| `1_scripts/` | BigQuery download helpers (`bq_downloader.py` + per-table scripts) → `raw_data/` |
| `2_data_exploration/` | EDA scripts; `run_all.py` runs them in sequence |
| `3_preprocessing/` | `preprocess.py` → `clean_data/` |
| `4_regression/` | `train_regression.py` → `results/regression_report.txt` |
| `5_deviation/` | `baseline_deviation.py` — optional deviation / error-band analysis |
| `main.py` | Optional end-to-end pipeline runner |

Raw CSVs live under **`raw_data/`** (not committed; use your own extracts or the download scripts). Generated artifacts use **`clean_data/`**, **`results/`**, and paths listed in **`.gitignore`**.
