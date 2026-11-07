# Sensor-based surface roughness prediction in Inconel 625 turning: validation leakage, measurement floor and uncertainty

This repository contains the code and derived data needed to reproduce every result of the manuscript:

> Özdamar, O. *Validation leakage, measurement floor and uncertainty in sensor-based surface roughness prediction: an evaluation on an open Inconel 625 turning dataset.* (Under review.)

## Main findings

- Shuffling 1 s signal windows in 5-fold cross-validation gives R² = 0.97 for Ra, with an error (0.021 µm) **below the profilometer's own repeatability** (0.051 µm). A classifier identifies the cut a window comes from with 97.8 % accuracy (chance: 3.7 %). Holding whole cuts out drops R² to −0.44.
- Under leave-one-cut-out validation none of 11 models (cutting parameters, 126 sensor features, or both) beats the mean predictor for Ra or Rq (permutation p = 0.15).
- For 90 % prediction intervals, observed coverage is 93 % for jackknife+, 81–85 % for Gaussian process regression, 74–78 % for quantile gradient boosting and 52 % for bootstrap random forests.

## Data

The raw data are **not** included. Source dataset:

- Harigovind, H., Nair, B.B., Sakthivel, N.R. (2025). *A sensor based turning dataset for data-driven surface roughness estimation* (V3). Mendeley Data. https://doi.org/10.17632/ybyhxv4czd.3 (CC BY 4.0)
- Sakthivel, N.R., Harigovind, H., Nair, B.B. (2026). A sensor based turning dataset for data-driven surface roughness estimation. *Scientific Data*, 13, 742. https://doi.org/10.1038/s41597-026-07061-1

The files in `features/` were derived from the dataset's "Synchronized" files with `extract_features.py`. They are shared under the same CC BY 4.0 licence with attribution to the source.

## Repository layout

```
extract_features.py          Feature extraction from the raw data (cut level and 1 s windows; 9 signals x 14 features)
features/                    Extracted features and the 3 roughness readings per cut (ready to use)
analysis/
  common.py, models.py       Shared data loading, metrics and model definitions
  as3_measurement_floor.py   RQ3: measurement repeatability and attainable error floor
  as2_models.py              RQ2: 12 models under leave-one-cut-out validation
  as2_robustness.py          RQ2: feature-count sensitivity, permutation test, multiple comparisons, ANOVA
  as1_validation.py          RQ1: five validation schemes on window-level data
  as1_fingerprint.py         RQ1: do windows reveal their cut identity?
  as4_uncertainty.py         RQ4: coverage and interval score of GPR, quantile GB, bootstrap RF and jackknife+
  figures.py, tables.py      Figures 1-4 and an Excel summary
run_all.py                   Runs all analyses in order
results/                     Figures and summary workbook used in the manuscript
```

## How to run

Python 3.10 or later.

```
pip install -r requirements.txt
python run_all.py
```

Outputs are written to `results/intermediate/`. The full run takes about 15-25 minutes on an ordinary laptop. To regenerate the features from the raw data, download the dataset and run:

```
python extract_features.py "<path>/Inconel-625 Dataset"
```

All preprocessing (scaling, feature selection, hyperparameter fitting) is done inside the validation folds, and all random seeds are fixed.

## Citation

If you use this code, please cite the manuscript and the Zenodo record (see `CITATION.cff`).

## Licence

Code: MIT (see `LICENSE`). Derived data (`features/`): CC BY 4.0, with attribution to the source dataset.
