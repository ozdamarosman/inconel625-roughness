"""Shared paths, data loading, feature groups and metrics."""
import os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "features"                 # output of extract_features.py
OUT = ROOT / "results" / "intermediate"      # intermediate results and figures are written here
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "fig").mkdir(exist_ok=True)
os.chdir(OUT)

cut = pd.read_csv(FEATURES / "features_cut.csv").sort_values("cut_id").reset_index(drop=True)
win = pd.read_csv(FEATURES / "features_window.csv")
raw = pd.read_csv(FEATURES / "roughness_raw.csv")

PAR = ["speed_rpm", "feed_mm_rev", "doc_mm"]
SIG_ACC = ["AccX", "AccY", "AccZ"]
SIG_DYN = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]


def cols(signals, df=cut):
    """Feature columns belonging to the given signals."""
    return [c for c in df.columns if c.split("_")[0] in signals]


SENS = cols(SIG_ACC + SIG_DYN)
ACC = cols(SIG_ACC)
DYN = cols(SIG_DYN)
TARGETS = ["Ra", "Rq", "Rz"]


def metrics(y, p):
    """RMSE, MAE and R^2 of predictions p for targets y."""
    y, p = np.asarray(y), np.asarray(p)
    rmse = np.sqrt(np.mean((y - p) ** 2))
    mae = np.mean(np.abs(y - p))
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return rmse, mae, r2
