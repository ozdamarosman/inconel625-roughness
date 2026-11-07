"""RQ2: twelve models compared with leave-one-cut-out validation (n = 27)."""
import os
import sys
import time
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from common import *
from models import make
from sklearn.base import clone

MODELS = {
    "Mean": ("mean", PAR),
    "Kinematic (f²)": ("kinematic", PAR),
    "Parameters–Linear": ("param_lin", PAR),
    "Parameters–GPR": ("param_gpr", PAR),
    "Parameters–RF": ("rf", PAR),
    "Sensors–Ridge": ("ridge", SENS),
    "Sensors–RF": ("rf", SENS),
    "Sensors–GPR (k=10)": ("gpr_k10", SENS),
    "Accelerometer–RF": ("rf", ACC),
    "Dynamometer–RF": ("rf", DYN),
    "Params+Sensors–RF": ("rf", PAR + SENS),
    "Params+Sensors–Ridge": ("ridge", PAR + SENS),
}


def loo_predict(model, X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        p[i] = clone(model).fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    return p


rows, preds = [], {}
for t in TARGETS:
    y = cut[f"{t}_mean"].values
    for label, (name, columns) in MODELS.items():
        t0 = time.time()
        p = loo_predict(make(name), cut[columns].values, y)
        preds[(t, label)] = p
        r = metrics(y, p)
        rows.append(dict(target=t, model=label, n_feat=len(columns), RMSE=r[0], MAE=r[1], R2=r[2]))
        print(f"{t:3s} {label:22s} RMSE={r[0]:.4f} R2={r[2]:.3f}  ({time.time() - t0:.0f} s)", flush=True)
pd.DataFrame(rows).to_csv("as2_results.csv", index=False)
pd.to_pickle(preds, "as2_preds.pkl")
