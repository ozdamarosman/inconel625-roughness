"""RQ1: effect of the validation scheme on window-level models (1740 one-second windows)."""
import os
import sys
import time
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from common import *
from models import make
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor

WS = cols(SIG_ACC + SIG_DYN, win)
X = win[WS].values
SCHEMES = {
    "Random 5-fold (windows)": None,
    "Leave-one-cut-out (27)": "cut_id",
    "Leave-one-speed-out (3)": "speed_rpm",
    "Leave-one-feed-out (3)": "feed_mm_rev",
    "Leave-one-depth-out (3)": "doc_mm",
}


def rf():
    return RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.33, random_state=0, n_jobs=-1)


MODELS = {"RF": rf, "Ridge": lambda: make("ridge")}
rows, store = [], {}
for t in TARGETS:
    y = win[f"{t}_mean"].values
    for mname, mk in MODELS.items():
        for sname, group in SCHEMES.items():
            t0 = time.time()
            splits = (KFold(5, shuffle=True, random_state=0).split(X) if group is None
                      else LeaveOneGroupOut().split(X, groups=win[group].values))
            p = np.zeros(len(y))
            for tr, te in splits:
                p[te] = mk().fit(X[tr], y[tr]).predict(X[te])
            rw = metrics(y, p)
            agg = pd.DataFrame({"c": win.cut_id, "y": y, "p": p}).groupby("c").mean()   # aggregate to cut level
            rc = metrics(agg.y, agg.p)
            rows.append(dict(target=t, model=mname, scheme=sname,
                             RMSE_win=rw[0], R2_win=rw[2], RMSE_cut=rc[0], R2_cut=rc[2]))
            store[(t, mname, sname)] = p
            print(f"{t} {mname:5s} {sname:26s} window R2={rw[2]:.3f} | cut R2={rc[2]:.3f} RMSE={rc[0]:.4f}"
                  f"  ({time.time() - t0:.0f} s)", flush=True)
pd.DataFrame(rows).to_csv("as1_results.csv", index=False)
pd.to_pickle(store, "as1_preds.pkl")
