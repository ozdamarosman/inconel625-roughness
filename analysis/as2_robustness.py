"""RQ2 robustness: feature-count sensitivity, permutation test, multiple-comparison check and ANOVA."""
import json
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from common import *
from models import make
from sklearn.base import clone
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf


def loo(model, X, y):
    p = np.zeros(len(y))
    for i in range(len(y)):
        tr = np.arange(len(y)) != i
        p[i] = clone(model).fit(X[tr], y[tr]).predict(X[i:i + 1])[0]
    return p


y = cut["Ra_mean"].values
X = cut[SENS].values
out = {}

# 1) sensitivity of Sensors-GPR to the number of selected features
for k in [3, 5, 20, 40]:
    out[f"gpr_k{k}"] = metrics(y, loo(make(f"gpr_k{k}"), X, y))[0]
    print("k", k, out[f"gpr_k{k}"], flush=True)

# 2) permutation test for Sensors-Ridge (200 permutations, p = (b+1)/(m+1))
rng = np.random.default_rng(0)
m = make("ridge")
obs = metrics(y, loo(m, X, y))[0]
null = [metrics(yp, loo(m, X, yp))[0] for yp in (rng.permutation(y) for _ in range(200))]
out["perm_ridge_obs"] = obs
out["perm_ridge_p"] = float((np.sum(np.array(null) <= obs) + 1) / 201)
print("permutation Sensors-Ridge", obs, out["perm_ridge_p"], flush=True)

# 3) multiple-comparison check: largest |r| over 126 features vs its permutation null
r = np.array([abs(stats.pearsonr(X[:, j], y)[0]) for j in range(X.shape[1])])
best = np.argsort(-r)[:5]
out["top_features"] = [(SENS[j], float(r[j]), float(stats.pearsonr(X[:, j], y)[1])) for j in best]
nullmax = [np.nanmax([abs(np.corrcoef(X[:, j], yp)[0, 1]) for j in range(X.shape[1])])
           for yp in (rng.permutation(y) for _ in range(1000))]
out["max_r_obs"] = float(r.max())
out["max_r_null95"] = float(np.percentile(nullmax, 95))
out["max_r_p"] = float((np.sum(np.array(nullmax) >= r.max()) + 1) / 1001)
out["n_p05"] = int(np.sum([stats.pearsonr(X[:, j], y)[1] < 0.05 for j in range(X.shape[1])]))
print(json.dumps({k_: v for k_, v in out.items() if k_ != "top_features"}, indent=1))

# 4) main effects of the cutting parameters on cut-mean Ra (type II ANOVA)
d = cut[PAR + ["Ra_mean"]].rename(columns={"speed_rpm": "v", "feed_mm_rev": "f", "doc_mm": "d", "Ra_mean": "Ra"})
a = sm.stats.anova_lm(smf.ols("Ra ~ C(v) + C(f) + C(d)", d).fit(), typ=2)
print(a)
out["anova"] = a.reset_index().to_dict("records")
json.dump(out, open("as2b.json", "w"), indent=1, default=float)
