"""RQ3: repeatability of the roughness measurement and the attainable error floor."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

out = {}
for t in TARGETS:
    g = raw.groupby("cut_id")[t]
    s2 = g.var(ddof=1)                         # within-cut variance of single readings
    sw = np.sqrt(s2.mean())                    # pooled SD of a single reading
    floor = sw / np.sqrt(3)                    # noise of the 3-reading mean = minimum attainable RMSE
    ybar = g.mean()
    sb2 = ybar.var(ddof=1)                     # variance of cut means
    r2max = max(sb2 - s2.mean() / 3, 0) / sb2
    k = 3
    msb, msw = k * ybar.var(ddof=1), s2.mean()
    icc = (msb - msw) / (msb + (k - 1) * msw)  # ICC(1)
    y = ybar.values
    loo = np.array([np.delete(y, i).mean() for i in range(len(y))])
    out[t] = dict(sd_single=sw, floor_rmse=floor, sd_between_means=np.sqrt(sb2), r2_max=r2max, icc=icc,
                  mean_model_rmse=metrics(y, loo)[0], cv_single_pct=100 * sw / raw[t].mean(), mean=float(raw[t].mean()))
    print(t, {k_: round(float(v), 4) for k_, v in out[t].items()})
json.dump(out, open("as3.json", "w"), indent=1, default=float)
