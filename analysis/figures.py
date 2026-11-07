"""Figures 1-4 of the manuscript (written to results/intermediate/fig/)."""
import json
import os
import sys
import warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")
from common import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "savefig.dpi": 300})
a3 = json.load(open("as3.json"))
FLOOR, MEAN = a3["Ra"]["floor_rmse"], a3["Ra"]["mean_model_rmse"]


def refs(ax):
    ax.axvline(FLOOR, color=INK2, ls=":", lw=1.2)
    ax.axvline(MEAN, color=INK2, ls="--", lw=1.2)


# Figure 1: measurement repeatability
order = cut.sort_values("Ra_mean").cut_id.values
pos = {c: i for i, c in enumerate(order)}
c_idx = cut.set_index("cut_id").loc[order]
fig, ax = plt.subplots(figsize=(6.3, 2.8))
ax.scatter(raw.cut_id.map(pos), raw.Ra, s=16, color=S[0], alpha=0.75, label="Single reading (3 per cut)", zorder=3)
ax.scatter(range(27), c_idx.Ra_mean, marker="_", s=120, color=INK, lw=1.6, label="Cut mean", zorder=4)
ax.set_xticks(range(27))
ax.set_xticklabels([f"{int(r.speed_rpm)}/{r.feed_mm_rev:.2f}/{r.doc_mm:.1f}" for _, r in c_idx.iterrows()],
                   rotation=90, fontsize=6)
ax.grid(axis="x", visible=False)
ax.set_ylabel("Ra (µm)")
ax.set_xlabel("Cut (spindle speed rpm / feed mm/rev / depth of cut mm), sorted by mean Ra")
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
plt.tight_layout(); plt.savefig("fig/figure_1.png"); plt.close()

# Figure 2: validation scheme (Ra, RF, cut level)
d = pd.read_csv("as1_results.csv")
d = d[(d.target == "Ra") & (d.model == "RF")]
label = {"Random 5-fold (windows)": "Windows shuffled (5-fold)", "Leave-one-cut-out (27)": "Cut held out (27 folds)",
         "Leave-one-speed-out (3)": "Speed level held out", "Leave-one-feed-out (3)": "Feed level held out",
         "Leave-one-depth-out (3)": "Depth level held out"}
fig, ax = plt.subplots(figsize=(6.3, 2.9))
yy = np.arange(len(d))[::-1]
ax.barh(yy, d.RMSE_cut, height=0.55, color=S[0])
for y_, v, r in zip(yy, d.RMSE_cut, d.R2_cut):
    ax.text(v + 0.003, y_, f"{v:.3f}  (R²={r:.2f})", va="center", fontsize=8, color=INK)
ax.set_yticks(yy); ax.set_yticklabels([label[s] for s in d.scheme]); ax.grid(axis="y", visible=False)
refs(ax); ax.set_xlim(0, 0.23)
ax.text(FLOOR, len(d) - 0.35, " measurement floor", fontsize=7.5, color=INK2, va="bottom")
ax.text(MEAN, len(d) - 0.35, " mean predictor", fontsize=7.5, color=INK2, va="bottom")
ax.set_xlabel("Cut-level RMSE, Ra (µm)"); ax.set_ylim(-0.6, len(d) - 0.1)
plt.tight_layout(); plt.savefig("fig/figure_2.png"); plt.close()

# Figure 3: model comparison (Ra, leave-one-cut-out)
d = pd.read_csv("as2_results.csv")
d = d[d.target == "Ra"].sort_values("RMSE", ascending=False)
fig, ax = plt.subplots(figsize=(6.3, 3.9))
yy = np.arange(len(d))
col = [INK2 if m == "Mean" else (S[1] if m.startswith(("Kinematic", "Parameters")) else S[0]) for m in d.model]
ax.barh(yy, d.RMSE, height=0.6, color=col)
for y_, v in zip(yy, d.RMSE):
    ax.text(v + 0.002, y_, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(yy); ax.set_yticklabels(d.model); ax.grid(axis="y", visible=False); refs(ax); ax.set_xlim(0, 0.19)
ax.legend(handles=[Patch(color=S[0], label="With sensor features"), Patch(color=S[1], label="Cutting parameters only"),
                   Patch(color=INK2, label="Mean predictor")],
          loc="lower center", bbox_to_anchor=(0.45, 1.06), ncol=3, fontsize=7.5, frameon=False)
ax.text(FLOOR, len(d) - 0.45, " measurement floor", fontsize=7.5, color=INK2, va="bottom")
ax.set_ylim(-0.6, len(d) + 0.2)
ax.set_xlabel("Leave-one-cut-out RMSE, Ra (µm)")
plt.tight_layout(); plt.savefig("fig/figure_3.png"); plt.close()

# Figure 4: reliability diagram (Ra, leave-one-cut-out)
d = pd.read_csv("as4_Ra_LOO.csv")
sel = [("Jackknife+ Mean", "—", "Jackknife+ (mean model)"), ("Jackknife+ RF", "Sensors", "Jackknife+ RF (sensors)"),
       ("GPR", "Sensors", "GPR (sensors)"), ("Quantile GB", "Parameters", "Quantile GB (parameters)"),
       ("Bootstrap RF", "Sensors", "Bootstrap RF (sensors)")]
markers = ["o", "s", "D", "^", "v"]
levels = [50, 80, 90, 95]
fig, ax = plt.subplots(figsize=(4.4, 4.0))
ax.plot([45, 100], [45, 100], color=INK2, lw=1, ls="--")
for i, (m, inp, lb) in enumerate(sel):
    r = d[(d.method == m) & (d.input == inp)].iloc[0]
    ax.plot(levels, [100 * r[f"cov{l}"] for l in levels], marker=markers[i], ms=6, lw=2, color=S[i], label=lb,
            markeredgecolor="white", markeredgewidth=1)
ax.set_xlim(45, 100); ax.set_ylim(10, 102); ax.set_xticks(levels)
ax.set_xlabel("Nominal coverage (%)"); ax.set_ylabel("Observed coverage (%), n = 27")
ax.text(62, 57, "ideal calibration", fontsize=7.5, color=INK2, ha="left", va="top", rotation=38)
ax.legend(fontsize=7.2, frameon=False, loc="lower right")
plt.tight_layout(); plt.savefig("fig/figure_4.png"); plt.close()
print("figures written to", OUT / "fig")
