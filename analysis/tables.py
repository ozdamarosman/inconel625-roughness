"""Collects all results into one Excel workbook (results/intermediate/results.xlsx)."""
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *

w = pd.ExcelWriter("results.xlsx")
pd.DataFrame(json.load(open("as3.json"))).T.round(4).to_excel(w, sheet_name="RQ3_measurement_floor")
pd.read_csv("as2_results.csv").round(4).to_excel(w, sheet_name="RQ2_models", index=False)
b = json.load(open("as2b.json"))
pd.DataFrame([{k: v for k, v in b.items() if k not in ("anova", "top_features")}]).T.to_excel(w, sheet_name="RQ2_robustness")
pd.DataFrame(b["top_features"], columns=["feature", "abs_r", "p"]).to_excel(w, sheet_name="RQ2_top_correlations", index=False)
pd.DataFrame(b["anova"]).to_excel(w, sheet_name="RQ2_ANOVA", index=False)
pd.read_csv("as1_results.csv").round(4).to_excel(w, sheet_name="RQ1_validation", index=False)
pd.DataFrame([json.load(open("as1_fingerprint.json"))]).to_excel(w, sheet_name="RQ1_fingerprint", index=False)
u = pd.concat([pd.read_csv(f) for f in ["as4_Ra_LOO.csv", "as4_Ra_feed_speed.csv", "as4_Rq_LOO.csv", "as4_Rz_LOO.csv"]])
u.round(3).to_excel(w, sheet_name="RQ4_uncertainty", index=False)
w.close()
print("results.xlsx written")
