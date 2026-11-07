"""Runs the full analysis in order.  Usage:  python run_all.py

Requires the 'features' folder (included in the repository; it can be regenerated with extract_features.py).
Total run time is about 15-25 minutes on an ordinary laptop."""
import subprocess
import sys
import time
from pathlib import Path

A = Path(__file__).resolve().parent / "analysis"
STEPS = [["as3_measurement_floor.py"], ["as2_models.py"], ["as2_robustness.py"], ["as1_validation.py"],
         ["as1_fingerprint.py"], ["as4_uncertainty.py", "Ra", "LOO"], ["as4_uncertainty.py", "Ra", "feed,speed"],
         ["as4_uncertainty.py", "Rq", "LOO"], ["as4_uncertainty.py", "Rz", "LOO"], ["figures.py"], ["tables.py"]]
for step in STEPS:
    t = time.time()
    print("\n>>>", " ".join(step), flush=True)
    if subprocess.run([sys.executable, str(A / step[0])] + step[1:], cwd=A).returncode:
        sys.exit(f"ERROR: {step[0]} failed")
    print(f"    ({time.time() - t:.0f} s)")
print("\nDone. Results in results/intermediate/ (figures: fig/, workbook: results.xlsx)")
