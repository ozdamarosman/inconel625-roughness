"""
FEATURE EXTRACTION FROM THE INCONEL 625 TURNING DATASET
=======================================================
Open dataset of Harigovind, Nair and Sakthivel (Mendeley Data, DOI: 10.17632/ybyhxv4czd.3).
Reads the accelerometer, dynamometer and roughness files in the "Synchronized" folder and
writes small CSV files for modelling.

USAGE
    pip install -r requirements.txt
    python extract_features.py "<path>/Inconel-625 Dataset"

    Without an argument the script looks for an "Inconel-625 Dataset" folder next to itself.

OUTPUT (folder "features")
    features_cut.csv      27 rows: per-cut features + mean Ra/Rq/Rz
    features_window.csv   one row per 1 s window (each row carries its cut_id)
    roughness_raw.csv     the 3 individual roughness readings of each cut
    cut_info.csv          samples, duration and sampling-rate check per cut
    log.txt               run log (column names, warnings)
"""

import sys
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats, signal

# ==================== SETTINGS ====================
FS = 10_000              # sampling rate (Hz) - 10 kHz in this dataset
WINDOW_SEC = 1.0         # window length (s) for window-level features (RQ1)
WELCH_NPERSEG = 4096     # Welch PSD segment length
ACC_NAMES = ["AccX", "AccY", "AccZ"]
DYN_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

# features per signal (10 time-domain + 4 frequency-domain = 14)
TIME_FEATS = ["mean", "median", "std", "rms", "skew", "kurt", "max", "min", "range", "iqr"]
FREQ_FEATS = ["spec_energy", "dom_freq", "spec_centroid", "spec_spread"]

FNAME_RE = re.compile(r"^([afr])_(\d+)_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)$", re.IGNORECASE)

LOG_LINES = []


def log(msg):
    print(msg)
    LOG_LINES.append(msg)


# ==================== FEATURE FUNCTIONS ====================
def time_features(x):
    x = np.asarray(x, dtype=np.float64)
    q75, q25 = np.percentile(x, [75, 25])
    return {
        "mean": x.mean(),
        "median": np.median(x),
        "std": x.std(ddof=1),
        "rms": np.sqrt(np.mean(x ** 2)),
        "skew": stats.skew(x),
        "kurt": stats.kurtosis(x, fisher=False),  # same definition as MATLAB kurtosis()
        "max": x.max(),
        "min": x.min(),
        "range": x.max() - x.min(),
        "iqr": q75 - q25,
    }


def freq_features(x):
    x = np.asarray(x, dtype=np.float64)
    nper = min(WELCH_NPERSEG, len(x))
    f, pxx = signal.welch(x, fs=FS, nperseg=nper, detrend="constant")
    df = f[1] - f[0]
    total = pxx.sum()
    if total <= 0:
        return {k: np.nan for k in FREQ_FEATS}
    centroid = (f * pxx).sum() / total
    spread = np.sqrt(((f - centroid) ** 2 * pxx).sum() / total)
    dom = f[1:][np.argmax(pxx[1:])]  # excluding the DC component
    return {
        "spec_energy": total * df,
        "dom_freq": dom,
        "spec_centroid": centroid,
        "spec_spread": spread,
    }


def signal_features(x, prefix):
    out = {}
    for k, v in time_features(x).items():
        out[f"{prefix}_{k}"] = v
    for k, v in freq_features(x).items():
        out[f"{prefix}_{k}"] = v
    return out


# ==================== FILE READING ====================
def find_files(folder, letter):
    """Return the a_/f_/r_ files of a folder keyed by (speed, feed, depth)."""
    found = {}
    for p in folder.iterdir():
        if p.suffix.lower() not in (".csv", ".xlsx", ".xls"):
            continue
        m = FNAME_RE.match(p.stem)
        if m and m.group(1).lower() == letter:
            key = (int(m.group(2)), float(m.group(3)), float(m.group(4)))
            # prefer csv when both csv and xlsx exist
            if key not in found or p.suffix.lower() == ".csv":
                found[key] = p
    return found


def read_table(path):
    if path.suffix.lower() == ".csv":
        try:
            return pd.read_csv(path, engine="pyarrow")
        except Exception:
            return pd.read_csv(path)
    return pd.read_excel(path)


def read_signal(path, names, kind):
    df = read_table(path)
    if df.shape[1] < len(names) + 1:
        raise ValueError(f"{path.name}: expected {len(names)+1} columns, found {df.shape[1]}")
    orig_cols = list(df.columns)
    t = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    data = df.iloc[:, 1:len(names) + 1].apply(pd.to_numeric, errors="coerce")
    data.columns = names
    n_nan = int(data.isna().sum().sum())
    if n_nan:
        log(f"   WARNING {path.name}: {n_nan} non-numeric values dropped")
        data = data.dropna()
    return t, data, orig_cols


def read_roughness(path):
    df = read_table(path)
    # re-read without header if the header row is numeric
    try:
        [float(c) for c in df.columns]
        df = pd.read_csv(path, header=None) if path.suffix.lower() == ".csv" else pd.read_excel(path, header=None)
    except (ValueError, TypeError):
        pass
    num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="any")
    if num.shape[1] < 3:
        raise ValueError(f"{path.name}: could not find 3 numeric columns (Ra, Rq, Rz). Columns: {list(df.columns)}")
    num = num.iloc[:, :3]
    num.columns = ["Ra", "Rq", "Rz"]
    return num.reset_index(drop=True), list(df.columns)


def estimate_fs(t):
    d = np.diff(t[: min(len(t), 100_000)])
    d = d[np.isfinite(d) & (d > 0)]
    return float(1.0 / np.median(d)) if len(d) else np.nan


# ==================== MAIN ====================
def main():
    t0 = time.time()
    if len(sys.argv) > 1:
        root = Path(sys.argv[1])
    else:
        root = Path(__file__).resolve().parent / "Inconel-625 Dataset"

    sync = root / "Synchronized"
    if not sync.exists():
        # the user may have passed the Synchronized folder itself
        sync = root if (root / "Dynamometer").exists() else None
    if sync is None:
        sys.exit(f"ERROR: no 'Synchronized' folder found in '{root}'.")

    dirs = {}
    for sub in ["Accelerometer", "Dynamometer", "Roughness"]:
        cand = [p for p in sync.iterdir() if p.is_dir() and p.name.lower() == sub.lower()]
        if not cand:
            sys.exit(f"ERROR: no '{sub}' folder in '{sync}'.")
        dirs[sub] = cand[0]

    out_dir = Path.cwd() / "features"
    out_dir.mkdir(exist_ok=True)
    log(f"Data folder  : {sync}")
    log(f"Output folder: {out_dir}")

    acc_files = find_files(dirs["Accelerometer"], "a")
    dyn_files = find_files(dirs["Dynamometer"], "f")
    rgh_files = find_files(dirs["Roughness"], "r")
    keys = sorted(set(acc_files) & set(dyn_files) & set(rgh_files))
    log(f"Files found: accelerometer={len(acc_files)}, dynamometer={len(dyn_files)}, "
        f"roughness={len(rgh_files)}, matched cuts={len(keys)}")
    missing = (set(acc_files) | set(dyn_files) | set(rgh_files)) - set(keys)
    if missing:
        log(f"   WARNING unmatched files: {sorted(missing)}")
    if not keys:
        sys.exit("ERROR: no matching files. Are the names like 'f_0400_0.05_0.20.csv'?")

    win = int(WINDOW_SEC * FS)
    cut_rows, win_rows, rgh_rows, info_rows = [], [], [], []

    for cut_id, key in enumerate(keys, start=1):
        speed, feed, doc = key
        log(f"[{cut_id:2d}/{len(keys)}] speed={speed} rpm, feed={feed} mm/rev, depth={doc} mm")

        ta, acc, acc_cols = read_signal(acc_files[key], ACC_NAMES, "acc")
        tf, dyn, dyn_cols = read_signal(dyn_files[key], DYN_NAMES, "dyn")
        rgh, rgh_cols = read_roughness(rgh_files[key])
        if cut_id == 1:
            log(f"   Accelerometer columns: {acc_cols}")
            log(f"   Dynamometer columns  : {dyn_cols}")
            log(f"   Roughness columns    : {rgh_cols}")

        base = {"cut_id": cut_id, "speed_rpm": speed, "feed_mm_rev": feed, "doc_mm": doc}

        # --- roughness: raw readings + mean/std
        for j, r in rgh.iterrows():
            rgh_rows.append({**base, "reading": j + 1, **r.to_dict()})
        rstats = {}
        for c in ["Ra", "Rq", "Rz"]:
            rstats[f"{c}_mean"] = rgh[c].mean()
            rstats[f"{c}_std"] = rgh[c].std(ddof=1) if len(rgh) > 1 else np.nan
        rstats["n_readings"] = len(rgh)

        # --- cut-level features (whole trimmed signal)
        feats = {}
        for name in ACC_NAMES:
            feats.update(signal_features(acc[name].to_numpy(), name))
        for name in DYN_NAMES:
            feats.update(signal_features(dyn[name].to_numpy(), name))
        cut_rows.append({**base, **rstats, **feats})

        # --- window-level features (windows of a cut share its cut_id)
        n_common = min(len(acc), len(dyn))
        n_win = n_common // win
        for w in range(n_win):
            s, e = w * win, (w + 1) * win
            wf = {}
            for name in ACC_NAMES:
                wf.update(signal_features(acc[name].to_numpy()[s:e], name))
            for name in DYN_NAMES:
                wf.update(signal_features(dyn[name].to_numpy()[s:e], name))
            win_rows.append({**base, "window": w + 1,
                             "Ra_mean": rstats["Ra_mean"], "Rq_mean": rstats["Rq_mean"],
                             "Rz_mean": rstats["Rz_mean"], **wf})

        info_rows.append({**base,
                          "acc_samples": len(acc), "dyn_samples": len(dyn),
                          "acc_duration_s": len(acc) / FS, "dyn_duration_s": len(dyn) / FS,
                          "acc_fs_est": estimate_fs(ta), "dyn_fs_est": estimate_fs(tf),
                          "acc_t0": ta[0] if len(ta) else np.nan, "dyn_t0": tf[0] if len(tf) else np.nan,
                          "n_windows": n_win})
        log(f"   duration: acc={len(acc)/FS:.1f}s, dyn={len(dyn)/FS:.1f}s, windows={n_win}, "
            f"Ra readings={rgh['Ra'].round(3).tolist()}")

    pd.DataFrame(cut_rows).to_csv(out_dir / "features_cut.csv", index=False)
    pd.DataFrame(win_rows).to_csv(out_dir / "features_window.csv", index=False)
    pd.DataFrame(rgh_rows).to_csv(out_dir / "roughness_raw.csv", index=False)
    pd.DataFrame(info_rows).to_csv(out_dir / "cut_info.csv", index=False)

    n_feat = len(ACC_NAMES + DYN_NAMES) * (len(TIME_FEATS) + len(FREQ_FEATS))
    log("")
    log(f"DONE ({time.time()-t0:.0f} s)")
    log(f"   features_cut.csv    : {len(cut_rows)} cuts x {n_feat} features")
    log(f"   features_window.csv : {len(win_rows)} windows ({WINDOW_SEC} s)")
    log(f"   roughness_raw.csv   : {len(rgh_rows)} readings")
    log(f"   cut_info.csv        : cut durations and sampling-rate check")
    log(f"Features written to '{out_dir.name}'.")
    (out_dir / "log.txt").write_text("\n".join(LOG_LINES), encoding="utf-8")


if __name__ == "__main__":
    main()
