"""
analyze_map_features.py
-----------------------
For each map CSV, determines which features (slope, uphill_angle, texture, color)
are actually used as arguments to compute `traversability` and `stuck_event`.

The function that maps features → targets may differ between maps, but this
script checks whether the *argument set* (i.e. which features matter) is
consistent across maps.

Usage:
    python analyze_map_features.py                        # scans current folder
    python analyze_map_features.py path/to/maps/          # scans a specific folder
    python analyze_map_features.py map_001.csv map_002.csv # explicit files

Output:
    - Per-map table of feature relevance scores
    - Summary table showing which features are "active" per map
    - Consistency report across maps
"""

import sys
import os
import glob
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── Configuration ────────────────────────────────────────────────────────────

FEATURES      = ["slope", "uphill_angle", "texture", "color"]
TARGETS       = {
    "traversability": "regression",
    "stuck_event":    "classification",
}

# A feature is considered "used" if its RF importance exceeds this threshold.
# RF sets near-irrelevant features to ~0, so 1/(2*n_features) cleanly separates
# truly used features from noise.  MI is used as a secondary tie-breaker.
RF_THRESHOLD = 1 / (2 * len(FEATURES))    # 0.125  (half the uniform baseline)

RF_N_ESTIMATORS = 200
RF_RANDOM_STATE = 42

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        missing = [c for c in FEATURES + list(TARGETS) if c not in df.columns]
        if missing:
            print(f"  [SKIP] {path} — missing columns: {missing}")
            return None
        return df
    except Exception as e:
        print(f"  [ERROR] {path} — {e}")
        return None


def normalise(arr: np.ndarray) -> np.ndarray:
    """Normalise an importance vector to sum to 1, guarding against all-zero."""
    total = arr.sum()
    return arr / total if total > 0 else np.ones_like(arr) / len(arr)


def analyse_target(X: pd.DataFrame, y: pd.Series, task: str) -> dict:
    """
    Returns a dict  {feature: {"mi": float, "rf": float, "combined": float}}
    for one target.  All scores are normalised to [0,1] (sum = 1).
    """
    X_arr = X.values.astype(float)
    y_arr = y.values

    # Mutual information
    if task == "regression":
        mi_raw = mutual_info_regression(X_arr, y_arr, random_state=RF_RANDOM_STATE)
        rf     = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, n_jobs=-1
        )
    else:
        # encode bool → int just in case
        if y_arr.dtype == bool:
            y_arr = y_arr.astype(int)
        mi_raw = mutual_info_classif(X_arr, y_arr, random_state=RF_RANDOM_STATE)
        rf     = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE, n_jobs=-1
        )

    rf.fit(X_arr, y_arr)
    rf_raw = rf.feature_importances_

    mi_norm  = normalise(mi_raw)
    rf_norm  = normalise(rf_raw)
    combined = normalise((mi_norm + rf_norm) / 2)

    return {
        feat: {
            "mi":       round(float(mi_norm[i]),  4),
            "rf":       round(float(rf_norm[i]),  4),
            "combined": round(float(combined[i]), 4),
        }
        for i, feat in enumerate(X.columns)
    }


def active_features(scores: dict) -> list[str]:
    """
    Return features that are meaningfully used.
    Primary criterion  : RF importance >= RF_THRESHOLD  (separates noise from signal)
    Secondary criterion: MI importance >= 1/n_features  (catches edge cases)
    A feature is active if EITHER criterion fires.
    """
    uniform = 1 / len(FEATURES)
    return [
        f for f, s in scores.items()
        if s["rf"] >= RF_THRESHOLD or s["mi"] >= uniform
    ]


def print_map_report(map_name: str, results: dict):
    print(f"\n{'━'*60}")
    print(f"  MAP: {map_name}")
    print(f"{'━'*60}")
    for target, scores in results.items():
        active = active_features(scores)
        print(f"\n  Target: {target}  (active: {active or 'none'!r})")
        print(f"  {'Feature':<14} {'MI':>8} {'RF':>8} {'Combined':>10}  {'Active?':>8}")
        print(f"  {'-'*52}")
        for feat, s in scores.items():
            marker = "  ✓" if feat in active else ""
            print(
                f"  {feat:<14} {s['mi']:>8.4f} {s['rf']:>8.4f} "
                f"{s['combined']:>10.4f} {marker}"
            )


def print_summary(all_results: dict[str, dict]):
    """Print a cross-map summary matrix."""
    maps   = list(all_results.keys())
    targets_list = list(TARGETS.keys())

    print(f"\n{'═'*60}")
    print("  CROSS-MAP SUMMARY")
    print(f"{'═'*60}")

    for target in targets_list:
        print(f"\n  Target: {target}")
        header = f"  {'Map':<28}" + "".join(f" {f[:5]:>6}" for f in FEATURES)
        print(header)
        print(f"  {'-'*56}")

        active_sets = []
        for map_name, results in all_results.items():
            scores = results[target]
            active = set(active_features(scores))
            active_sets.append(active)
            row = f"  {map_name:<28}" + "".join(
                f"  {'✓':>4}" if f in active else f"  {'·':>4}"
                for f in FEATURES
            )
            print(row)

        # Consistency check
        if len(active_sets) > 1:
            intersection = set.intersection(*active_sets)
            union        = set.union(*active_sets)
            inconsistent = union - intersection
            print(f"\n  Always active  : {sorted(intersection) or '—'}")
            print(f"  Sometimes active: {sorted(union - intersection) or '—'}")
            print(f"  Never active   : {sorted(set(FEATURES) - union) or '—'}")

            if not inconsistent:
                print(f"  ✅ Argument set is CONSISTENT across all maps for '{target}'")
            else:
                print(
                    f"  ⚠️  Argument set is INCONSISTENT across maps for '{target}'\n"
                    f"     Features that vary: {sorted(inconsistent)}"
                )


# ── Main ──────────────────────────────────────────────────────────────────────

def collect_files(argv: list[str]) -> list[str]:
    if len(argv) > 1:
        files = []
        for arg in argv[1:]:
            if os.path.isdir(arg):
                files += sorted(glob.glob(os.path.join(arg, "*.csv")))
            else:
                files.append(arg)
        return files
    # default: current directory
    return sorted(glob.glob("*.csv"))


def main():
    files = collect_files(sys.argv)

    if not files:
        print("No CSV files found. Pass a folder or file paths as arguments.")
        sys.exit(1)

    print(f"\nFound {len(files)} CSV file(s).")
    print(f"Features  : {FEATURES}")
    print(f"Targets   : {list(TARGETS)}")
    print(f"Threshold : RF >= {RF_THRESHOLD:.3f} OR MI >= {1/len(FEATURES):.3f} → feature is 'active'")

    all_results: dict[str, dict] = {}

    for path in files:
        map_name = Path(path).stem
        df = load_csv(path)
        if df is None:
            continue

        X = df[FEATURES]
        results = {}
        for target, task in TARGETS.items():
            y = df[target]
            results[target] = analyse_target(X, y, task)

        all_results[map_name] = results
        print_map_report(map_name, results)

    if len(all_results) >= 2:
        print_summary(all_results)
    elif len(all_results) == 1:
        print("\n(Only one map analysed — run with multiple CSVs to compare consistency.)")

    print(f"\n{'═'*60}")
    print("  Done.")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()