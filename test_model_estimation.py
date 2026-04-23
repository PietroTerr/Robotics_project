"""
test_model_estimation.py
========================
Offline evaluation of TerrainPredictor using a pre-generated map CSV.

Pipeline
--------
1. Load 'generated_maps/map_001_seed1.csv' and build CellData objects.
2. All rows carry full ground-truth data (traversability + stuck_event),
   so every cell is treated as *visited* (ground truth known).
3. Split cells into a training set and a held-out test set (80 / 20).
4. Fit a TerrainPredictor on the training cells.
5. Run TerrainPredictorEvaluator on the test cells → full report + plots.
"""

import os
import sys
import csv
import random

# ── Make sure the project root is importable ──────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from CellData import CellData
from TerrainPredictor import TerrainPredictor
from TerrainPredictorEvaluator import TerrainPredictorEvaluator

# ── Configuration ─────────────────────────────────────────────────────────────
CSV_PATH   = os.path.join(ROOT, "generated_maps", "map_002_seed2.csv")
TRAIN_RATIO = 0.80          # fraction of cells used for training
RANDOM_SEED = 42            # reproducible split
K_FOLDS     = 5             # cross-validation folds


# ── 1. Load CSV → list[CellData] ─────────────────────────────────────────────
def load_cells(path: str) -> list[CellData]:
    """Parse the map CSV and return fully-populated CellData objects."""
    cells: list[CellData] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cell = CellData(x=int(row["x"]), y=int(row["y"]))

            # Sensor observations
            cell.set_texture(float(row["texture"]))
            cell.set_color(float(row["color"]))
            cell.set_slope(float(row["slope"]))
            cell.set_uphill_angle(float(row["uphill_angle"]))

            # Ground-truth labels (makes cell.is_visited = True)
            cell.set_real_traversability(float(row["traversability"]))
            cell.set_is_stuck(row["stuck_event"].strip().lower() == "true")

            cells.append(cell)
    return cells


# ── 2. Train / test split ─────────────────────────────────────────────────────
def train_test_split(
    cells: list[CellData],
    train_ratio: float,
    seed: int,
) -> tuple[list[CellData], list[CellData]]:
    shuffled = cells[:]
    random.seed(seed)
    random.shuffle(shuffled)
    cut = int(len(shuffled) * train_ratio)
    return shuffled[:cut], shuffled[cut:]


# ── 3. Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"\n{'═' * 60}")
    print("  TerrainPredictor — Model Estimation Test")
    print(f"{'═' * 60}\n")

    # --- Load data -----------------------------------------------------------
    print(f"[1/4] Loading map:  {CSV_PATH}")
    all_cells = load_cells(CSV_PATH)

    visited = [c for c in all_cells if c.is_visited]
    print(f"      Total cells   : {len(all_cells)}")
    print(f"      Visited cells : {len(visited)}")
    stuck_count = sum(1 for c in visited if c.is_stuck)
    print(f"      Stuck events  : {stuck_count} ({stuck_count/len(visited)*100:.1f} %)\n")

    if len(visited) < 10:
        raise RuntimeError(
            "Not enough visited cells to run a meaningful evaluation "
            f"(found {len(visited)})."
        )

    # --- Split ---------------------------------------------------------------
    print(f"[2/4] Splitting data  (train={TRAIN_RATIO*100:.0f} % / test={100-TRAIN_RATIO*100:.0f} %)")
    train_cells, test_cells = train_test_split(visited, TRAIN_RATIO, RANDOM_SEED)
    print(f"      Train : {len(train_cells)}  |  Test : {len(test_cells)}\n")

    # --- Fit predictor -------------------------------------------------------
    print("[3/4] Fitting TerrainPredictor on training cells …")
    predictor = TerrainPredictor()
    # observed_cells = all train cells (predictor will also predict on them);
    # visited_cells  = same set (ground truth comes from here).
    predictor.update_predictor_model(
        observed_cells=train_cells,
        visited_cells=train_cells,
    )
    print(f"      GP regression fitted : {predictor._model_fitted}")
    print(f"      GP classifier fitted : {predictor._stuck_model_ready}\n")

    # --- Evaluate ------------------------------------------------------------
    print(f"[4/4] Evaluating on held-out test set ({len(test_cells)} cells) …\n")
    evaluator = TerrainPredictorEvaluator(predictor, test_cells)
    report = evaluator.full_report(k_folds=K_FOLDS)

    print(report.summary())

    # --- Diagnostic plots ----------------------------------------------------
    print("\nGenerating diagnostic plots …  (close the window to exit)")
    try:
        evaluator.plot_diagnostics()
    except ImportError as exc:
        print(f"[WARN] Could not show plots: {exc}")


if __name__ == "__main__":
    main()
