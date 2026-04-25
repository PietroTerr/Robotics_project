from __future__ import annotations

"""
TerrainPredictorEvaluator
=========================
Offline evaluation utilities for :class:`TerrainPredictor`.

Usage example
-------------
>>> from TerrainPredictorEvaluator import TerrainPredictorEvaluator
>>> evaluator = TerrainPredictorEvaluator(predictor, visited_cells)
>>> report = evaluator.full_report()
>>> print(report.summary())
>>> evaluator.plot_diagnostics()           # requires matplotlib
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import KFold

if TYPE_CHECKING:
    from CellData import CellData
    from TerrainPredictor import TerrainPredictor


# ── Result containers ─────────────────────────────────────────────────────────

@dataclass
class TraversabilityMetrics:
    """Regression metrics for the traversability GP."""
    mae: float
    rmse: float
    r2: float
    mean_confidence: float
    confidence_calibration_error: float   # |mean_confidence - actual_accuracy|

    def summary(self) -> str:
        lines = [
            "── Traversability (regression) ──────────────────",
            f"  MAE                        : {self.mae:.4f}",
            f"  RMSE                       : {self.rmse:.4f}",
            f"  R²                         : {self.r2:.4f}",
            f"  Mean model confidence      : {self.mean_confidence:.4f}",
            f"  Confidence calibration err : {self.confidence_calibration_error:.4f}",
        ]
        return "\n".join(lines)


@dataclass
class StuckMetrics:
    """Classification metrics for the stuck-probability GP."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: np.ndarray

    def summary(self) -> str:
        tn, fp, fn, tp = self.confusion.ravel()
        lines = [
            "── Stuck prediction (classification) ────────────",
            f"  Accuracy  : {self.accuracy:.4f}",
            f"  Precision : {self.precision:.4f}",
            f"  Recall    : {self.recall:.4f}",
            f"  F1        : {self.f1:.4f}",
            f"  ROC-AUC   : {self.roc_auc:.4f}",
            f"  Confusion : TN={tn}  FP={fp}  FN={fn}  TP={tp}",
        ]
        return "\n".join(lines)


@dataclass
class CrossValidationReport:
    """Aggregated k-fold cross-validation scores."""
    trav_mae: list[float] = field(default_factory=list)
    trav_rmse: list[float] = field(default_factory=list)
    trav_r2: list[float] = field(default_factory=list)
    stuck_f1: list[float] = field(default_factory=list)
    stuck_roc_auc: list[float] = field(default_factory=list)

    def summary(self) -> str:
        def fmt(values: list[float]) -> str:
            if not values:
                return "n/a"
            arr = np.array(values)
            return f"{arr.mean():.4f} ± {arr.std():.4f}"

        lines = [
            "── Cross-validation results ──────────────────────",
            f"  Trav MAE     : {fmt(self.trav_mae)}",
            f"  Trav RMSE    : {fmt(self.trav_rmse)}",
            f"  Trav R²      : {fmt(self.trav_r2)}",
            f"  Stuck F1     : {fmt(self.stuck_f1)}",
            f"  Stuck ROC-AUC: {fmt(self.stuck_roc_auc)}",
        ]
        return "\n".join(lines)


@dataclass
class EvaluationReport:
    traversability: TraversabilityMetrics | None
    stuck: StuckMetrics | None
    cross_validation: CrossValidationReport | None

    def summary(self) -> str:
        parts = ["═" * 50, "  TerrainPredictor Evaluation Report", "═" * 50]
        if self.traversability:
            parts.append(self.traversability.summary())
        if self.stuck:
            parts.append(self.stuck.summary())
        if self.cross_validation:
            parts.append(self.cross_validation.summary())
        parts.append("═" * 50)
        return "\n".join(parts)


# ── Evaluator ─────────────────────────────────────────────────────────────────

class TerrainPredictorEvaluator:
    """
    Evaluates a fitted :class:`TerrainPredictor` against a held-out or
    cross-validated set of :class:`CellData` objects that carry ground truth.

    Parameters
    ----------
    predictor:
        A *fitted* TerrainPredictor instance.
    ground_truth_cells:
        Cells where ``real_traversability`` and ``is_stuck`` are known
        (i.e. ``is_visited == True``).
    """

    def __init__(
        self,
        predictor: TerrainPredictor,
        ground_truth_cells: list[CellData],
    ) -> None:
        if not ground_truth_cells:
            raise ValueError("ground_truth_cells must not be empty.")
        self.predictor = predictor
        self.cells = ground_truth_cells

    # ── Public API ────────────────────────────────────────────────────────────

    def full_report(self, k_folds: int = 5) -> EvaluationReport:
        """
        Compute traversability metrics, stuck metrics, and k-fold CV in one call.
        """
        return EvaluationReport(
            traversability=self.evaluate_traversability(),
            stuck=self.evaluate_stuck(),
            cross_validation=self.cross_validate(k=k_folds),
        )

    def evaluate_traversability(self) -> TraversabilityMetrics:
        """
        Score the GPR traversability model against ground-truth labels.

        Steps
        -----
        1. Run inference on every ground-truth cell.
        2. Compare ``traversability_estimate`` vs ``real_traversability``.
        3. Compute confidence calibration: a well-calibrated model's reported
           confidence should correlate with actual prediction accuracy.
        """
        self.predictor.update_prediction(self.cells)

        y_true = np.array([c.real_traversability for c in self.cells], dtype=float)
        y_pred = np.array([c.traversability_estimate for c in self.cells], dtype=float)
        confidences = np.array([c.confidence or 0.0 for c in self.cells], dtype=float)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred) ** 0.5
        r2 = r2_score(y_true, y_pred)
        mean_conf = float(np.mean(confidences))

        # Calibration: bin predictions by confidence, compare to actual accuracy.
        errors = np.abs(y_true - y_pred)
        threshold = float(np.median(errors))
        actual_accuracy = float(np.mean(errors <= threshold))
        calibration_error = abs(mean_conf - actual_accuracy)

        return TraversabilityMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mean_confidence=mean_conf,
            confidence_calibration_error=calibration_error,
        )

    def evaluate_stuck(self) -> StuckMetrics | None:
        """
        Score the GPC stuck model. Returns ``None`` if the stuck model was
        never fitted (no positive stuck examples in training data).
        """
        if not self.predictor._stuck_model_ready:
            return None

        self.predictor.update_prediction(self.cells)

        y_true = np.array([int(c.is_stuck or 0) for c in self.cells], dtype=int)
        y_prob = np.array([c.stuck_probability_estimate for c in self.cells], dtype=float)
        y_pred = (y_prob >= 0.5).astype(int)

        if len(np.unique(y_true)) < 2:
            # Cannot compute ROC-AUC without both classes present.
            return None

        return StuckMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            roc_auc=roc_auc_score(y_true, y_prob),
            confusion=confusion_matrix(y_true, y_pred),
        )

    def cross_validate(self, k: int = 5) -> CrossValidationReport:
        """
        K-fold cross-validation using only the ground-truth cells.

        For each fold, a fresh :class:`TerrainPredictor` is trained on the
        training split and evaluated on the held-out split, avoiding any
        data leakage through the scaler.
        """
        from TerrainPredictor import TerrainPredictor  # local import avoids circulars

        report = CrossValidationReport()
        cells = np.array(self.cells, dtype=object)
        kf = KFold(n_splits=min(k, len(cells)), shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(cells):
            train_cells = cells[train_idx].tolist()
            test_cells = cells[test_idx].tolist()

            # Train a fresh predictor on this fold's training split.
            fold_predictor = TerrainPredictor()
            fold_predictor.refit_predictor_model(train_cells, train_cells)

            if not fold_predictor._model_fitted:
                continue

            # ── Traversability ────────────────────────────────────────────────
            fold_predictor.update_prediction(test_cells)
            y_true_t = np.array([c.real_traversability for c in test_cells], dtype=float)
            y_pred_t = np.array([c.traversability_estimate for c in test_cells], dtype=float)

            report.trav_mae.append(mean_absolute_error(y_true_t, y_pred_t))
            report.trav_rmse.append(mean_squared_error(y_true_t, y_pred_t) ** 0.5)
            report.trav_r2.append(r2_score(y_true_t, y_pred_t))

            # ── Stuck (only when classifier was fitted) ───────────────────────
            if fold_predictor._stuck_model_ready:
                y_true_s = np.array([int(c.is_stuck or 0) for c in test_cells], dtype=int)
                y_prob_s = np.array([c.stuck_probability_estimate for c in test_cells], dtype=float)
                y_pred_s = (y_prob_s >= 0.5).astype(int)

                if len(np.unique(y_true_s)) > 1:
                    report.stuck_f1.append(f1_score(y_true_s, y_pred_s, zero_division=0))
                    report.stuck_roc_auc.append(roc_auc_score(y_true_s, y_prob_s))

        return report

    # ── Optional visualisation ────────────────────────────────────────────────

    def plot_diagnostics(self) -> None:
        """
        Render a 2×2 diagnostic figure (requires matplotlib).

          [0,0] Predicted vs actual traversability (scatter + ideal line)
          [0,1] Residuals vs confidence (should show no strong trend)
          [1,0] Stuck probability distribution per true class
          [1,1] Confusion matrix heat-map
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            raise ImportError("matplotlib is required for plot_diagnostics().")

        self.predictor.update_prediction(self.cells)

        y_true = np.array([c.real_traversability for c in self.cells], dtype=float)
        y_pred = np.array([c.traversability_estimate for c in self.cells], dtype=float)
        confs = np.array([c.confidence or 0.0 for c in self.cells], dtype=float)
        y_stuck = np.array([int(c.is_stuck or 0) for c in self.cells], dtype=int)
        y_prob_stuck = np.array([c.stuck_probability_estimate for c in self.cells], dtype=float)

        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
        fig.suptitle("TerrainPredictor — Diagnostic Plots", fontsize=14, fontweight="bold")

        # ── [0,0] Predicted vs actual ─────────────────────────────────────────
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.scatter(y_true, y_pred, alpha=0.6, edgecolors="k", linewidths=0.4, c=confs,
                    cmap="plasma", label="predictions")
        lims = [min(y_true.min(), y_pred.min()) - 0.05,
                max(y_true.max(), y_pred.max()) + 0.05]
        ax0.plot(lims, lims, "r--", lw=1.5, label="ideal")
        ax0.set_xlim(lims); ax0.set_ylim(lims)
        ax0.set_xlabel("Actual traversability"); ax0.set_ylabel("Predicted traversability")
        ax0.set_title("Predicted vs Actual")
        ax0.legend(fontsize=8)

        # ── [0,1] Residuals vs confidence ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 1])
        residuals = np.abs(y_true - y_pred)
        ax1.scatter(confs, residuals, alpha=0.6, edgecolors="k", linewidths=0.4)
        ax1.axhline(np.mean(residuals), color="r", linestyle="--", lw=1.5, label="mean residual")
        ax1.set_xlabel("Model confidence"); ax1.set_ylabel("|Residual|")
        ax1.set_title("Residuals vs Confidence\n(good model: residuals fall as confidence rises)")
        ax1.legend(fontsize=8)

        # ── [1,0] Stuck probability distribution ─────────────────────────────
        ax2 = fig.add_subplot(gs[1, 0])
        for label, colour in [(0, "steelblue"), (1, "tomato")]:
            mask = y_stuck == label
            if mask.sum() > 0:
                ax2.hist(y_prob_stuck[mask], bins=15, alpha=0.65, color=colour,
                         label=f"is_stuck={bool(label)}", density=True)
        ax2.axvline(0.5, color="k", linestyle="--", lw=1.2, label="decision boundary")
        ax2.set_xlabel("P(stuck)"); ax2.set_ylabel("Density")
        ax2.set_title("Stuck Probability Distribution")
        ax2.legend(fontsize=8)

        # ── [1,1] Confusion matrix ────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 1])
        cm = confusion_matrix(y_stuck, (y_prob_stuck >= 0.5).astype(int))
        im = ax3.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax3)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax3.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12)
        ax3.set_xticks([0, 1]); ax3.set_xticklabels(["Pred: safe", "Pred: stuck"])
        ax3.set_yticks([0, 1]); ax3.set_yticklabels(["True: safe", "True: stuck"])
        ax3.set_title("Stuck Confusion Matrix")

        plt.show()


