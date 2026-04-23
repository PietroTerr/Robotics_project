from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from CellData import CellData


class TerrainPredictor:
    """
    Maintains a GP regression model (traversability) and a GP classification
    model (stuck probability) and keeps all observed CellData objects up to date
    with the latest predictions.
    """

    def __init__(self) -> None:
        self._build_models()
        self.scaler = StandardScaler()
        self._model_fitted = False
        self._stuck_model_ready = False

    # ── Public API ────────────────────────────────────────────────────────────

    def update_predictor_model(
        self,
        observed_cells: list[CellData],
        visited_cells: list[CellData],
    ) -> None:
        """
        Re-fit both GP models on *visited* cells (ground truth available),
        then refresh predictions for all *observed* cells.
        """
        if not visited_cells:
            return

        x_train = self._extract_features(visited_cells)
        x_train = self.scaler.fit_transform(x_train)

        y_trav = np.array([c.real_traversability for c in visited_cells], dtype=float)
        y_stuck = np.array([c.is_stuck for c in visited_cells], dtype=int)

        self._fit_traversability_model(x_train, y_trav)
        self._fit_stuck_model(x_train, y_stuck)
        self._model_fitted = True

        self.update_prediction(observed_cells)

    def update_prediction(self, observed_cells: list[CellData]) -> None:
        """
        Push fresh predictions into every cell in *observed_cells*.
        Safe to call at any time; silently does nothing if the model is not
        yet fitted or the list is empty.
        """
        if not self._model_fitted or not observed_cells:
            return

        x_pred = self.scaler.transform(self._extract_features(observed_cells))

        self._predict_traversability(observed_cells, x_pred)
        if self._stuck_model_ready:
            self._predict_stuck(observed_cells, x_pred)

    # ── Private: model construction ───────────────────────────────────────────

    def _build_models(self) -> None:
        kernel = (
            ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            + WhiteKernel(noise_level=0.1)
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=0,
            normalize_y=True,
        )
        self.gpc = GaussianProcessClassifier(
            kernel=ConstantKernel() * Matern(nu=2.5)
        )

    # ── Private: fitting ──────────────────────────────────────────────────────

    def _fit_traversability_model(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        self.gpr.fit(x_train, y_train)

    def _fit_stuck_model(
        self, x_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        # Classification requires at least one positive and one negative example.
        if len(np.unique(y_train)) > 1:
            self.gpc.fit(x_train, y_train)
            self._stuck_model_ready = True
        else:
            self._stuck_model_ready = False

    # ── Private: prediction ───────────────────────────────────────────────────

    def _predict_traversability(
        self, cells: list[CellData], x_pred: np.ndarray
    ) -> None:
        """Write traversability_estimate and confidence into each cell."""
        means, stds = self.gpr.predict(x_pred, return_std=True)
        for cell, mean, std in zip(cells, means, stds):
            cell.traversability_estimate = float(mean)
            # Confidence approaches 0 for high uncertainty, 1 for low uncertainty.
            cell.confidence = float(1.0 - std / (std + 1.0))

    def _predict_stuck(
        self, cells: list[CellData], x_pred: np.ndarray
    ) -> None:
        """Write stuck_probability_estimate into each cell."""
        # BUG FIX: old code called zip(x_pred, proba) — x_pred is a numpy matrix
        # so each 'cell' in the loop was a feature row, not a CellData object,
        # making cell.stuck_probability_estimate raise AttributeError.
        proba = self.gpc.predict_proba(x_pred)  # shape (n, 2)
        for cell, p in zip(cells, proba):
            cell.stuck_probability_estimate = float(p[1])  # P(stuck=True)

    # ── Private: helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_features(cells: list[CellData]) -> np.ndarray:
        """Return an (n, 4) feature matrix for a list of CellData objects."""
        return np.array(
            [[c.texture, c.color, c.slope, c.uphill_angle] for c in cells],
            dtype=float,
        )