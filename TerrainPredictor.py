
from __future__ import annotations

import warnings

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

from CellData import CellData


class TerrainPredictor:
    """
    Gaussian-process-based terrain model manager.

    This class maintains two probabilistic models trained from visited cells:

    1) Traversability regressor (`gpr`)
       Predicts continuous traversability from observed terrain features.

    2) Stuck classifier (`gpc`)
       Predicts probability that a cell is stuck/non-traversable.

    The predictor writes model outputs directly into `CellData` objects
    (in-place updates), so downstream modules can consume updated estimates
    without extra conversion steps.

    Feature vector schema
    ---------------------
    For each cell, the model uses 4 features:

    `[texture, color, slope, uphill_angle]`
    """

    # Silence sklearn warnings for cleaner simulation output.
    warnings.filterwarnings('ignore', module='sklearn')

    def __init__(self) -> None:
        """
        Initialize GP models and preprocessing state.

        Attributes
        ----------
        gpr : GaussianProcessRegressor
            Model for continuous traversability prediction.
        gpc : GaussianProcessClassifier
            Model for binary stuck classification.
        scaler : StandardScaler
            Feature normalization fitted on visited-cell training features.
        _model_fitted : bool
            True once the traversability model has been fitted at least once.
        _stuck_model_ready : bool
            True only when classifier has enough class diversity to train.
        """
        self._build_models()
        self.scaler = StandardScaler()
        self._model_fitted = False
        self._stuck_model_ready = False

    # ── Public API ────────────────────────────────────────────────────────────

    def refit_predictor_model(
            self,
            observed_cells: list[CellData],
            visited_cells: list[CellData],
    ) -> None:
        """
        Refit GP models from ground-truth visited data, then refresh observed predictions.

        Training uses only `visited_cells` because they contain ground-truth labels:
        - `real_traversability` for regression
        - `is_stuck` for classification

        Parameters
        ----------
        observed_cells : list[CellData]
            Cells to receive refreshed predictions after fitting.
        visited_cells : list[CellData]
            Cells with known labels used to train both models.

        Notes
        -----
        - If `visited_cells` is empty, method returns immediately.
        - Classifier training is skipped if only one class is present.
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
        Push latest model predictions into all observed cells.

        Parameters
        ----------
        observed_cells : list[CellData]
            Cells whose prediction fields should be updated.

        Behavior
        --------
        - Safe no-op if model has not been fitted yet.
        - Safe no-op if `observed_cells` is empty.
        - Always updates traversability + confidence.
        - Updates stuck probability only if classifier is train-ready.
        """
        if not self._model_fitted or not observed_cells:
            return

        x_pred = self.scaler.transform(self._extract_features(observed_cells))

        self._predict_traversability(observed_cells, x_pred)
        if self._stuck_model_ready:
            self._predict_stuck(observed_cells, x_pred)

    # ── Private: model construction ───────────────────────────────────────────

    def _build_models(self) -> None:
        """
        Construct GP regression/classification models with Matérn-based kernels.

        Kernel choices
        --------------
        - Regressor:
            Constant * Matern(nu=2.5) + WhiteKernel(noise)
        - Classifier:
            Constant * Matern(nu=2.5)

        Design rationale
        ----------------
        Matérn kernels provide flexible smoothness for terrain-like signals,
        while WhiteKernel in regression accounts for observation noise.
        """
        kernel = (
                ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
                + WhiteKernel(noise_level=0.1)
        )
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
        )
        self.gpc = GaussianProcessClassifier(
            kernel=ConstantKernel() * Matern(nu=2.5)
        )

    # ── Private: fitting ──────────────────────────────────────────────────────

    def _fit_traversability_model(
            self, x_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """
        Fit traversability regressor.

        Parameters
        ----------
        x_train : np.ndarray
            Scaled training matrix of shape `(n_samples, 4)`.
        y_train : np.ndarray
            Traversability targets of shape `(n_samples,)`.
        """
        self.gpr.fit(x_train, y_train)

    def _fit_stuck_model(
            self, x_train: np.ndarray, y_train: np.ndarray
    ) -> None:
        """
        Fit stuck classifier if class diversity is sufficient.

        Parameters
        ----------
        x_train : np.ndarray
            Scaled training matrix of shape `(n_samples, 4)`.
        y_train : np.ndarray
            Binary stuck labels of shape `(n_samples,)`.

        Notes
        -----
        GP classification requires at least two distinct labels.
        If all labels are identical, classifier fitting is skipped and
        `_stuck_model_ready` is set to `False`.
        """
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
        """
        Predict traversability and confidence, then write into each `CellData`.

        Parameters
        ----------
        cells : list[CellData]
            Target cells to update in-place.
        x_pred : np.ndarray
            Scaled feature matrix of shape `(n_samples, 4)`.

        Writes
        ------
        - `cell.traversability_estimate` <- predictive mean
        - `cell.confidence` <- monotonic transform of predictive std:
            `1 - std / (std + 1)` in (0, 1], where higher is more confident.
        """
        means, stds = self.gpr.predict(x_pred, return_std=True)
        for cell, mean, std in zip(cells, means, stds):
            cell.traversability_estimate = float(mean)
            # Confidence approaches 0 for high uncertainty, 1 for low uncertainty.
            cell.confidence = float(1.0 - std / (std + 1.0))

    def _predict_stuck(
            self, cells: list[CellData], x_pred: np.ndarray
    ) -> None:
        """
        Predict stuck probabilities and write into each `CellData`.

        Parameters
        ----------
        cells : list[CellData]
            Target cells to update in-place.
        x_pred : np.ndarray
            Scaled feature matrix of shape `(n_samples, 4)`.

        Writes
        ------
        - `cell.stuck_probability_estimate` <- P(stuck=True)
        """
        # BUG FIX NOTE:
        # Old logic zipped `(x_pred, proba)` by mistake, so loop variable `cell`
        # held a feature row instead of a `CellData` object.
        proba = self.gpc.predict_proba(x_pred)  # shape (n, 2)
        for cell, p in zip(cells, proba):
            cell.stuck_probability_estimate = float(p[1])  # P(stuck=True)

    # ── Private: helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_features(cells: list[CellData]) -> np.ndarray:
        """
        Build feature matrix from cells.

        Parameters
        ----------
        cells : list[CellData]
            Input cells.

        Returns
        -------
        np.ndarray
            Matrix with shape `(n_cells, 4)` and columns:
            `[texture, color, slope, uphill_angle]`.

        Notes
        -----
        This assumes all four features are populated (non-None) for each cell.
        """
        return np.array(
            [[c.texture, c.color, c.slope, c.uphill_angle] for c in cells],
            dtype=float,
        )