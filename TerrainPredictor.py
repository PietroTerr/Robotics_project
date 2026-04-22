import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class TerrainPredictor:

    def __init__(self,):
        self._create_models()
        self.stuck_model_ready = False
        self.scaler = StandardScaler()

    def update_predictor(self, observed_cells, visited_cells):
        self._fit_traversability_model(visited_cells)
        self._fit_stuck_model(visited_cells)

        self._predict_traversability(observed_cells)
        if self.stuck_model_ready:
            self._predict_stuck_model(observed_cells)


    def _create_models(self):
        # ── 1. Define the kernel ──────────────────────────────────────────────────────
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5) + WhiteKernel(noise_level=0.1)
        # ── 2. Create models ───────────────────────────────────────────────
        self.gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True)
        self.gpc = GaussianProcessClassifier(kernel=ConstantKernel() * Matern(nu=2.5))

    # Traversability model (regression)
    def _fit_traversability_model(self, visited_cells):
        x_train = np.array([[c.texture, c.color, c.slope, c.uphill_angle]
                     for c in visited_cells])
        x_train = self.scaler.fit_transform(x_train) # Scale data
        y_train = np.array([c.real_traversability for c in visited_cells])
        self.gpr.fit(x_train, y_train)


    def _predict_traversability(self, observed_cells):
        x_pred = np.array([[c.texture, c.color, c.slope, c.uphill_angle]
                           for c in observed_cells])
        x_pred = self.scaler.transform(x_pred)
        means, stds = self.gpr.predict(x_pred, return_std=True)
        for cell, mean, std in zip(observed_cells, means, stds):
            cell.traversability_estimate,cell.confidence = _inference_regression(mean, std)

    # Stuck model (classification)
    def _fit_stuck_model(self, visited_cells):
        x_train = np.array([[c.texture, c.color, c.slope, c.uphill_angle]
                            for c in visited_cells])
        x_train = self.scaler.fit_transform(x_train)
        y_train = np.array([c.is_stuck for c in visited_cells])

        # Only fit if we have at least one stuck event and one safe event
        if len(np.unique(y_train)) > 1:
            self.gpc.fit(x_train, y_train)
            self.stuck_model_ready = True
        else:
            self.stuck_model_ready = False

    def _predict_stuck_model(self, observed_cells):
        x_pred = np.array([[c.texture, c.color, c.slope, c.uphill_angle]
                           for c in observed_cells])
        x_pred = self.scaler.transform(x_pred)
        proba = self.gpc.predict_proba(x_pred)  # shape (n, 2)
        for cell, p in zip(observed_cells, proba):
            cell.stuck_probability_estimate = p[1]  # Probability of being stuck (class 1)



def _inference_regression(mean: float, std: float):
        return mean, 1.0 - (std / (std + 1e-9))
def _inference_binary_classification(probability_estimate: float):
    return False if probability_estimate < 0.5 else True