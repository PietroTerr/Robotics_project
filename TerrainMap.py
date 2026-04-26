
from __future__ import annotations

import math

from CellData import CellData
from TerrainPredictor import TerrainPredictor
from TerrainGraph import TerrainGraph


class TerrainMap:
    """
    Terrain state manager that fuses sensing, traversal feedback, prediction, and path graph updates.

    Responsibilities
    ----------------
    1. Store per-cell terrain data as `CellData`.
    2. Ingest new sensor observations (texture/color/slope/uphill direction).
    3. Ingest movement telemetry to compute ground-truth traversability for visited cells.
    4. Maintain GP-based terrain predictions via `TerrainPredictor`.
    5. Keep `TerrainGraph` synchronized for planning (rover/scout/drone graph views).

    Notes
    -----
    - The implementation pre-populates the full grid in `initialize_map`.
    - GP refits are throttled using `REFIT_INTERVAL` to avoid expensive O(n^3) updates on every step.
    """

    # GP refit is O(n³) — only refit after this many new visited cells.
    # Tune this value to balance prediction accuracy vs. computation cost.
    REFIT_INTERVAL: int = 5

    def __init__(self, width: int = 50, height: int = 50) -> None:
        """
        Initialize map storage, predictor, and planning graph.

        Parameters
        ----------
        width : int, default=50
            Number of columns in the map.
        height : int, default=50
            Number of rows in the map.
        """
        self.grid: dict[tuple[int, int], CellData] = {}
        self.width = width
        self.height = height
        self.grid_size = (self.width, self.height)
        self.terrain_predictor = TerrainPredictor()
        self.terrain_graph = TerrainGraph()

        # Throttle counter: number of new visited cells since the last GP refit
        self._new_visited_since_refit: int = 0

        self._initialize_map()

    def _initialize_map(self):
        """
        Pre-populate every coordinate with an empty `CellData`.

        Also registers each initial cell in `terrain_graph`.
        Since fresh cells are neither observed nor visited, graph internals
        decide whether and where each cell is included.
        """
        for x in range(self.width):
            for y in range(self.height):
                self.grid[(x, y)] = CellData(x, y)
                self.terrain_graph.add_cell(self.grid[(x, y)])
    # ── Cell access ───────────────────────────────────────────────────────────

    def get_cell(self, x: int, y: int) -> CellData:
        """
        Return the `CellData` at `(x, y)`, creating it if absent.

        Parameters
        ----------
        x : int
            X coordinate.
        y : int
            Y coordinate.

        Returns
        -------
        CellData
            Cell object associated with integerized coordinates.
        """
        coords = (int(x), int(y))
        if coords not in self.grid:
            self.grid[coords] = CellData(coords[0], coords[1])
        return self.grid[coords]

    # ── Cell queries ──────────────────────────────────────────────────────────

    def get_observed_cells(self) -> list[CellData]:
        """
        Return all cells with complete observation fields populated.

        Returns
        -------
        list[CellData]
            Cells where `cell.is_observed == True`.
        """
        return [cell for cell in self.grid.values() if cell.is_observed]

    def get_visited_cells(self) -> list[CellData]:
        """
        Return all cells that have traversal-derived ground-truth data.

        Returns
        -------
        list[CellData]
            Cells where `cell.is_visited == True`.
        """
        return [cell for cell in self.grid.values() if cell.is_visited]

    # ── Ingestion: sensor observations ───────────────────────────────────────

    def update_map(self, observations: dict, movement: dict) -> None:
        """
        Ingest one simulation tick of observations and movement telemetry.

        Update policy
        -------------
        - If new visited cells are found:
            - Increment refit throttle counter.
            - Refit GP model every `REFIT_INTERVAL` visited updates.
            - Otherwise perform cheap prediction refresh.
            - Rewire graph edges for all visited cells.
        - Else if only new observations exist:
            - Refresh GP predictions.
            - Add observed cells to graph.

        Parameters
        ----------
        observations : dict
            Output from sensing agents, keyed by `(x, y)` with feature values.
        movement : dict
            Output from agent stepping, keyed by `(x, y)` with motion telemetry.
        """
        new_observation = self._store_observation(observations)
        new_visited = self._store_movement_information(movement)

        visited_cells = self.get_visited_cells()
        observed_cells = self.get_observed_cells()

        if new_visited:
            self._new_visited_since_refit += 1

            if self._new_visited_since_refit % self.REFIT_INTERVAL == 0:
                # Full GP refit on ground-truth data, then refresh all estimates
                self.terrain_predictor.refit_predictor_model(observed_cells, visited_cells)
            else:
                # Cheap update: push current GP predictions into observed cells
                self.terrain_predictor.update_prediction(observed_cells)

            # Update graph for every visited cell.
            for cell in visited_cells:
                self.terrain_graph.update_cell(cell)

        elif new_observation:
            # New sensor data only — refresh GP estimates, add new cells to graph
            self.terrain_predictor.update_prediction(observed_cells)
            for cell in observed_cells:
                self.terrain_graph.add_cell(cell)

    def _store_observation(self, obs) -> bool:
        """
        Store newly perceived terrain features into cells.

        Expected per-cell fields in `obs[(x, y)]`:
        - `"texture"`
        - `"color"`
        - `"slope"`
        - `"uphill_angle"`

        Already-observed cells are skipped to avoid redundant writes.

        Parameters
        ----------
        obs : dict
            Observation mapping from coordinates to feature dict.

        Returns
        -------
        bool
            True if at least one cell became newly observed in this batch.
        """
        new_observations = False
        for (x, y), info in obs.items():
            coords = (int(x), int(y))
            cell = self.get_cell(coords[0], coords[1])
            if self.grid.get(coords).is_observed:
                continue
            cell.set_texture(info["texture"])
            cell.set_color(info["color"])
            cell.set_slope(info["slope"])
            cell.set_uphill_angle(info["uphill_angle"])
            new_observations = True
        return new_observations

    # ── Ingestion: movement / traversal feedback ──────────────────────────────

    def _store_movement_information(self, movement_information: dict):
        """
        Process movement telemetry and compute real traversability for new visited cells.

        A cell is considered for update only if:
        - It was not already visited, and
        - Telemetry contains `"heading"` (filters non-ground or non-motion payloads).

        Traversability model
        --------------------
        `raw_trav = actual_velocity / (command_velocity * slope_factor + 1e-9)`

        The result is clamped to `[0.0, 1.0]`.

        Parameters
        ----------
        movement_information : dict
            Mapping `(x, y)` -> telemetry dict with at least:
            - `"heading"`
            - `"is_stuck"`
            - `"command_velocity"`
            - `"actual_velocity"`

        Returns
        -------
        bool
            True if at least one new cell received movement-derived updates.
        """
        visited_set: set[tuple[int, int]] = {
            (c.x, c.y) for c in self.get_visited_cells()
        }
        new_cells = False

        for (x, y), info in movement_information.items():

            coords = (int(x), int(y))
            if coords in visited_set:
                continue

            # Look only at entries containing movement heading.
            if "heading" not in info:
                continue

            cell = self.get_cell(x, y)
            cell.set_is_stuck(info["is_stuck"])

            # real_traversability is a fixed terrain property — compute once.
            if cell.real_traversability is None:
                slope_f = _directional_slope_factor(
                    cell.slope or 0.0,
                    cell.uphill_angle or 0.0,
                    info["heading"],
                )
                commanded = info["command_velocity"]
                raw_trav = info["actual_velocity"] / (commanded * slope_f + 1e-9)
                cell.set_real_traversability(max(0.0, min(1.0, raw_trav)))
            new_cells = True
        return new_cells

    def get_grid_snapshot(self) -> dict[tuple[int, int], CellData]:
        """
        Return a shallow snapshot of current grid state.

        Returns
        -------
        dict[tuple[int, int], CellData]
            New dict object containing existing `CellData` references.
        """
        return {coords: cell for coords, cell in self.grid.items()}


# ── Module-level utilities ────────────────────────────────────────────────────

def _directional_slope_factor(
    slope: float,
    uphill_angle: float,
    movement_orientation: float,
    *,
    uphill_penalty: float = 1.0,
    downhill_boost: float = 0.2,
    slope_max_degrees: float = 30.0,
) -> float:
    """
    Compute direction-aware slope multiplier for commanded velocity normalization.

    Interpretation
    --------------
    - Positive alignment means moving uphill: factor decreases toward 0.
    - Negative alignment means moving downhill: factor increases up to 1.2.
    - Orthogonal movement yields ~1.0.

    Parameters
    ----------
    slope : float
        Local slope magnitude in degrees.
    uphill_angle : float
        Global orientation (radians) of steepest uphill direction.
    movement_orientation : float
        Agent heading (radians).
    uphill_penalty : float, default=1.0
        Strength of uphill slowdown.
    downhill_boost : float, default=0.2
        Strength of downhill speedup.
    slope_max_degrees : float, default=30.0
        Normalization cap for slope magnitude.

    Returns
    -------
    float
        Multiplicative factor typically in `(0, 1.2]`.
    """
    ux, uy = math.cos(uphill_angle), math.sin(uphill_angle)
    mx, my = math.cos(movement_orientation), math.sin(movement_orientation)
    alignment = ux * mx + uy * my                            # dot product ∈ [-1, 1]
    slope_norm = _clamp(slope / slope_max_degrees, 0.0, 1.0)
    signed_grade = slope_norm * alignment

    if signed_grade >= 0:
        return 1.0 - uphill_penalty * signed_grade

    downhill_alignment = -signed_grade
    return 1.0 + downhill_boost * downhill_alignment


def _clamp(value: float, low: float, high: float) -> float:
    """
    Clamp a numeric value to the inclusive interval [low, high].
    """
    return max(low, min(high, value))