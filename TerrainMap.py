from __future__ import annotations

import math

from CellData import CellData
from TerrainPredictor import TerrainPredictor


class TerrainMap:
    """
    A sparse dictionary-based terrain map that stores coordinate-keyed
    :class:`CellData` objects and keeps them enriched with model predictions.

    Only cells that have actually been observed or visited are stored,
    avoiding the memory cost of a dense grid.
    """

    def __init__(self, width: int = 50, height: int = 50) -> None:
        self.grid: dict[tuple[int, int], CellData] = {}
        self.width = width
        self.height = height
        self.grid_size = (self.width, self.height)
        self.terrain_predictor = TerrainPredictor()

    # ── Cell access ───────────────────────────────────────────────────────────

    def get_cell(self, x: int, y: int) -> CellData:
        """Return the cell at (x, y), creating it lazily if needed."""
        coords = (int(x), int(y))
        if coords not in self.grid:
            self.grid[coords] = CellData(coords[0], coords[1])
        return self.grid[coords]

    # ── Cell queries ──────────────────────────────────────────────────────────

    def get_observed_cells(self) -> list[CellData]:
        return [cell for cell in self.grid.values() if cell.is_observed]

    def get_visited_cells(self) -> list[CellData]:
        return [cell for cell in self.grid.values() if cell.is_visited]

    # ── Ingestion: sensor observations ───────────────────────────────────────

    def update_map(self, new_map_information: dict) -> None:
        new_observation = self._store_observation(new_map_information)
        new_visited = self._store_movement_information(new_map_information)

        if new_visited: #if new visited cell train the model again
            self.terrain_predictor.update_predictor_model(
                self.get_observed_cells(),
                self.get_visited_cells(),
            )
        elif new_observation: # if just new observation wwe just comute the estimation
            self.terrain_predictor.update_prediction(self.get_observed_cells())


    def _store_observation(self, obs) -> bool:
            """
            Record sensor observations for a batch of cells.
            """

            # Build a set of already-observed coords for O(1) membership tests.
            already_observed: set[tuple[int, int]] = {
                (c.x, c.y) for c in self.get_observed_cells()
            }

            new_observations = False
            for x,y,info in obs:
                coords = (int(x), int(y))
                if coords in already_observed:
                    continue
                cell = self.get_cell(coords[0], coords[1])
                cell.set_texture(info["texture"])
                cell.set_color(info["color"])
                cell.set_slope(info["slope"])
                cell.set_uphill_angle(info["uphill_angle"])
                new_observations = True
            return new_observations


    # ── Ingestion: movement / traversal feedback ──────────────────────────────

    def _store_movement_information(self, movement_information: dict):
        """
        Process post-movement telemetry from all agents, compute ground-truth
        traversability
        """
        visited_set: set[tuple[int, int]] = {
            (c.x, c.y) for c in self.get_visited_cells()
        }
        new_cells = False

        for x, y, info in movement_information:

            coords = (int(x), int(y))
            if coords in visited_set:
                continue

            # look for visited cells
            if "heading" not in movement_information[coords]:
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
    Return a scalar in (0, 1.2] that adjusts commanded velocity based on the
    angle between the agent's heading and the uphill direction.

    - Going directly uphill  → factor approaches (1 - uphill_penalty) = 0.
    - Going directly downhill → factor approaches (1 + downhill_boost) = 1.2.
    - Perpendicular to slope  → factor = 1.0 (no adjustment).
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


# ── Module-level utilities ────────────────────────────────────────────────────


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))