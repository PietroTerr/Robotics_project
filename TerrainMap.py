import math

from TerrainPredictor import TerrainPredictor
from motion import Rover, Scout
from CellData import CellData

from src.map_api_core import TerrainObservation

class TerrainMap:
    """
    A sparse dictionary-based mapping system that houses coordinate-mapped `CellData` objects.
    Useful for partial explorations without maintaining large static memory overheads.
    """
    def __init__(self, width: int = 50, height: int = 50):
        self.grid: dict[tuple[int, int], CellData] = {}
        self.width = width
        self.height = height
        self.grid_size = (self.width, self.height)
        self.observed_cells = set() # cells observed by perceive
        self.visited_cells = set() # cells visited by scout or rover
        self.terrain_predictor = TerrainPredictor()

    def get_cell(self, x: int, y: int) -> CellData:
        """Helper to get a cell, automatically generating it if it doesn't already exist."""
        coords = (int(x), int(y))
        if coords not in self.grid:
            self.grid[coords] = CellData(coords[0], coords[1])
        return self.grid[coords]

    def store_observation(self, obs: list[TerrainObservation]):
        if obs is not None:
            for ob in obs:
                cell = self.get_cell(ob.x, ob.y)
                self.observed_cells.add(cell)
                cell.set_texture(ob.features.get("texture"))
                cell.set_color(ob.features.get("color"))
                cell.set_slope(ob.features.get("slope"))
                cell.set_uphill_angle(ob.features.get("uphill_angle"))


    def __store_stuck_information(self, x: int, y: int, get_stuck):
        cell = self.get_cell(x, y)
        if get_stuck is not None:
            cell.is_stuck = get_stuck


    def store_movement_information(self, movement_information: dict):
        # --- Update known cells from agent feedback ---
        new_cells = False
        for agent, info in movement_information.items():
            if isinstance(agent, (Rover, Scout)):
                cell = self.get_cell(agent.x, agent.y)
                if cell in self.visited_cells:
                    continue
                new_cells = True
                self.visited_cells.add(cell)
                self.__store_stuck_information(agent.x, agent.y, info["is_stuck"])
                # Only compute real_traversability once — it's a fixed terrain property
                if cell.real_traversability is None:
                    slope_f = _directional_slope_factor(
                        cell.slope or 0, cell.uphill_angle or 0, info["heading"]
                    )
                    commanded = info["command_velocity"]
                    new_trav = info["actual_velocity"] / (commanded * slope_f + 1e-9)
                    cell.real_traversability = max(0.0, min(1.0, new_trav))
        if new_cells:
            self.terrain_predictor.update_predictor(self.observed_cells, self.visited_cells)


def _directional_slope_factor(slope: float,
    uphill_angle: float,
    movement_orientation: float,
) -> float:
    uphill_penalty: float = 1.0
    downhill_boost: float = 0.2
    slope_max_degrees_for_velocity: float = 30.0

    ux, uy = math.cos(uphill_angle), math.sin(uphill_angle)
    mx, my = math.cos(movement_orientation), math.sin(movement_orientation)
    alignment = ux * mx + uy * my
    slope_norm = _clamp(slope / slope_max_degrees_for_velocity, 0.0, 1.0)
    signed_grade = slope_norm * alignment

    if signed_grade >= 0:
        return 1.0 - uphill_penalty * signed_grade

    downhill_alignment = -signed_grade
    return 1.0 + downhill_boost * downhill_alignment

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

