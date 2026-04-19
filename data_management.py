from dataclasses import dataclass, field

from src.map_api_core import TerrainObservation


@dataclass
class CellData:
    """
    Representation of a single grid cell within the TerrainMap. 
    Divided cleanly into layers for raw sensory input, analytical estimates, and exploration tracking.
    """
    # --- Layer 1: Raw observations from agents ---
    texture: float | None = None
    color: float | None = None
    slope: float | None = None
    uphill_angle: float | None = None

    is_stuck: bool | None= None # None mean that we still don't have any information

    # --- Layer 2: Derived estimates (output of predictive model) ---
    traversability_estimate: float | None = None
    stuck_probability_estimate: float = 0.0
    confidence: float = 0.0

    def set_texture(self, texture: float):
        if self.texture is None:
            self.texture = texture
        else:
            self.texture = (self.texture + texture)/2

    def set_color(self, color: float):
        if self.color is None:
            self.color = color
        else:
            self.color = (self.color + color)/2
    def set_slope(self, slope: float):
        if self.slope is None:
            self.slope = slope
        else:
            self.slope = (self.slope + slope)/2
    def set_uphill_angle(self, uphill_angle: float | None):
        if uphill_angle is None:
            self.uphill_angle = uphill_angle
        else:
            self.uphill_angle = (self.uphill_angle + uphill_angle)/2


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

    def get_cell(self, x: int, y: int) -> CellData:
        """Helper to get a cell, automatically generating it if it doesn't already exist."""
        coords = (int(x), int(y))
        if coords not in self.grid:
            self.grid[coords] = CellData()
        return self.grid[coords]

    def store_observation(self, obs: list[TerrainObservation]):
        if obs is not None:
            for ob in obs:
                cell = self.get_cell(ob.x, ob.y)
                cell.set_texture(ob.features.get("texture"))
                cell.set_color(ob.features.get("color"))
                cell.set_slope(ob.features.get("slope"))
                cell.set_uphill_angle(ob.features.get("uphill_angle"))

    def __store_stuck_information(self, x: int, y: int, get_stuck):
        cell = self.get_cell(x,y)
        if get_stuck is not None:
            cell.is_stuck = get_stuck


    def refresh_estimation(self,movement_information):
        """Placeholder for a method that would run the predictive model to update traversability and stuck probability estimates based on the current state of the map. and save stuck event"""
        pass



