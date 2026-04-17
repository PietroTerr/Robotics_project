"""

"""


import math
import networkx as nx
from dataclasses import dataclass, field

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

    # --- Layer 2: Derived estimates (output of predictive model) ---
    traversability_estimate: float | None = None
    stuck_probability_estimate: float = 0.0
    confidence: float = 0.0

    # --- Layer 3: Exploration metadata ---
    observed_by: set[str] = field(default_factory=set)
    visit_count: int = 0
    last_updated: int = 0


class TerrainMap:
    """
    A sparse dictionary-based mapping system that houses coordinate-mapped `CellData` objects.
    Useful for partial explorations without maintaining large static memory overheads.
    """
    def __init__(self, width: int = 50, height: int = 50):
        self.grid: dict[tuple[int, int], CellData] = {}
        self.width = width
        self.height = height
        
    def get_cell(self, x: int, y: int) -> CellData:
        """Helper to get a cell, automatically generating it if it doesn't already exist."""
        coords = (int(x), int(y))
        if coords not in self.grid:
            self.grid[coords] = CellData()
        return self.grid[coords]


def get_neighbors_8(x: int, y: int):
    """Yields the 8 neighbors for a given grid cell."""
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            yield (x + dx, y + dy)


def compute_edge_cost(source_cell: CellData, target_cell: CellData, direction: tuple[int, int]) -> float:
    """
    Computes the cost to traverse from source_cell to target_cell.
    Handles defaults for unexplored cells.
    """
    # 1. Base traversability (Optimistic approach for unvisited: 1.0)
    t_est = target_cell.traversability_estimate if target_cell.traversability_estimate is not None else 1.0
    
    # Avoid div/0 and cap maximum speed to reasonable baseline
    t_est = max(0.001, t_est)
    base_cost = 1.0 / t_est
    
    # 2. Confidence term (Penalty lambda)
    lam = 2.0
    confidence_penalty = lam * (1.0 - target_cell.confidence)
    
    # 3. Slope and direction logic 
    slope_factor = 1.0
    if target_cell.slope is not None and target_cell.uphill_angle is not None and target_cell.slope > 0:
        dir_angle = math.atan2(direction[1], direction[0])
        # Vector alignment to determine if running against or with the slope
        alignment = math.cos(dir_angle - target_cell.uphill_angle)
        
        if alignment > 0: 
            # Uphill penalty
            slope_factor = 1.0 + (target_cell.slope / 30.0) * alignment
        else:
            # Downhill boost
            slope_factor = 1.0 - 0.2 * abs(alignment)
            
    # 4. Stuck probability penalty
    stuck_penalty = 1000.0 if target_cell.stuck_probability_estimate > 0.5 else 0.0
    
    return (base_cost * slope_factor) + confidence_penalty + stuck_penalty


def build_weighted_graph(terrain_map: TerrainMap) -> nx.DiGraph:
    """
    Lifts the Layered grid map into a standard Networkx graph framework to be used in A* or Dijkstra pathfinding.
    """
    G = nx.DiGraph()
    for (x, y), cell in terrain_map.grid.items():
        for (nx_, ny_) in get_neighbors_8(x, y):
            if (nx_, ny_) in terrain_map.grid:
                cost = compute_edge_cost(cell, terrain_map.grid[(nx_, ny_)], direction=(nx_-x, ny_-y))
                G.add_edge((x, y), (nx_, ny_), weight=cost)
    return G 
    ## with this, we can use A* or Dijkstra to find the shortest path, 
    ## but we need to make sure that the graph is connected, so we need to add all the cells to the graph,
    ##  which we do in the TerrainMap class