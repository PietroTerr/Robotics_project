import math
import networkx as nx
from src.map_api_core import TerrainObservation
from CellData import CellData
from TerrainPredictor import TerrainPredictor

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
        self.terrain_predictor = TerrainPredictor()  # ML model — persists across phases

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
                cell.set_texture(ob.features.get("texture"))
                cell.set_color(ob.features.get("color"))
                cell.set_slope(ob.features.get("slope"))
                cell.set_uphill_angle(ob.features.get("uphill_angle"))

    def __store_stuck_information(self, x: int, y: int, get_stuck):
        cell = self.get_cell(x, y)
        if get_stuck is not None:
            cell.is_stuck = get_stuck

    def refresh_estimation(self, movement_information=None):
        """
        Trains the TerrainPredictor GP model on cells physically visited by the scout,
        then propagates traversability and stuck-probability estimates to all observed cells.
        The graph builder reads those ML predictions for A* edge costs.
        """
        observed_cells = [c for c in self.grid.values() if c.is_observed]
        visited_cells  = [c for c in self.grid.values() if c.is_visited]

        if not visited_cells:
            print("  [WARN] No physically visited cells — TerrainPredictor cannot be trained yet.")
            return

        print(f"  [ML] Training TerrainPredictor: {len(visited_cells)} visited / "
              f"{len(observed_cells)} observed cells.")

        # Reuse the existing predictor instance (preserves kernel hyperparameters)
        self.terrain_predictor.update_predictor_model(
            observed_cells=observed_cells,
            visited_cells=visited_cells,
        )
        print(f"  [ML] GP regression fitted: {self.terrain_predictor._model_fitted} | "
              f"stuck classifier ready: {self.terrain_predictor._stuck_model_ready}")



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
    Uses ML predictions (traversability_estimate, confidence, stuck_probability_estimate)
    to steer A* away from dangerous or uncertain terrain.
    """
    # 1. Base traversability cost: cells with low estimate cost MUCH more
    #    If traversability_estimate is None (unobserved), treat it as 0.5 (neutral guess)
    t_est = target_cell.traversability_estimate if target_cell.traversability_estimate is not None else 0.5
    t_est = max(0.01, min(1.0, t_est))

    # Exponential penalty: trav=1.0 → cost=1.0, trav=0.5 → cost=4.0, trav=0.1 → cost=100
    base_cost = (1.0 / t_est) ** 2

    # 2. Uncertainty penalty: low confidence = large penalty (encourages rover to use well-explored paths)
    #    confidence=1.0 → penalty=0, confidence=0.0 → penalty=5.0
    conf = target_cell.confidence if target_cell.confidence is not None else 0.0
    uncertainty_penalty = 5.0 * (1.0 - conf)

    # 3. Slope and direction penalty
    slope_factor = 1.0
    if target_cell.slope is not None and target_cell.uphill_angle is not None and target_cell.slope > 0:
        dir_angle = math.atan2(direction[1], direction[0])
        alignment = math.cos(dir_angle - target_cell.uphill_angle)
        if alignment > 0:
            slope_factor = 1.0 + (target_cell.slope / 30.0) * alignment
        else:
            slope_factor = max(0.8, 1.0 - 0.2 * abs(alignment))

    # 4. Stuck probability: hard penalty (cells where scout got stuck are near-impassable)
    sp = target_cell.stuck_probability_estimate
    if sp > 0.5:
        stuck_penalty = 10_000.0   # near-impossible to traverse
    elif sp > 0.2:
        stuck_penalty = 500.0 * sp  # scaled soft penalty
    else:
        stuck_penalty = 0.0

    # 5. Real stuck flag penalty (discovered during rover traversal or scout pass)
    if target_cell.is_stuck:
        stuck_penalty = max(stuck_penalty, 10_000.0)

    return (base_cost * slope_factor) + uncertainty_penalty + stuck_penalty



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