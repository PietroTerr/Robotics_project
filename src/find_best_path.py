import math

from data_management import CellData


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
                cost = compute_edge_cost(cell, terrain_map.grid[(nx_, ny_)], direction=(nx_ - x, ny_ - y))
                G.add_edge((x, y), (nx_, ny_), weight=cost)
    return G
    ## with this, we can use A* or Dijkstra to find the shortest path,
    ## but we need to make sure that the graph is connected, so we need to add all the cells to the graph,
    ##  which we do in the TerrainMap class