"""
TerrainGraph
============
Maintains a single directed, weighted adjacency graph over the full terrain grid.
All cells (unobserved, observed, visited, stuck) live in one graph; traversability
is resolved by a confidence-weighted formula so agents naturally prefer well-known,
safe terrain without needing separate graph structures.

Edge weight formula (src → dst):
    w = (1 / effective_trav(dst)) × slope_factor(src→dst) × diagonal_mult

effective_trav resolution (in priority order):
    1. is_stuck                       → 0.01  (passable but extremely costly)
    2. real_traversability available  → real_traversability  (visited ground truth)
    3. is_observed                    → confidence * estimate + (1-confidence) * pessimistic_default
                                        then multiplied by (1 - stuck_probability_estimate)
    4. unobserved                     → pessimistic_default  (tunable, default 0.3)

Agent views (via _PenalizedView):
    rover  → raw complete_graph, no penalty   (confidence weights guide preference)
    scout  → complete_graph, visited cells penalised  (encourages new exploration)
    drone  → complete_graph, observed cells penalised (encourages new scanning)

The revisit penalty is NOT baked into stored weights — applied lazily by
_PenalizedView so the same graph is safely shared across agents.
"""

from __future__ import annotations

import math
from typing import Iterator, Literal

from CellData import CellData

# ── Constants ─────────────────────────────────────────────────────────────────

SQRT2 = math.sqrt(2)
_DIRECTIONS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]
_STUCK_TRAVERSABILITY = 0.01   # kept passable so scout/drone are never hard-blocked


# ── Live penalty view ─────────────────────────────────────────────────────────

class _PenalizedView:
    """
    A lazy, read-only view over an adjacency dict that multiplies
    the edge weight by `penalty` for every *destination* node that
    belongs to `penalty_nodes`.

    No data is ever copied — the view reads directly from the live graph
    and the live penalty set, so any subsequent add/update calls are
    immediately reflected.

    The interface is intentionally minimal: only the operations that
    standard A* implementations need are exposed.
    """

    def __init__(
            self,
            graph: dict[tuple, dict[tuple, float]],
            penalty_nodes: set[tuple],
            penalty: float,
    ) -> None:
        self._graph = graph
        self._penalty_nodes = penalty_nodes
        self._penalty = penalty

    # ── Mapping interface expected by A* ──────────────────────────────────────

    def __contains__(self, node: object) -> bool:
        return node in self._graph

    def __iter__(self) -> Iterator[tuple]:
        return iter(self._graph)

    def __len__(self) -> int:
        return len(self._graph)

    def keys(self):
        return self._graph.keys()

    def __getitem__(self, node: tuple) -> dict[tuple, float]:
        """
        Return the neighbour→weight dict for `node`, with penalty applied
        to any destination that is in the penalty set.

        Fast path: if none of this node's neighbours are penalised, return
        the raw inner dict directly (zero allocation).
        """
        neighbors: dict[tuple, float] = self._graph[node]

        # Fast path — avoids dict creation when no penalised neighbours present.
        if self._penalty_nodes.isdisjoint(neighbors):
            return neighbors

        return {
            nb: w * self._penalty if nb in self._penalty_nodes else w
            for nb, w in neighbors.items()
        }


# ── Main class ────────────────────────────────────────────────────────────────

class TerrainGraph:
    """
    Incremental directed graph manager over a single unified graph.

    Public API
    ----------
    add_cell(cell)      called when a cell's state changes (observed / visited)
    update_cell(cell)   called after GP refresh or observed→visited transition
    remove_cell(x, y)   called when a cell is stuck: rewires with 0.01 trav
                        (does NOT delete — keeps graph fully connected)
    get_graph(agent)    returns a live graph view for the given agent
    """

    def __init__(
            self,
            grid_dimension: tuple = (50, 50),
            revisit_penalty_scout: float = 3.0,
            revisit_penalty_drone: float = 2.0,
            pessimistic_default: float = 0.5,
    ) -> None:
        self._grid_dimension = grid_dimension
        self.revisit_penalty_scout = revisit_penalty_scout
        self.revisit_penalty_drone = revisit_penalty_drone
        self.pessimistic_default = pessimistic_default

        # Single adjacency dict — base weights, no penalty baked in
        self._graph: dict[tuple, dict[tuple, float]] = {}

        # Penalty sets — live references read by _PenalizedView
        self._visited_nodes: set[tuple] = set()   # scout avoids re-visiting
        self._observed_nodes: set[tuple] = set()  # drone  avoids re-scanning

        # Cell registry for traversability / slope lookups
        self._cells: dict[tuple, CellData] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def add_cell(self, cell: CellData) -> None:
        """
        Register or refresh a cell in the graph.
        Stuck cells are kept with _STUCK_TRAVERSABILITY so agents always
        have a route out; they are just very expensive to traverse.
        """
        coords = (cell.x, cell.y)
        self._cells[coords] = cell

        if cell.is_observed:
            self._observed_nodes.add(coords)
        if cell.is_visited:
            self._visited_nodes.add(coords)

        if coords not in self._graph:
            self._graph[coords] = {}
        self._wire_edges(self._graph, cell)

    def update_cell(self, cell: CellData) -> None:
        """
        Recompute all edges touching `cell` after a state change:
          - GP estimate refresh
          - observed → visited transition
          - stuck discovery (rewires with 0.01 trav, does NOT delete)
        """
        coords = (cell.x, cell.y)
        self._cells[coords] = cell

        if cell.is_observed:
            self._observed_nodes.add(coords)
        if cell.is_visited:
            self._visited_nodes.add(coords)

        if coords in self._graph:
            self._rewire_edges(self._graph, cell)
        else:
            self.add_cell(cell)

    # just for completeness, not used
    def remove_cell(self, x: int, y: int) -> None:
        """
        Called when a cell is confirmed stuck.
        Rewires its edges with _STUCK_TRAVERSABILITY instead of deleting,
        so the graph stays fully connected and agents already inside the
        cell can always find a way out.
        """
        coords = (x, y)
        if coords in self._cells:
            # Cell's is_stuck flag is already True; _traversability will
            # return _STUCK_TRAVERSABILITY — just rewire to reflect this.
            self._rewire_edges(self._graph, self._cells[coords])

    def get_graph(self, agent: Literal["rover", "scout", "drone"]):
        """
        Return a live graph view suitable for the requested agent.

            rover  → raw complete_graph, no penalty
                     (confidence weighting already encodes visited preference)
            scout  → complete_graph, visited cells penalised
            drone  → complete_graph, observed cells penalised

        The returned object is a live view: mutations from add/update/remove
        are visible immediately without calling get_graph again.
        """
        if agent == "rover":
            return self._graph

        if agent == "scout":
            return _PenalizedView(
                self._graph,
                self._visited_nodes,
                self.revisit_penalty_scout,
            )

        if agent == "drone":
            return _PenalizedView(
                self._graph,
                self._observed_nodes,
                self.revisit_penalty_drone,
            )

        raise ValueError(f"Unknown agent '{agent}'. Expected 'rover', 'scout' or 'drone'.")

    # ── Private: edge wiring ──────────────────────────────────────────────────

    def _wire_edges(
            self,
            graph: dict[tuple, dict[tuple, float]],
            cell: CellData,
    ) -> None:
        """
        Connect `cell` to every existing neighbour in `graph`.
        Both directions (src→dst and dst→src) are written.
        """
        coords = (cell.x, cell.y)

        for nb_coords, nb_cell in self._existing_neighbours(cell, graph):
            diagonal = nb_coords[0] != cell.x and nb_coords[1] != cell.y

            # outgoing: cell → neighbour
            graph[coords][nb_coords] = _edge_weight(
                cell, nb_cell, diagonal, self.pessimistic_default
            )
            # incoming: neighbour → cell
            graph[nb_coords][coords] = _edge_weight(
                nb_cell, cell, diagonal, self.pessimistic_default
            )

    def _rewire_edges(
            self,
            graph: dict[tuple, dict[tuple, float]],
            cell: CellData,
    ) -> None:
        """
        Clear and recompute all edges touching `cell`.
        Incoming edges are also refreshed because traversability(dst) changed.
        """
        coords = (cell.x, cell.y)

        # Remove all outgoing edges from this node
        graph[coords] = {}

        # Remove all incoming edges pointing TO this node
        for neighbours in graph.values():
            neighbours.pop(coords, None)

        # Recompute both directions from scratch
        self._wire_edges(graph, cell)

    def _existing_neighbours(
            self,
            cell: CellData,
            graph: dict,
    ) -> list[tuple[tuple, CellData]]:
        """
        Return (coords, CellData) for the 8-connected neighbours
        that are already present in `graph`.
        """
        result = []
        for dx, dy in _DIRECTIONS:
            nb = (cell.x + dx, cell.y + dy)
            if nb in graph and nb in self._cells:
                result.append((nb, self._cells[nb]))
        return result


# ── Module-level weight helpers ───────────────────────────────────────────────

def _edge_weight(
        src: CellData,
        dst: CellData,
        diagonal: bool,
        pessimistic_default: float,
) -> float:
    """
    Directed edge cost from src to dst.

        w = (1 / effective_trav(dst)) × * slope_factor(src→dst) × diagonal_mult

    effective_trav(dst) drives the base cost; slope_factor accounts for
    whether the agent is heading uphill, downhill, or across the slope.
    """

    trav = _traversability(dst, pessimistic_default)
    trav = max(trav, 1e-3)  # guard against zero

    heading = math.atan2(dst.y - src.y, dst.x - src.x)
    slope_f = _directional_slope_factor(
        dst.slope or 0.0,
        dst.uphill_angle or 0.0,
        heading,
    )
    slope_f = max(slope_f, 1e-3)  # guard against zero

    diagonal_mult = SQRT2 if diagonal else 1.0
    return (1.0 / trav) * slope_f * diagonal_mult


def _traversability(cell: CellData, pessimistic_default: float) -> float:
    """
    Resolve effective traversability for edge weight computation.

    Priority / formula:
      1. Stuck cell                    → _STUCK_TRAVERSABILITY (0.01)
      2. Visited (ground truth)        → real_traversability
      3. Observed (GP estimate)        → confidence * estimate
                                          + (1 - confidence) * pessimistic_default
                                          then × (1 - stuck_probability_estimate)
      4. Unobserved                    → pessimistic_default

    Note: for observed cells, confidence=0 collapses the formula to
    pessimistic_default regardless of the raw estimate, so early GP
    predictions with high uncertainty don't mislead the planner.
    """
    if cell.is_stuck:
        return _STUCK_TRAVERSABILITY

    # Ground truth available (visited cell)
    if cell.real_traversability is not None:
        return cell.real_traversability

    # Observed but not yet visited: confidence-weighted estimate
    if cell.is_observed:
        estimate = (
            cell.traversability_estimate
            if cell.traversability_estimate is not None
            else pessimistic_default
        )
        trav = cell.confidence * estimate + (1.0 - cell.confidence) * pessimistic_default
        trav *= (1.0 - cell.stuck_probability_estimate)
        return max(trav, 1e-3)

    # Unobserved: assume pessimistic default
    return pessimistic_default


def _directional_slope_factor(
        slope: float,
        uphill_angle: float,
        movement_orientation: float,
        uphill_penalty: float = 1.0,
        downhill_boost: float = 0.2,
        slope_max_degrees: float = 30.0,
) -> float:
    """
    Return a scalar in (0, 1.2] that modulates cost based on the
    relationship between the agent's heading and the terrain slope.

    Duplicated from TerrainMap to avoid a circular import.
    Consider extracting to terrain_utils.py if this diverges.
    """
    ux, uy = math.cos(uphill_angle), math.sin(uphill_angle)
    mx, my = math.cos(movement_orientation), math.sin(movement_orientation)
    alignment = ux * mx + uy * my
    slope_norm = max(0.0, min(slope / slope_max_degrees, 1.0))
    signed_grade = slope_norm * alignment

    if signed_grade >= 0:
        return 1.0 - uphill_penalty * signed_grade

    return 1.0 + downhill_boost * (-signed_grade)