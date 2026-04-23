"""
TerrainGraph
============
Maintains two directed, weighted adjacency graphs over the explored terrain:

    visited_graph   nodes: is_visited=True cells   weights: real traversability
    observed_graph  nodes: is_observed=True cells   weights: GP estimate

Stuck cells are excluded from both graphs.
Edges are 8-directional; diagonals carry a ×√2 cost multiplier.

Edge weight formula (src → dst):
    w = (1 / traversability(dst)) × slope_factor(src→dst) × diagonal_mult

The revisit penalty is NOT baked into the stored weights — it is applied
lazily by _PenalizedView so the same underlying graph can be shared
by multiple agents with different penalty values.
"""

from __future__ import annotations

import math
from typing import Iterator, Literal

from CellData import CellData

# ── Constants ─────────────────────────────────────────────────────────────────

SQRT2 = math.sqrt(2)
_DIRECTIONS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


# ── Live penalty view ─────────────────────────────────────────────────────────

class _PenalizedView:
    """
    A lazy, read-only view over an adjacency dict that multiplies
    the edge weight by `penalty` for every *destination* node that
    belongs to `penalty_nodes`.

    No data is ever copied — the view reads directly from the live graph
    and the live penalty set, so any subsequent add/remove/update calls
    are immediately reflected.

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

        # Fast path — avoids dict creation in the common case where a node
        # has no penalised neighbours (early-exploration phase).
        if self._penalty_nodes.isdisjoint(neighbors):
            return neighbors

        return {
            nb: w * self._penalty if nb in self._penalty_nodes else w
            for nb, w in neighbors.items()
        }


# ── Main class ────────────────────────────────────────────────────────────────

class TerrainGraph:
    """
    Incremental directed graph manager.

    Public API
    ----------
    add_cell(cell)          called when a cell becomes observed or visited
    update_cell(cell)       called when GP estimates are refreshed, or
                            when a cell transitions observed → visited
    remove_cell(x, y)       called when a cell is discovered to be stuck
    get_graph(agent)        returns a live graph view for the given agent
    """

    def __init__(
        self,
        revisit_penalty_scout: float = 3.0,
        revisit_penalty_drone: float = 2.0,
    ) -> None:
        self.revisit_penalty_scout = revisit_penalty_scout
        self.revisit_penalty_drone = revisit_penalty_drone

        # adjacency dicts — base weights, no penalty
        self._visited_graph:  dict[tuple, dict[tuple, float]] = {}
        self._observed_graph: dict[tuple, dict[tuple, float]] = {}

        # penalty sets — live references read by _PenalizedView
        self._visited_nodes:  set[tuple] = set()   # scout avoids these
        self._observed_nodes: set[tuple] = set()   # drone  avoids these

        # cell registry for traversability / slope lookups
        self._cells: dict[tuple, CellData] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def add_cell(self, cell: CellData) -> None:
        """
        Register a newly observed or visited cell.
        Stuck cells are silently ignored.
        """
        if cell.is_stuck:
            return

        coords = (cell.x, cell.y)
        self._cells[coords] = cell

        if cell.is_observed:
            self._observed_nodes.add(coords)
            if coords not in self._observed_graph:
                self._observed_graph[coords] = {}
            self._wire_edges(self._observed_graph, cell, use_real=False)

        if cell.is_visited:
            self._visited_nodes.add(coords)
            if coords not in self._visited_graph:
                self._visited_graph[coords] = {}
            self._wire_edges(self._visited_graph, cell, use_real=True)

    def update_cell(self, cell: CellData) -> None:
        """
        Recompute all edges touching `cell`:
          - called after every GP prediction refresh (observed graph)
          - called when a cell transitions from observed → visited
          - automatically removes the cell if it is stuck
        """
        if cell.is_stuck:
            self.remove_cell(cell.x, cell.y)
            return

        coords = (cell.x, cell.y)
        self._cells[coords] = cell

        if cell.is_observed and coords in self._observed_graph:
            self._rewire_edges(self._observed_graph, cell, use_real=False)

        if cell.is_visited:
            if coords not in self._visited_graph:
                # First time this cell is marked visited — add it
                self.add_cell(cell)
            else:
                self._rewire_edges(self._visited_graph, cell, use_real=True)

    def remove_cell(self, x: int, y: int) -> None:
        """
        Purge a stuck cell and every edge that references it.
        """
        coords = (x, y)
        for graph in (self._visited_graph, self._observed_graph):
            if coords in graph:
                del graph[coords]
            for neighbours in graph.values():
                neighbours.pop(coords, None)

        self._visited_nodes.discard(coords)
        self._observed_nodes.discard(coords)
        self._cells.pop(coords, None)

    def get_graph(self, agent: Literal["rover", "scout", "drone"]):
        """
        Return a live graph view suitable for the requested agent.

            rover  → raw visited_graph,  no penalty
            scout  → observed_graph,     visited  cells penalised  (already traversed)
            drone  → observed_graph,     observed cells penalised  (already scanned)

        The returned object is a live view: mutations from add/update/remove
        are visible immediately without calling get_graph again.
        """
        if agent == "rover":
            return self._visited_graph

        if agent == "scout":
            return _PenalizedView(
                self._observed_graph,
                self._visited_nodes,
                self.revisit_penalty_scout,
            )

        if agent == "drone":
            return _PenalizedView(
                self._observed_graph,
                self._observed_nodes,
                self.revisit_penalty_drone,
            )

        raise ValueError(f"Unknown agent '{agent}'. Expected 'rover', 'scout' or 'drone'.")

    # ── Private: edge wiring ──────────────────────────────────────────────────

    def _wire_edges(
        self,
        graph: dict[tuple, dict[tuple, float]],
        cell: CellData,
        use_real: bool,
    ) -> None:
        """
        Connect `cell` to every existing neighbour in `graph`.
        Both directions (src→dst and dst→src) are written.
        """
        coords = (cell.x, cell.y)

        for nb_coords, nb_cell in self._existing_neighbours(cell, graph):
            diagonal = nb_coords[0] != cell.x and nb_coords[1] != cell.y

            # outgoing: cell → neighbour
            graph[coords][nb_coords] = _edge_weight(cell, nb_cell, diagonal, use_real)

            # incoming: neighbour → cell
            graph[nb_coords][coords] = _edge_weight(nb_cell, cell, diagonal, use_real)

    def _rewire_edges(
        self,
        graph: dict[tuple, dict[tuple, float]],
        cell: CellData,
        use_real: bool,
    ) -> None:
        """
        Clear and recompute all edges touching `cell`.
        Incoming edges (from neighbours to `cell`) are also refreshed
        because traversability(dst) changed.
        """
        coords = (cell.x, cell.y)

        # Remove all outgoing edges from this node
        graph[coords] = {}

        # Remove all incoming edges that point TO this node
        for neighbours in graph.values():
            neighbours.pop(coords, None)

        # Recompute both directions from scratch
        self._wire_edges(graph, cell, use_real)

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
    use_real: bool,
) -> float:
    """
    Directed edge cost from src to dst.

        w = (1 / trav(dst))  ×  slope_factor(src→dst)  ×  diagonal_mult

    traversability(dst) drives the base cost; slope_factor accounts for
    whether the agent is heading uphill, downhill or across the slope.
    """
    trav = _traversability(dst, use_real)
    trav = max(trav, 1e-3)                           # guard against zero

    heading = math.atan2(dst.y - src.y, dst.x - src.x)
    slope_f = _directional_slope_factor(
        dst.slope or 0.0,
        dst.uphill_angle or 0.0,
        heading,
    )
    slope_f = max(slope_f, 1e-3)                     # guard against zero

    diagonal_mult = SQRT2 if diagonal else 1.0
    return (1.0 / trav) * slope_f * diagonal_mult


def _traversability(cell: CellData, use_real: bool) -> float:
    """
    Resolve the traversability value to use for a cell.

    Priority:
      1. real_traversability  (if use_real and ground truth is available)
      2. traversability_estimate  (GP prediction)
      3. 0.5  (neutral fallback — model not yet fitted)
    """
    if use_real and cell.real_traversability is not None:
        return cell.real_traversability
    if cell.traversability_estimate is not None:
        return cell.traversability_estimate
    return 0.5


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
    Return a scalar in (0, 1.2] that modulates cost based on the
    relationship between the agent's heading and the terrain slope.

    Duplicated from TerrainMap to avoid a circular import.
    If you later extract utilities to terrain_utils.py, both modules
    should import from there instead.
    """
    ux, uy = math.cos(uphill_angle), math.sin(uphill_angle)
    mx, my = math.cos(movement_orientation), math.sin(movement_orientation)
    alignment = ux * mx + uy * my
    slope_norm = max(0.0, min(slope / slope_max_degrees, 1.0))
    signed_grade = slope_norm * alignment

    if signed_grade >= 0:
        return 1.0 - uphill_penalty * signed_grade

    return 1.0 + downhill_boost * (-signed_grade)
