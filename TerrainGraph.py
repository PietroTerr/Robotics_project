from __future__ import annotations

import math
from typing import Iterator, Literal

from CellData import CellData

# ── Constants ─────────────────────────────────────────────────────────────────

SQRT2 = math.sqrt(2)


# ── Live penalty view (Untouched) ─────────────────────────────────────────────

class _PenalizedView:
    """
    A lazy, read-only view over an adjacency dict that multiplies
    the edge weight by `penalty` for every *destination* node that
    belongs to `penalty_nodes`.
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

    def __contains__(self, node: object) -> bool:
        return node in self._graph

    def __iter__(self) -> Iterator[tuple]:
        return iter(self._graph)

    def __len__(self) -> int:
        return len(self._graph)

    def keys(self):
        return self._graph.keys()

    def __getitem__(self, node: tuple) -> dict[tuple, float]:
        neighbors: dict[tuple, float] = self._graph[node]
        if self._penalty_nodes.isdisjoint(neighbors):
            return neighbors

        return {
            nb: w * self._penalty if nb in self._penalty_nodes else w
            for nb, w in neighbors.items()
        }


# ── Base Class ────────────────────────────────────────────────────────────────

class BaseTerrainGraph:
    """
    Base Incremental directed graph manager.
    Handles node tracking, penalty sets, and dict mutations.
    Child classes must implement neighborhood and weight logic.
    """

    def __init__(
            self,
            grid_dimension: tuple = (50, 50),
            revisit_penalty_scout: float = 3.0,
            revisit_penalty_drone: float = 2.0,
    ) -> None:
        self._grid_dimension = grid_dimension
        self.revisit_penalty_scout = revisit_penalty_scout
        self.revisit_penalty_drone = revisit_penalty_drone

        self._complete_graph: dict[tuple, dict[tuple, float]] = {}
        self._visited_graph: dict[tuple, dict[tuple, float]] = {}
        self._observed_graph: dict[tuple, dict[tuple, float]] = {}

        self._all_nodes: set[tuple] = set()
        self._visited_nodes: set[tuple] = set()
        self._observed_nodes: set[tuple] = set()

        self._cells: dict[tuple, CellData] = {}

    def add_cell(self, cell: CellData) -> None:
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

        self._all_nodes.add(coords)
        if coords not in self._complete_graph:
            self._complete_graph[coords] = {}
        self._wire_edges(self._complete_graph, cell, use_real=False)

    def update_cell(self, cell: CellData) -> None:
        if cell.is_stuck:
            self.remove_cell(cell.x, cell.y)
            return

        coords = (cell.x, cell.y)
        self._cells[coords] = cell

        if cell.is_observed and coords in self._observed_graph:
            self._rewire_edges(self._observed_graph, cell, use_real=False)

        if cell.is_visited:
            if coords not in self._visited_graph:
                self.add_cell(cell)
            else:
                self._rewire_edges(self._visited_graph, cell, use_real=True)

    def remove_cell(self, x: int, y: int) -> None:
        coords = (x, y)
        for graph in (self._visited_graph, self._observed_graph, self._complete_graph):
            if coords in graph:
                del graph[coords]
            for neighbours in graph.values():
                neighbours.pop(coords, None)

        self._visited_nodes.discard(coords)
        self._observed_nodes.discard(coords)
        self._cells.pop(coords, None)

    def get_graph(self, agent: Literal["rover", "scout", "drone"]):
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
                self._complete_graph,
                self._observed_nodes,
                self.revisit_penalty_drone,
            )
        raise ValueError(f"Unknown agent '{agent}'.")

    # ── Edge Wiring (Delegates math to children) ──

    def _wire_edges(self, graph: dict, cell: CellData, use_real: bool) -> None:
        coords = (cell.x, cell.y)
        for nb_coords, nb_cell in self._get_neighbours(cell, graph):
            w_out, w_in = self._calculate_weights(cell, nb_cell, use_real)
            graph[coords][nb_coords] = w_out
            graph[nb_coords][coords] = w_in

    def _rewire_edges(self, graph: dict, cell: CellData, use_real: bool) -> None:
        coords = (cell.x, cell.y)
        graph[coords] = {}
        for neighbours in graph.values():
            neighbours.pop(coords, None)
        self._wire_edges(graph, cell, use_real)

    # ── Abstract Methods ──

    def _get_neighbours(self, cell: CellData, graph: dict) -> list[tuple[tuple, CellData]]:
        raise NotImplementedError

    def _calculate_weights(self, cell: CellData, nb_cell: CellData, use_real: bool) -> tuple[float, float]:
        raise NotImplementedError


# ── Subclass 1: Complex 8-Way Graph ───────────────────────────────────────────

class ComplexTerrainGraph(BaseTerrainGraph):
    """
    Original 8-way directional graph with slope penalties.
    """
    _DIRECTIONS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

    def _get_neighbours(self, cell: CellData, graph: dict) -> list[tuple[tuple, CellData]]:
        result = []
        for dx, dy in self._DIRECTIONS:
            nb = (cell.x + dx, cell.y + dy)
            if nb in graph and nb in self._cells:
                result.append((nb, self._cells[nb]))
        return result

    def _calculate_weights(self, cell: CellData, nb_cell: CellData, use_real: bool) -> tuple[float, float]:
        diagonal = nb_cell.x != cell.x and nb_cell.y != cell.y
        w_out = _complex_edge_weight(cell, nb_cell, diagonal, use_real)
        w_in = _complex_edge_weight(nb_cell, cell, diagonal, use_real)
        return w_out, w_in


# ── Subclass 2: Simple 4-Way Graph ────────────────────────────────────────────

class SimpleTerrainGraph(BaseTerrainGraph):
    """
    New 4-way symmetrical graph based solely on traversability.
    """
    _DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def _get_neighbours(self, cell: CellData, graph: dict) -> list[tuple[tuple, CellData]]:
        result = []
        for dx, dy in self._DIRECTIONS:
            nb = (cell.x + dx, cell.y + dy)
            if nb in graph and nb in self._cells:
                result.append((nb, self._cells[nb]))
        return result

    def _calculate_weights(self, cell: CellData, nb_cell: CellData, use_real: bool) -> tuple[float, float]:
        # Symmetrical: src -> dst costs the exact same as dst -> src
        w = _symmetric_edge_weight(cell, nb_cell, use_real)
        return w, w


# ── Module-level weight helpers ───────────────────────────────────────────────

def _traversability(cell: CellData, use_real: bool) -> float:
    if use_real and cell.real_traversability is not None:
        return cell.real_traversability
    if cell.traversability_estimate is not None:
        return cell.traversability_estimate
    return 0.5


def _symmetric_edge_weight(src: CellData, dst: CellData, use_real: bool) -> float:
    """
    Symmetrical weight: 0.5 * (cost of src + cost of dst)
    Ignores diagonals and slopes completely.
    """
    trav_src = max(_traversability(src, use_real), 1e-3)
    trav_dst = max(_traversability(dst, use_real), 1e-3)

    return 0.5 * ((1.0 / trav_src) + (1.0 / trav_dst))


def _complex_edge_weight(src: CellData, dst: CellData, diagonal: bool, use_real: bool) -> float:
    trav = max(_traversability(dst, use_real), 1e-3)

    heading = math.atan2(dst.y - src.y, dst.x - src.x)
    slope_f = _directional_slope_factor(
        dst.slope or 0.0,
        dst.uphill_angle or 0.0,
        heading,
    )
    slope_f = max(slope_f, 1e-3)

    diagonal_mult = SQRT2 if diagonal else 1.0
    return (1.0 / trav) * slope_f * diagonal_mult


def _directional_slope_factor(
        slope: float,
        uphill_angle: float,
        movement_orientation: float,
        *,
        uphill_penalty: float = 1.0,
        downhill_boost: float = 0.2,
        slope_max_degrees: float = 30.0,
) -> float:
    ux, uy = math.cos(uphill_angle), math.sin(uphill_angle)
    mx, my = math.cos(movement_orientation), math.sin(movement_orientation)
    alignment = ux * mx + uy * my
    slope_norm = max(0.0, min(slope / slope_max_degrees, 1.0))
    signed_grade = slope_norm * alignment

    if signed_grade >= 0:
        return 1.0 - uphill_penalty * signed_grade

    return 1.0 + downhill_boost * (-signed_grade)