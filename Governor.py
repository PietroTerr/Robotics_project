import math
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from TerrainMap import TerrainMap

# ─── Certification thresholds (tunable) ──────────────────────────────────────

ROVER_CONF_THRESHOLD  = 0.7    # minimum confidence for rover to enter a cell
ROVER_STUCK_THRESHOLD = 0.05   # maximum stuck probability for rover to enter a cell


# ─── AgentState ───────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    agent: Any                           # Drone, Scout, or Rover
    goals: list[tuple[float, float]]     # Ordered waypoints to cycle through
    goal_index: int = 0                  # Which goal is currently targeted
    current_step: tuple = field(default=None)   # Next cell the agent is heading to
    terminal: bool = False               # If True, done when goal is reached
    finished: bool = False               # Set True when terminal goal reached

    # Rover planned path — read by scout to find uncertified cells
    planned_path: list[tuple] = field(default_factory=list)

    # Scout certification target — fixed until reached, then re-evaluated
    cert_target: tuple = field(default=None)

    def __post_init__(self):
        self.current_step = None   # force A* on first tick

    @property
    def current_goal(self) -> tuple[float, float]:
        return self.goals[self.goal_index]

    def advance_goal(self):
        if self.goal_index == len(self.goals) - 1:
            self.finished = True
        self.goal_index = (self.goal_index + 1) % len(self.goals)


# ─── Governor ─────────────────────────────────────────────────────────────────

class Governor:

    def __init__(self, terrain_map: TerrainMap, agents: list[AgentState]):
        self.terrain_map = terrain_map
        self.agents: list[AgentState] = agents
        self.done = False

    def get_headings(self) -> dict[str, float | None]:
        """
        Return a dict mapping each robot_id to its heading (radians) or None.
        None means the agent should stay still.
        """
        return {
            state.agent.robot_id: self._get_agent_heading(state)
            for state in self.agents
        }

    # ─── Private: dispatch by robot type ──────────────────────────────────────

    def _get_agent_heading(self, state: AgentState) -> float | None:
        if getattr(state.agent, "needs_pause", False):
            return None

        t = state.agent.robot_type
        if t == "rover":
            return self._heading_rover(state)
        if t == "scout":
            return self._heading_scout(state)
        if t == "drone":
            return self._heading_drone(state)
        return None

    # ─── Rover ────────────────────────────────────────────────────────────────

    def _heading_rover(self, state: AgentState) -> float | None:
        """
        Rover logic:
          1. Replan A* when current_step is None or finished.
             Store full path in state.planned_path.
          2. Before moving into next cell, check certification thresholds.
             If not certified → wait (return None).
          3. If certified → return heading.
        """
        agent = state.agent
        current_pos = (agent.x, agent.y)

        # ── Goal reached ──────────────────────────────────────────────────────
        if _to_cell_coords(current_pos) == _to_cell_coords(state.current_goal):
            if state.terminal:
                self.done = True
                return None
            state.advance_goal()
            state.current_step = None   # force replan

        # ── Replan if needed ──────────────────────────────────────────────────
        if state.current_step is None or _step_is_finished(current_pos, state.current_step):
            source = _to_cell_coords(current_pos)
            target = _to_cell_coords(state.current_goal)

            if source == target:
                state.current_step = source
                return _get_direction_to_cell(current_pos, state.current_step)

            G = _to_networkx(self.terrain_map.terrain_graph.get_graph("rover"))
            effective_source = source if source in G else _nearest_valid_source(source, G)
            if effective_source is None:
                state.planned_path = []
                state.current_step = None
                return None

            try:
                path = nx.astar_path(
                    G,
                    source=effective_source,
                    target=target,
                    heuristic=lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]),
                    weight="weight",
                )
                state.planned_path = path          # expose full path to scout
                state.current_step = path[1] if len(path) > 1 else target
            except (nx.NodeNotFound, nx.NetworkXNoPath, nx.exception.NetworkXError):
                state.planned_path = []
                state.current_step = None
                return None

        # ── Certification check before moving ─────────────────────────────────
        next_cell = state.current_step
        if not _is_certified(next_cell, self.terrain_map):
            return None    # wait for scout to certify

        return _get_direction_to_cell(current_pos, next_cell)

    # ─── Scout ────────────────────────────────────────────────────────────────

    def _heading_scout(self, state: AgentState) -> float | None:
        """
        Scout logic:
          1. If rover has a planned path, find the first uncertified cell on it.
             Use that as certification target (fixed until reached, no oscillation).
          2. If rover path is fully certified or empty, fall back to own goal.
          3. Only switch certification target when current cert_target is reached.
        """
        agent = state.agent
        current_pos = (agent.x, agent.y)

        # ── Own goal reached (fallback cycling) ───────────────────────────────
        if _to_cell_coords(current_pos) == _to_cell_coords(state.current_goal):
            state.advance_goal()
            state.current_step = None

        # ── Determine target for this tick ────────────────────────────────────
        target_pos = self._get_scout_target(state, current_pos)

        # ── Replan if needed ──────────────────────────────────────────────────
        if state.current_step is None or _step_is_finished(current_pos, state.current_step):
            source = _to_cell_coords(current_pos)
            target = _to_cell_coords(target_pos)

            if source == target:
                # Certification target reached — clear so it re-evaluates next tick
                state.cert_target = None
                state.current_step = source
                return _get_direction_to_cell(current_pos, state.current_step)

            G = _to_networkx(self.terrain_map.terrain_graph.get_graph("scout"))
            effective_source = source if source in G else _nearest_valid_source(source, G)
            if effective_source is None:
                state.current_step = None
                return None

            try:
                path = nx.astar_path(
                    G,
                    source=effective_source,
                    target=target,
                    heuristic=lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]),
                    weight="weight",
                )
                state.current_step = path[1] if len(path) > 1 else target
            except (nx.NodeNotFound, nx.NetworkXNoPath, nx.exception.NetworkXError):
                state.current_step = None
                return None

        return _get_direction_to_cell(current_pos, state.current_step)

    def _get_scout_target(self, state: AgentState, current_pos: tuple) -> tuple:
        """
        Return the scout's current target position.

        Priority:
          1. Active cert_target not yet reached → keep it (prevents oscillation)
          2. Find first uncertified cell on rover's planned path → new cert_target
          3. Fall back to scout's own current_goal
        """
        # 1. Keep existing cert_target until physically reached
        if state.cert_target is not None:
            if _to_cell_coords(current_pos) != _to_cell_coords(state.cert_target):
                return state.cert_target
            else:
                state.cert_target = None   # reached — re-evaluate

        # 2. Find first uncertified cell on rover's planned path
        rover_state = self._get_rover_state()
        if rover_state is not None and rover_state.planned_path:
            uncertified = _first_uncertified_cell(
                rover_state.planned_path, self.terrain_map
            )
            if uncertified is not None:
                state.cert_target = uncertified
                return uncertified

        # 3. Fall back to scout's own goal
        return state.current_goal

    def _get_rover_state(self) -> AgentState | None:
        """Return the AgentState whose agent has robot_type == 'rover'."""
        for s in self.agents:
            if s.agent.robot_type == "rover":
                return s
        return None

    # ─── Drone ────────────────────────────────────────────────────────────────

    def _heading_drone(self, state: AgentState) -> float | None:
        """
        Drone logic: unchanged from original.
        A* toward cycling goals, penalizing already-observed cells.
        """
        agent = state.agent
        current_pos = (agent.x, agent.y)

        if _to_cell_coords(current_pos) == _to_cell_coords(state.current_goal):
            if state.terminal:
                self.done = True
                return None
            state.advance_goal()
            state.current_step = None

        if state.current_step is None or _step_is_finished(current_pos, state.current_step):
            source = _to_cell_coords(current_pos)
            target = _to_cell_coords(state.current_goal)

            if source == target:
                state.current_step = source
                return _get_direction_to_cell(current_pos, state.current_step)

            G = _to_networkx(self.terrain_map.terrain_graph.get_graph("drone"))
            effective_source = source if source in G else _nearest_valid_source(source, G)
            if effective_source is None:
                state.current_step = None
                return None

            try:
                path = nx.astar_path(
                    G,
                    source=effective_source,
                    target=target,
                    heuristic=lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]),
                    weight="weight",
                )
                state.current_step = path[1] if len(path) > 1 else target
            except (nx.NodeNotFound, nx.NetworkXNoPath, nx.exception.NetworkXError):
                state.current_step = None
                return None

        return _get_direction_to_cell(current_pos, state.current_step)


# ─── Certification helpers ────────────────────────────────────────────────────

def _is_certified(cell_coords: tuple, terrain_map: TerrainMap) -> bool:
    """
    Return True if the cell is safe enough for the rover to enter.

    Certified if:
      - Visited and not stuck (ground truth, best case), OR
      - Observed with confidence >= ROVER_CONF_THRESHOLD
        AND stuck_probability_estimate <= ROVER_STUCK_THRESHOLD
    """
    cell = terrain_map.grid.get(cell_coords)
    if cell is None:
        return False
    if cell.is_visited:
        return not cell.is_stuck
    if not cell.is_observed:
        return False
    return (
        cell.confidence >= ROVER_CONF_THRESHOLD
        and cell.stuck_probability_estimate <= ROVER_STUCK_THRESHOLD
    )


def _first_uncertified_cell(
        path: list[tuple], terrain_map: TerrainMap
) -> tuple | None:
    """
    Scan the rover's planned path and return the first cell that is not
    certified. Returns None if the entire path is already certified.
    """
    for cell_coords in path:
        if not _is_certified(cell_coords, terrain_map):
            return cell_coords
    return None


# ─── Graph / geometry helpers ─────────────────────────────────────────────────

def _step_is_finished(position: tuple, goal: tuple) -> bool:
    return _to_cell_coords(position) == _to_cell_coords(goal)


def _nearest_valid_source(source: tuple, G) -> tuple | None:
    """
    Return the closest 8-connected neighbour of `source` that exists in G.
    Used when the agent is physically inside a stuck cell.
    """
    x, y = source
    for dx, dy in [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]:
        nb = (x + dx, y + dy)
        if nb in G:
            return nb
    return None


def _to_networkx(graph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in graph:
        for nb, weight in graph[node].items():
            G.add_edge(node, nb, weight=weight)
    return G


def _to_cell_coords(position: tuple[float, float]) -> tuple[int, int]:
    return int(position[0]), int(position[1])


def _get_direction_to_cell(start: tuple, end: tuple) -> float:
    """Angle in radians from start toward the centre of cell end."""
    end_x = end[0] + 0.5
    end_y = end[1] + 0.5
    return math.atan2(end_y - start[1], end_x - start[0])