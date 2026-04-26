import math
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

from TerrainMap import TerrainMap


# ─── AgentState ───────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    agent: Any                          # Drone, Scout, Rover, or any future agent
    goals: list[tuple[float, float]]    # Ordered list of waypoints to cycle through
    goal_index: int = 0                 # Which goal is currently targeted
    current_step: tuple = field(default=None)  # Next cell the agent is heading to
    terminal: bool = False              # If True, signals simulation end on goal reached

    def __post_init__(self):
        # current_step starts as None — A* is run on the very first call
        # to get_headings() rather than driving straight toward the goal.
        self.current_step = None

    @property
    def current_goal(self) -> tuple[float, float]:
        return self.goals[self.goal_index]

    def advance_goal(self):
        """Move to the next goal, cycling back to 0 when the list is exhausted."""
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
        None means the agent should stay still (e.g. drone recharging, or
        no valid path exists yet).
        """
        return {
            state.agent.robot_id: self._get_agent_heading(state)
            for state in self.agents
        }

    # ─── Private ──────────────────────────────────────────────────────────────

    def _get_agent_heading(self, state: AgentState) -> float | None:
        if getattr(state.agent, "needs_pause", False):
            return None

        agent = state.agent
        current_pos = (agent.x, agent.y)

        # Replan when:
        #   - first call (current_step is None), or
        #   - agent has reached the previously planned next cell
        if state.current_step is None or _step_is_finished(current_pos, state.current_step):

            # Check if the current goal has been reached
            if _to_cell_coords(current_pos) == _to_cell_coords(state.current_goal):
                if state.terminal:
                    self.done = True
                    return None
                state.advance_goal()

            source = _to_cell_coords(current_pos)
            target = _to_cell_coords(state.current_goal)

            # Edge case: source and goal are in the same cell after advance_goal
            if source == target:
                state.current_step = source
                return _get_direction_to_cell(current_pos, state.current_step)

            G = _to_networkx(
                self.terrain_map.terrain_graph.get_graph(agent.robot_type)
            )

            # Handle source not in graph (e.g. scout/drone physically inside a
            # stuck cell that was rewired but may have lost incoming edges).
            effective_source = source if source in G else _nearest_valid_source(source, G)
            if effective_source is None:
                # Completely surrounded — no valid neighbour in graph, wait.
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
                # path[0] is effective_source (may equal source), path[1] is next cell
                state.current_step = path[1] if len(path) > 1 else target
            except (nx.NodeNotFound, nx.NetworkXNoPath, nx.exception.NetworkXError):
                # No connected path yet (graph too sparse at start, or all routes
                # blocked by stuck cells) — agent waits for more exploration.
                state.current_step = None
                return None

        return _get_direction_to_cell(current_pos, state.current_step)


# ─── Geometry / Graph Helpers ─────────────────────────────────────────────────

def _step_is_finished(position: tuple, goal: tuple) -> bool:
    """True when the agent has entered the goal cell."""
    return _to_cell_coords(position) == _to_cell_coords(goal)


def _nearest_valid_source(source: tuple, G) -> tuple | None:
    """
    Scan the 8-connected neighbours of `source` and return the first one
    present in G. Returns None if no valid neighbour exists.

    Used when an agent is physically inside a stuck cell that has been
    rewired with very high cost — the agent needs a valid graph node to
    start A* from.
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