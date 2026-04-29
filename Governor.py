import math
from dataclasses import dataclass, field
from typing import Any
import networkx as nx
from TerrainMap import TerrainMap
from motion import RobotMovementBase


# ─── AgentState ───────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    agent: RobotMovementBase
    goals: list[tuple[float, float]]
    goal_index: int = 0
    current_step: tuple = field(default=None)
    terminal: bool = False
    finished: bool = False  # True when agent has reached final goal

    # --- Oscillation ---
    use_oscillation: bool = False
    oscillation_step: int = 0

    # --- Planned path (populated for rover, readable by others) ---
    planned_path: list = field(default_factory=list)

    # --- Agent to escort/protect ---
    # state_to_protect_reached: 0 = not started, 0.5 = in progress, 1 = done
    state_to_protect: Any = None
    state_to_protect_reached: float = 0

    @property
    def is_moving(self) -> bool:
        return self.agent.current_speed != 0.0

    @property
    def current_goal(self) -> tuple[float, float]:
        return self.goals[self.goal_index]

    def advance_goal(self):
        if self.goal_index == len(self.goals) - 1:
            self.finished = True
        self.goal_index = (self.goal_index + 1) % len(self.goals)


# ─── Governor ─────────────────────────────────────────────────────────────────

class Governor:
    def __init__(self, terrain_map: TerrainMap, agents: list[AgentState],
                 osc_amplitude=0.9, osc_frequency=0.01):
        self.terrain_map = terrain_map
        self.agents: list[AgentState] = agents
        self.done = False

        self.osc_amplitude = osc_amplitude
        self.osc_frequency = osc_frequency

    def get_headings(self) -> dict[str, float | None]:
        return {
            state.agent.robot_id: self._get_agent_heading(state)
            for state in self.agents
        }

    def _get_agent_heading(self, state: AgentState) -> float | None:
        if getattr(state.agent, "needs_pause", False):
            return None

        agent = state.agent
        current_pos = (agent.x, agent.y)

        # ── Escort logic: follow/protect another agent ────────────────────────
        if (
            state.state_to_protect is not None
            and state.state_to_protect_reached != 1
            and state.state_to_protect.is_moving
            and state.state_to_protect.planned_path
        ):
            protected = state.state_to_protect
            path_set = set(map(tuple, protected.planned_path))
            protected_pos = (protected.agent.x, protected.agent.y)
            protected_cell = _to_cell_coords(protected_pos)
            scout_cell = _to_cell_coords(current_pos)

            if scout_cell in path_set:
                # On the path — check if we've caught up to the protected agent
                if check_if_same_cell(current_pos, protected_pos):
                    # Reached the protected agent: resume own goals
                    state.advance_goal()
                    state.state_to_protect_reached = 1
                else:
                    # Still on the path but not yet caught up: chase the protected agent
                    if state.state_to_protect_reached == 0.5:
                        # Update existing injected goal to follow protected agent
                        state.goals[state.goal_index] = protected_cell
                    else:
                        # First time on the path: inject protected agent's cell as goal
                        state.goals.insert(state.goal_index, protected_cell)
                        state.state_to_protect_reached = 0.5
            else:
                # Not on the path yet: head to the nearest path cell (relative to self)
                nearest = min(
                    protected.planned_path,
                    key=lambda c: math.hypot(c[0] - scout_cell[0], c[1] - scout_cell[1]),
                )
                if state.state_to_protect_reached == 0.5:
                    # Update existing injected goal
                    state.goals[state.goal_index] = nearest
                else:
                    # First approach: inject nearest path cell as goal
                    state.goals.insert(state.goal_index, nearest)
                    state.state_to_protect_reached = 0.5

        # ── Standard A* toward current goal ───────────────────────────────────
        target_pos = state.current_goal

        if state.current_step is None or check_if_same_cell(current_pos, state.current_step):
            if _to_cell_coords(current_pos) == _to_cell_coords(target_pos):
                if state.terminal:
                    self.done = True
                    return None
                state.advance_goal()
                return self._get_agent_heading(state)

            source = _to_cell_coords(current_pos)
            target = _to_cell_coords(target_pos)

            if source == target:
                state.current_step = source
            else:
                G = self.terrain_map.terrain_graph.get_graph(agent.robot_type)
                effective_source = source if source in G else _nearest_valid_source(source, G)
                if effective_source is None:
                    return None

                if agent.robot_type == "rover":
                    # Rover stores full path so other agents can reference it
                    full = astar_graph(G, start=effective_source, goal=target, full_path=True)
                    if full:
                        state.planned_path = full
                        state.current_step = full[1] if len(full) > 1 else full[0]
                    else:
                        print("ERROR IN GRAPH SEARCH: No path found from", source, "to", target, "for agent", agent.robot_id)
                        return None
                else:
                    next_step = astar_graph(G, start=effective_source, goal=target)
                    if next_step is not None:
                        state.current_step = next_step
                    else:
                        print("ERROR IN GRAPH SEARCH: No path found from", source, "to", target, "for agent", agent.robot_id)
                        return None

        # ── Heading + optional oscillation ────────────────────────────────────
        heading = _get_direction_to_cell(current_pos, state.current_step)

        if state.use_oscillation:
            state.oscillation_step += 1
            heading += self.osc_amplitude * math.sin(self.osc_frequency * state.oscillation_step)

        return heading


# ─── Geometry / Graph Helpers ─────────────────────────────────────────────────

def check_if_same_cell(position: tuple, goal: tuple) -> bool:
    """True when position and goal map to the same grid cell."""
    return _to_cell_coords(position) == _to_cell_coords(goal)


def _nearest_valid_source(source: tuple, G) -> tuple | None:
    """
    Return the first 8-connected neighbour of source that exists in G.
    Used when an agent is inside a stuck cell that has been rewired.
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


# ─── A* ───────────────────────────────────────────────────────────────────────

import heapq


def astar_graph(graph, start: tuple, goal: tuple, full_path: bool = False):
    """
    A* over a TerrainGraph adjacency dict or _PenalizedView.

    full_path=False (default): returns only the next step toward goal.
    full_path=True:            returns the complete path as list[tuple].
    Returns None if no path exists.
    """
    open_heap = [(0, start)]
    g = {start: 0}
    came_from = {}

    while open_heap:
        f, current = heapq.heappop(open_heap)

        if current == goal:
            if not full_path:
                if current == start:
                    return start
                while came_from.get(current) != start:
                    current = came_from[current]
                return current
            else:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

        neighbors = graph[current] if current in graph else {}
        for nb, weight in neighbors.items():
            new_g = g[current] + weight
            if new_g < g.get(nb, float('inf')):
                g[nb] = new_g
                f_score = new_g + math.hypot(nb[0] - goal[0], nb[1] - goal[1])
                heapq.heappush(open_heap, (f_score, nb))
                came_from[nb] = current

    return None