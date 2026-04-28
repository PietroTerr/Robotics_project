
# ─── AgentState ───────────────────────────────────────────────────────────────

import math
from dataclasses import dataclass, field
from typing import Any
import networkx as nx
from TerrainMap import TerrainMap


@dataclass
class AgentState:
    agent: Any
    goals: list[tuple[float, float]]
    goal_index: int = 0
    current_step: tuple = field(default=None)
    terminal: bool = False
    finished: bool = False

    # Zigzag fields (kept for backward compatibility)
    use_zigzag: bool = False
    scout_side: int = 1
    active_zigzag_wp: tuple = None

    # --- New Oscillation Fields ---
    use_oscillation: bool = False
    oscillation_step: int = 0

    def __post_init__(self):
        self.current_step = None

    @property
    def current_goal(self) -> tuple[float, float]:
        return self.goals[self.goal_index]

    def advance_goal(self):
        if self.goal_index == len(self.goals) - 1:
            self.finished = True
        self.goal_index = (self.goal_index + 1) % len(self.goals)


class Governor:
    def __init__(self, terrain_map: TerrainMap, agents: list[AgentState],
                 zig_lookahead=5.0, zig_width=4.0,
                 osc_amplitude=0.9, osc_frequency=0.01):
        self.terrain_map = terrain_map
        self.agents: list[AgentState] = agents
        self.done = False
        self.zig_lookahead = zig_lookahead
        self.zig_width = zig_width

        # Oscillation settings (amplitude in radians, frequency in steps)
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

        # 1. Determine Target
        target_pos = state.current_goal
        if state.use_zigzag:
            target_pos = self._calculate_zigzag_target(state, current_pos)

        # 2. Path Planning (A*)
        if state.current_step is None or _step_is_finished(current_pos, state.current_step):
            if _to_cell_coords(current_pos) == _to_cell_coords(state.current_goal):
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
                G = _to_networkx(self.terrain_map.terrain_graph.get_graph(agent.robot_type))
                effective_source = source if source in G else _nearest_valid_source(source, G)
                if effective_source is None: return None

                try:
                    path = nx.astar_path(G, source=effective_source, target=target,
                                         heuristic=lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]),
                                         weight="weight")
                    state.current_step = path[1] if len(path) > 1 else target
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Robustness: if zigzag target is unreachable, fall back to main goal
                    if state.use_zigzag:
                        state.use_zigzag = False
                        return self._get_agent_heading(state)
                    return None

        # 3. Calculate Heading and apply Oscillation
        heading = _get_direction_to_cell(current_pos, state.current_step)

        if state.use_oscillation:
            state.oscillation_step += 1
            # Adds a small sinusoidal offset to the heading
            heading += self.osc_amplitude * math.sin(self.osc_frequency * state.oscillation_step)

        return heading

    def _calculate_zigzag_target(self, state: AgentState, current_pos: tuple) -> tuple:
        """Computes the intermediate zigzag waypoint based on unobserved terrain."""
        # Setup geometry
        origin = state.goals[state.goal_index - 1] if state.goal_index > 0 else (0.0, 0.0)
        goal = state.current_goal

        dx, dy = goal[0] - origin[0], goal[1] - origin[1]
        dist = math.hypot(dx, dy)
        if dist == 0: return goal

        axis = (dx / dist, dy / dist)
        perp = (-axis[1], axis[0])

        # Project agent onto the path axis
        agent_vec = (current_pos[0] - origin[0], current_pos[1] - origin[1])
        scout_proj = agent_vec[0] * axis[0] + agent_vec[1] * axis[1]

        # Determine if we need a new zigzag waypoint
        needs_new_wp = (state.active_zigzag_wp is None or
                        math.hypot(current_pos[0] - state.active_zigzag_wp[0],
                                   current_pos[1] - state.active_zigzag_wp[1]) < 1.5)

        if needs_new_wp:
            base_proj = min(scout_proj + self.zig_lookahead, dist)

            # Logic for choosing side based on unobserved cells
            left_unobs = 0
            right_unobs = 0
            for d_dist in range(2, int(self.zig_lookahead * 1.5) + 1):
                for w in range(1, int(self.zig_width)):
                    for side_mult, counter in [(-1, "left"), (1, "right")]:
                        tx = int(current_pos[0] + d_dist * axis[0] + side_mult * w * perp[0])
                        ty = int(current_pos[1] + d_dist * axis[1] + side_mult * w * perp[1])

                        if 0 <= tx < 50 and 0 <= ty < 50:
                            cell = self.terrain_map.grid.get((tx, ty))
                            if cell is None or not cell.is_observed:
                                if side_mult == -1:
                                    left_unobs += 1
                                else:
                                    right_unobs += 1

            if left_unobs > right_unobs:
                state.scout_side = -1
            elif right_unobs > left_unobs:
                state.scout_side = 1
            else:
                state.scout_side *= -1  # Flip side if equal or no info

            raw_x = origin[0] + base_proj * axis[0] + state.scout_side * self.zig_width * perp[0]
            raw_y = origin[1] + base_proj * axis[1] + state.scout_side * self.zig_width * perp[1]

            state.active_zigzag_wp = (max(0.0, min(49.0, raw_x)), max(0.0, min(49.0, raw_y)))

        return state.active_zigzag_wp


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
