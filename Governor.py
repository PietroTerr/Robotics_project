import math
from dataclasses import dataclass, field
from typing import Any

from TerrainMap import TerrainMap
from motion import Drone, Rover, Scout
import networkx as nx


# ─── AgentState ───────────────────────────────────────────────────────────────

@dataclass
class AgentState:
    agent: Any                          # Drone, Scout, Rover, or any future agent
    goals: list[tuple[float, float]]    # Ordered list of waypoints to cycle through
    goal_index: int = 0                 # Which goal is currently targeted
    current_step: tuple = field(default=None)  # Next cell the agent is heading to
    terminal: bool = False              # If True, signals simulation end on goal reached

    def __post_init__(self):
        self.current_step = self.goals[0]

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
        None means the agent should stay still (e.g. drone recharging).
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

        if _step_is_finished(current_pos, state.current_step):

            # ← cell-level check, consistent with A* granularity
            if _to_cell_coords(current_pos) == _to_cell_coords(state.current_goal):
                if state.terminal:
                    self.done = True  # ← Governor signals completion
                    return None
                state.advance_goal()

            source = _to_cell_coords(current_pos)
            target = _to_cell_coords(state.current_goal)

            # source == target can still happen if advance_goal wraps around
            # to a goal in the same cell; just aim for the centre directly
            if source == target:
                state.current_step = source
                return _get_direction_to_cell(current_pos, state.current_step)

            G = _to_networkx(self.terrain_map.terrain_graph.get_graph(agent.robot_type))
            path = nx.astar_path(
                G,
                source=source,
                target=target,
                heuristic=lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]),
                weight="weight",
            )
            state.current_step = path[1]

        return _get_direction_to_cell(current_pos, state.current_step)



# ─── Geometry / Graph Helpers ─────────────────────────────────────────────────

def _step_is_finished(position: tuple, goal: tuple) -> bool:
    """True when position is within a small radius of the cell centre."""
    radius = 0.05
    goal_x = int(goal[0]) + 0.5
    goal_y = int(goal[1]) + 0.5
    return (goal_x - radius < position[0] < goal_x + radius and
            goal_y - radius < position[1] < goal_y + radius)


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