
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
    finished: bool = False # True when agent has reached final goal

    # --- New Oscillation Fields ---
    use_oscillation: bool = False
    oscillation_step: int = 0

    @property
    def current_goal(self) -> tuple[float, float]:
        return self.goals[self.goal_index]

    def advance_goal(self):
        if self.goal_index == len(self.goals) - 1:
            self.finished = True
        self.goal_index = (self.goal_index + 1) % len(self.goals)


class Governor:
    def __init__(self, terrain_map: TerrainMap, agents: list[AgentState],
                 osc_amplitude=0.9, osc_frequency=0.01):
        self.terrain_map = terrain_map
        self.agents: list[AgentState] = agents
        self.done = False # Terminal agent has reached target

        # Oscillation settings (amplitude in radians, frequency in steps)
        self.osc_amplitude = osc_amplitude
        self.osc_frequency = osc_frequency

    def get_headings(self) -> dict[str, float | None]:
        return {
            state.agent.robot_id: self._get_agent_heading(state)
            for state in self.agents
        }

    def _get_agent_heading(self, state: AgentState) -> float | None:
        if getattr(state.agent, "needs_pause", False): # if agent (in motion) has a "needs_pause" property, and it's True, skip movement
            return None

        agent = state.agent
        current_pos = (agent.x, agent.y)

        # 1. Determine Target
        target_pos = state.current_goal

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
                # Grab the live graph view directly
                G = self.terrain_map.terrain_graph.get_graph(agent.robot_type)
                effective_source = source if source in G else _nearest_valid_source(source, G)
                if effective_source is None: return None

                # Use the custom A* implementation
                next_step = astar_graph(G, start=effective_source, goal=target)

                if next_step is not None:
                    state.current_step = next_step
                else:
                    # No path found; hold position and wait for next step.
                    print("ERROR IN GRAPH SEARCH: No path found from", source, "to", target, "for agent", agent.robot_id)
                    return None

        # 3. Calculate Heading and apply Oscillation
        heading = _get_direction_to_cell(current_pos, state.current_step)

        if state.use_oscillation:
            state.oscillation_step += 1
            # Adds a small sinusoidal offset to the heading
            heading += self.osc_amplitude * math.sin(self.osc_frequency * state.oscillation_step)

        return heading

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



import heapq
import math


def astar_graph(graph, start: tuple, goal: tuple):
    """
    A custom A* implementation that works directly with the TerrainGraph
    adjacency dict or _PenalizedView. Returns the NEXT step (coordinate tuple)
    to take towards the goal, or None if no path is found.
    """
    open_heap = [(0, start)]
    g = {start: 0}
    came_from = {}

    while open_heap:
        f, current = heapq.heappop(open_heap)

        if current == goal:
            # Reconstruct only the first step
            if current == start:
                return start
            while came_from.get(current) != start:
                current = came_from[current]
            return current  # next hop only

        # Extract neighbors from the dict or _PenalizedView
        if current in graph:
            neighbors = graph[current]
        else:
            neighbors = {}

        for nb, weight in neighbors.items():
            new_g = g[current] + weight

            # If we found a shorter path to the neighbor
            if new_g < g.get(nb, float('inf')):
                g[nb] = new_g
                # Heuristic: Euclidean distance
                f_score = new_g + math.hypot(nb[0] - goal[0], nb[1] - goal[1])
                heapq.heappush(open_heap, (f_score, nb))
                came_from[nb] = current

    return None