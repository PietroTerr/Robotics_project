import math

from TerrainMap import TerrainMap
from motion import Drone, Rover, Scout
import networkx as nx

class Governor:


    def __init__(self, terrain_map: TerrainMap, rover: Rover, scout: Scout, drone: Drone,start:tuple, target: tuple):
        self.terrain_map = terrain_map
        self.rover = rover
        self.scout = scout
        self.drone = drone
        self.TARGET = target
        self.START = start

        self.DRONE_DETERMINISTIC = True
        self.drone_goal = None # None if the drone doesn't have a current goal
        self.drone_must_recharge = False

        self.SCOUT_DETERMINISTIC = True


    def get_heading(self, drone_battery):
        """Return the heading of movement for each agent. This is a placeholder function and should be implemented with actual logic to determine the heading based on the current state of the agents and the environment."""
        if self.drone.x < 40 and self.drone.y<40 and self.rover.x < 20 and self.rover.y < 20 and self.scout.y < 20 and self.scout.x < 20:
            drone = self.get_drone_heading()
            scout = math.pi/5
            rover = math.pi/6
        else:
            drone = None
            scout = None
            rover = None
        return rover, scout, drone

    def get_drone_heading(self):
        if self.drone_must_recharge:
            return None

        if self.DRONE_DETERMINISTIC:
            heading = _get_straight_direction_to_a_cell((self.drone.x, self.drone.y), self.TARGET)
            if does_have_reached_cell((self.drone.x, self.drone.y), self.TARGET):
                self.DRONE_DETERMINISTIC = False
            return heading

        if self.drone_goal is None or does_have_reached_cell((self.drone.x, self.drone.y), self.drone_goal):
            G = to_networkx(self.terrain_map.terrain_graph.get_graph("drone"))
            source = _to_cell_coords((self.drone.x, self.drone.y))
            target = _to_cell_coords(self.TARGET)

            if source not in G or target not in G:
                # Graph may still be sparse at startup; keep moving toward mission target.
                return _get_straight_direction_to_a_cell((self.drone.x, self.drone.y), target)

            try:
                path = nx.astar_path(
                    G,
                    source=source,
                    target=target,
                    heuristic=lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1]),
                    weight="weight",
                )
            except (nx.NodeNotFound, nx.NetworkXNoPath):
                return _get_straight_direction_to_a_cell((self.drone.x, self.drone.y), target)

            if len(path) < 2:
                self.drone_goal = None
                return None

            self.drone_goal = path[1]
            drone_heading = _get_straight_direction_to_a_cell((self.drone.x, self.drone.y), self.drone_goal)
            return drone_heading
        else:
            return _get_straight_direction_to_a_cell((self.drone.x, self.drone.y), self.drone_goal)

    def get_scout_heading(self,):
        if self.SCOUT_DETERMINISTIC:
            heading = _get_straight_direction_to_a_cell((self.scout.x, self.scout.y), self.TARGET)
            if does_have_reached_cell((self.scout.x, self.scout.y), self.TARGET):
                self.DRONE_DETERMINISTIC = False
            return heading



def does_have_reached_cell(position: tuple, goal: tuple) -> bool:
    """
    Return True if the position is in the goal or not. Goal is achieved if position is near the center of the cell
    """
    radius = 0.05
    goal_x = int(goal[0]) + 0.5
    goal_y = int(goal[1]) + 0.5
    if goal_x-radius < position[0] < goal_x+radius:
        if goal_y-radius < position[1] < goal_y+radius:
            return True
    return False

def to_networkx(graph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node in graph:
        for nb, weight in graph[node].items():
            G.add_edge(node, nb, weight=weight)
    return G


def _to_cell_coords(position: tuple[float, float]) -> tuple[int, int]:
    return int(position[0]), int(position[1])


# ─── Geometry Helpers ─────────────────────────────────────────────────────────
def _get_straight_direction_to_a_cell(start: tuple, end: tuple) -> float:
    """
    Return the angle for straight direction between start and end.
    """
    end_x = end[0] + 0.5
    end_y = end[1] + 0.5
    dx = end_x - start[0]
    dy = end_y - start[1]

    # math.atan2(dy, dx) restituisce l'angolo in radianti
    # tra il semiasse positivo x e il punto (dx, dy)
    heading = math.atan2(dy, dx)
    return heading

