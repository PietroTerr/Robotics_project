import math

from TerrainMap import TerrainMap
from motion import Drone, Rover, Scout
import networkx as nx

class Governor:
    DRONE_DETERMINISTIC = True

    def __init__(self, map_information: TerrainMap, rover: Rover, scout: Scout, drone: Drone, target: tuple):
        self.map_information = map_information
        self.rover = rover
        self.scout = scout
        self.drone = drone
        self.TARGET = target

    def get_heading(self):
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
        if self.DRONE_DETERMINISTIC:
            heading = _get_straight_direction((self.drone.x, self.drone.y), self.TARGET)
            return heading

def to_networkx(graph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node, neighbours in graph.items():
        for nb, weight in neighbours.items():
            G.add_edge(node, nb, weight=weight)
    return G


# ─── Geometry Helpers ─────────────────────────────────────────────────────────
def _get_straight_direction(start: tuple, end: tuple) -> tuple:
    """
    Return the angle for straight direction between start and end.
    """
    dx = end[0] - start[0]
    dy =end[1] - start[1]

    # math.atan2(dy, dx) restituisce l'angolo in radianti
    # tra il semiasse positivo x e il punto (dx, dy)
    heading = math.atan2(dy, dx)

    return heading

