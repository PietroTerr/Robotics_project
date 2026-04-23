import math

from motion import Drone, Rover, Scout
import networkx as nx

class Governor:

    def __init__(self, map_information, rover: Rover, scout: Scout, drone: Drone):
        self.map_information = map_information
        self.rover = rover
        self.scout = scout
        self.drone = drone

    def get_heading(self):
        """Return the heading of movement for each agent. This is a placeholder function and should be implemented with actual logic to determine the heading based on the current state of the agents and the environment."""
        if self.drone.x < 20 and self.drone.y<20 and self.rover.x < 20 and self.rover.y < 20 and self.scout.y < 20 and self.scout.x < 20:
            drone = math.pi/4  # drone = none if t has to recharge
            scout = math.pi/5
            rover = math.pi/6
        else:
            drone = None
            scout = None
            rover = None
        return rover, scout, drone

def to_networkx(graph) -> nx.DiGraph:
    G = nx.DiGraph()
    for node, neighbours in graph.items():
        for nb, weight in neighbours.items():
            G.add_edge(node, nb, weight=weight)
    return G

"""
# Rover: plans through real ground truth only
rover_graph = terrain_map.terrain_graph.get_graph("rover")

# Scout: observed graph, avoids already-traversed cells
G = to_networkx(terrain_map.terrain_graph.get_graph("scout"))
path = nx.astar_path(G, source=(0, 0), target=(49, 49),
                     heuristic=lambda a, b: math.hypot(a[0]-b[0], a[1]-b[1]),
                     weight="weight")


# Drone: observed graph, avoids already-scanned cells
drone_graph = terrain_map.terrain_graph.get_graph("drone")

# Pass any of these directly into your A* as the adjacency structure
path = astar(scout_graph, start=(0,0), goal=(49,49))"""