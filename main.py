from matplotlib import pyplot as plt

from Governor import Governor
from TerrainMap import TerrainMap
from motion import Drone, Scout, Rover
from real_time_plot import MapPlotter
from src.map_api import MapAPI


def main():

    map_api = get_map_api()
    start_pos = (5,5)
    target = (10,40)
    drone = Drone(map_api, "drone_01", start_pos)
    scout = Scout(map_api, "scout_01", start_pos)
    rover = Rover(map_api, "rover_01", start_pos)

    plotter = MapPlotter(grid_size=50)

    terrain_map = TerrainMap()
    governor = Governor(terrain_map, rover, scout, drone, start_pos, target)
    while True:
        # -- Get heading for each agent
        (rover_heading, scout_heading, drone_heading) = governor.get_heading()
        if rover_heading is None:
            break

        observations = {}
        observations.update(scout.perceive())
        observations.update(drone.perceive())

        movement_information = {}
        step_rover_result = rover.step_towards(rover_heading)
        movement_information[rover.x, rover.y] = step_rover_result
        step_scout_result = scout.step_towards(scout_heading)
        movement_information[scout.x, scout.y] = step_scout_result
        drone.step_towards(drone_heading)
        terrain_map.update_map(observations, movement_information)

        agents_positions = [
            (rover.x, rover.y),
            (scout.x, scout.y),
            (drone.x, drone.y)
        ]
        # 2. Package the snapshot using your method!
        snapshot = {
            "grid": terrain_map.get_grid_snapshot(),
            "agents": agents_positions
        }

        plotter.update(snapshot["grid"], snapshot["agents"])

        #print(agents_positions)

    plt.ioff()
    plt.show()

def get_map_api():
    print("Loading MapAPI & Components...")
    csv_path = "src/map_001_seed42.csv"
    map_api = MapAPI(terrain=csv_path, rng_seed=42)
    return map_api

if __name__ == "__main__":
    main()
