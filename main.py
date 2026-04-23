from Governor import Governor
from TerrainMap import TerrainMap
from motion import Drone, Scout, Rover, RobotMovementBase
from real_time_plot import RealTimePlot
from src.map_api import MapAPI


def main():

    map_api = get_map_api()
    start_pos = (25,25)

    drone = Drone(map_api, "drone_01", start_pos)
    scout = Scout(map_api, "scout_01", start_pos)
    rover = Rover(map_api, "rover_01", start_pos)

    terrain_map = TerrainMap()
    governor = Governor(terrain_map,rover,scout,drone)
    plotter = RealTimePlot(terrain_map,[rover,scout,drone])
    i=0
    while True:
        # -- Get heading for each agent
        (rover_heading, scout_heading, drone_heading) = governor.get_heading()
        if rover_heading is None:
            print("Rover Heading is None")
            drone.perceive()
            break # Reached target

        observations = {}
        observations.update(rover.perceive())
        observations.update(scout.perceive())
        observations.update(drone.perceive())

        movement_information = {}
        movement_information[rover.x, rover.y] = rover.step_towards(rover_heading)
        movement_information[scout.x, scout.y] = scout.step_towards(scout_heading)

        terrain_map.update_map(observations, movement_information)

def get_map_api():
    print("Loading MapAPI & Components...")
    csv_path = "src/map_001_seed42.csv"
    map_api = MapAPI(terrain=csv_path, rng_seed=42)
    return map_api

if __name__ == "__main__":
        main()