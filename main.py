from data_management import TerrainMap
from Governor import Governor
from motion import Drone, Scout, Rover
from real_time_plot import RealTimePlot
from src.map_api import MapAPI


def main():

    dt = 0.1  # Time step for the simulation
    map_api = get_map_api()
    start_pos = (0,0)

    drone = Drone(map_api, "drone_01", start_pos)
    scout = Scout(map_api, "scout_01", start_pos)
    rover = Rover(map_api, "rover_01", start_pos)

    map = TerrainMap()
    governor = Governor(map)
    plotter = RealTimePlot(map, [drone, scout, rover])

    while True:

        # -- Get heading for each agent
        (rover_heading, scout_heading, drone_heading) = governor.get_heading()
        if rover_heading is None: break # Reached target


        # -- Move agents
        r_is_stuck, r_actual_velocity = rover.step_towards_2(rover_heading)
        s_is_stuck, s_actual_velocity = scout.step_towards_2(scout_heading)
        drone.step_towards_2(drone_heading)

        movement_info = {
            "rover": {"position": (rover.x, rover.y),"heading": rover_heading, "is_stuck": r_is_stuck, "actual_velocity": r_actual_velocity},
            "scout": {"position": (scout.x, scout.y), "heading": scout_heading, "is_stuck": s_is_stuck, "actual_velocity": s_actual_velocity},
            "drone": {"position": (drone.x, drone.y), "heading": drone_heading, },
        }


        # -- Perceive
        d_obs = drone.perceive()
        map.store_observation(d_obs)

        s_obs = scout.perceive()
        map.store_observation(s_obs)
        map.refresh_estimation(movement_info)

        r_obs = rover.perceive()
        map.store_observation(r_obs)
        plotter.plot()

def get_map_api():
    print("Loading MapAPI & Components...")
    csv_path = "src/map_001_seed42.csv"
    map_api = MapAPI(terrain=csv_path, rng_seed=42)
    return map_api