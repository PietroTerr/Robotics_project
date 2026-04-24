from matplotlib import pyplot as plt

from Governor import Governor, AgentState
from TerrainMap import TerrainMap
from motion import Drone, Scout, Rover
from real_time_plot import MapPlotter
from src.map_api import MapAPI


def main():
    map_api = get_map_api()
    start_pos = (5, 5)
    target = (7, 7)
    drone = Drone(map_api, "drone", start_pos)
    scout = Scout(map_api, "scout", start_pos)
    rover = Rover(map_api, "rover", start_pos)

    plotter = MapPlotter(grid_size=50)
    terrain_map = TerrainMap()

    drone_state = AgentState(
        agent=drone,
        goals=[target,start_pos],
        terminal=True,
    )
    scout_state = AgentState(
        agent=scout,
        goals=[target, start_pos],
    )
    rover_state = AgentState(
        agent=rover,
        goals=[target],
    )

    governor = Governor(terrain_map, [drone_state, scout_state, rover_state])

    while True:
        # -- Get heading for each agent
        headings = governor.get_headings()
        if governor.done:
            break

        observations = {}
        observations.update(scout.perceive())
        observations.update(drone.perceive())

        movement_information = {}
        step_rover_result = rover.step_towards(headings["rover"])
        movement_information[rover.x, rover.y] = step_rover_result
        step_scout_result = scout.step_towards(headings["scout"])
        movement_information[scout.x, scout.y] = step_scout_result
        drone.step_towards(headings["drone"])
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

        # print(agents_positions)
    print("Done")
    plt.ioff()
    plt.show()



def get_map_api():
    print("Loading MapAPI & Components...")
    csv_path = "src/map_001_seed42.csv"
    map_api = MapAPI(terrain=csv_path, rng_seed=42)
    return map_api


if __name__ == "__main__":
    main()
