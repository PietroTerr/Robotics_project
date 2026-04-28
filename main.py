import math
from pathlib import Path

from Governor import Governor, AgentState
from SimulationLogger import SimulationLogger
from TerrainMap import TerrainMap
import cost_function_calculator
from motion import Drone, Scout, Rover
from real_time_plot import MapPlotter
from src.map_api import MapAPI

BASE_DIR = Path(__file__).resolve().parent


def main(map, live=False):
    map_api = get_map_api("generated_maps/" + map + ".csv")
    start_pos = (10, 1)
    target = (40, 37)
    drone = Drone(map_api, "drone", start_pos)
    scout = Scout(map_api, "scout", start_pos)
    rover = Rover(map_api, "rover", start_pos)

    ten_seconds = int(1 / scout.dt * 10)
    sim_logger = SimulationLogger(log_interval=ten_seconds)
    plotter = MapPlotter(grid_size=50, live=live)
    terrain_map = TerrainMap()

    drone_state = AgentState(
        agent=drone,
        goals=[target, start_pos]
    )
    scout_state = AgentState(
        agent=scout,
        goals=[target, start_pos],
        use_zigzag=True,
    )
    rover_state = AgentState(
        agent=rover,
        goals=[target],
        terminal=True,
    )

    governor = Governor(terrain_map, [drone_state, scout_state, rover_state])

    sim_logger.start(total_steps=None,
                     start=start_pos,
                     target=target, )
    step = 0
    while True:
        step += 1
        time_elapsed = step * scout.dt

        # -- Get heading for each agent
        headings = governor.get_headings()
        if governor.done:
            reached_target = True
            break
        if time_elapsed > 100000:
            reached_target = False
            break

        # ------ Perceive ----------
        observations = {}
        observations.update(scout.perceive())
        observations.update(drone.perceive())

        # ------ Step ----------
        movement_information = {}

        if drone_state.finished:  # rover wait that drone has done a full journey
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
        plotter.update(snapshot["grid"], governor.agents)
        sim_logger.log_step(step=step, simulation_time=time_elapsed, drone_position=(drone.x, drone.y),
                            scout_position=(scout.x, scout.y),
                            rover_position=(rover.x, rover.y))
    plotter.save(map, fps=15)

    perceive_calls = drone.method_counts["perceive"] + scout.method_counts["perceive"]
    step_calls = drone.method_counts["step"] + scout.method_counts["step"] + rover.method_counts["step"]
    stuck_event = rover.get_stuck_events_number()

    drone_travel = drone.get_travel()
    scout_travel = scout.get_travel()
    rover_travel = rover.get_travel()

    last_distance_from_target = math.sqrt((rover.x - target[0]) ** 2 + (rover.y - target[1]) ** 2)

    single_run_score = cost_function_calculator.compute_run_cost(
        total_time=time_elapsed,
        stuck_events=stuck_event,
        final_distance=last_distance_from_target,
    )
    print(
        f"Single-run score: {single_run_score:.2f} "
        f"(T={time_elapsed:.2f}, stuck={stuck_event}, dist={last_distance_from_target:.2f})"
    )

    sim_logger.end(
        map=map,
        reached_target=reached_target,
        last_distance_from_target=last_distance_from_target,
        time_elapsed=time_elapsed,
        drone_travel=drone_travel, scout_travel=scout_travel, rover_travel=rover_travel,
        perceive_calls=perceive_calls, step_calls=step_calls,
        stuck_events=stuck_event,
    )

    return reached_target, last_distance_from_target, time_elapsed, drone_travel, scout_travel, rover_travel, perceive_calls, step_calls, stuck_event


def get_map_api(csv_path):
    print("Loading MapAPI & Components...")
    csv_path = Path(csv_path)
    if not csv_path.is_absolute():
        csv_path = BASE_DIR / csv_path
    map_api = MapAPI(terrain=str(csv_path), time_step=0.90)
    return map_api


if __name__ == "__main__":
    main(map="map_013_seed13")
