import math
import os

from Governor import Governor, AgentState
from SimulationLogger import SimulationLogger
from TerrainMap import TerrainMap
from motion import Drone, Scout, Rover
from real_time_plot import MapPlotter
from src.map_api import MapAPI


def main(terrain_map,
         start_pos: tuple[int, int],
         target: tuple[int, int],
         step_limit=100000,
         revisit_penalty_scout: float = 3.0,
         revisit_penalty_drone: float = 2.0,
         pessimistic_default: float = 0.1,
         zig_lookahead=5.0,
         zig_width=4.0,
         live=False
         ):
    map_name = terrain_map
    base_dir = os.path.dirname(os.path.abspath(__file__))
    map_path = os.path.join(base_dir, "generated_maps", f"{map_name}.csv")
    map_api = get_map_api(map_path)
    start_pos = start_pos
    target = target
    drone = Drone(map_api, "drone", start_pos)
    scout = Scout(map_api, "scout", start_pos)
    rover = Rover(map_api, "rover", start_pos)

    ten_seconds = int(1 / scout.dt * 10)
    sim_logger = SimulationLogger(log_interval=ten_seconds)
    plotter = MapPlotter(grid_size=50, live=live)
    terrain_map = TerrainMap(revisit_penalty_scout=revisit_penalty_scout, revisit_penalty_drone=revisit_penalty_drone,
                             pessimistic_default=pessimistic_default)

    drone_state = AgentState(
        agent=drone,
        goals=[target, start_pos]
    )
    scout_state = AgentState(
        agent=scout,
        goals=[target, start_pos],
        use_zigzag=True
    )
    rover_state = AgentState(
        agent=rover,
        goals=[target],
        terminal=True,
    )
    governor = Governor(terrain_map, [drone_state, scout_state, rover_state], zig_lookahead=zig_lookahead,
                        zig_width=zig_width)

    sim_logger.start(total_steps=None,
                     start=start_pos,
                     target=target, )
    step = 0
    while True:
        step += 1
        time_elapsed = step * scout.dt
        if time_elapsed > step_limit:
            reached_target = False
            break

        # -- Get heading for each agent
        headings = governor.get_headings()
        if governor.done:
            reached_target = True
            break

        # ------ Perceive ----------
        observations = {}
        observations.update(scout.perceive())
        observations.update(drone.perceive())

        # ------ Step ----------
        movement_information = {}
        if drone_state.finished:
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
    plotter.save(f"{map_name}.mp4", fps=15)

    perceive_calls = drone.method_counts["perceive"] + scout.method_counts["perceive"]
    step_calls = drone.method_counts["step"] + scout.method_counts["step"] + rover.method_counts["step"]
    stuck_event = rover.get_stuck_events_number()

    drone_travel = drone.get_travel()
    scout_travel = scout.get_travel()
    rover_travel = rover.get_travel()

    last_distance_from_target = math.sqrt((rover.x - target[0]) ** 2 + (rover.y - target[1]) ** 2)

    sim_logger.end(
        map=terrain_map,
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
    map_api = MapAPI(terrain=csv_path, time_step=0.90)
    return map_api


if __name__ == "__main__":
    terrain_map = "map_013_seed13"
    start = (10, 1)
    target = (40, 37)
    revisit_penalty_scout: float = 10.0
    revisit_penalty_drone: float = 10.0
    pessimistic_default: float = 0.05
    zig_lookahead = 6.0
    zig_width = 10.0
    main(terrain_map=terrain_map, start_pos=start, target=target, step_limit=100000,
         revisit_penalty_scout=revisit_penalty_scout, revisit_penalty_drone=revisit_penalty_drone,
         pessimistic_default=pessimistic_default, zig_lookahead=zig_lookahead, zig_width=zig_width, live=False)
