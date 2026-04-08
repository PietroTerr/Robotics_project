"""Minimal student workflow example for the student MapAPI.

Sequence:
1) instantiate map from organizer-provided CSV
2) register robot
3) step once
4) perceive nearby cells
"""

from pathlib import Path

from map_api import MapAPI, MapConfig, RobotType


# 1) Instantiate map from CSV provided in this same folder.
workspace_dir = Path(__file__).resolve().parent
csv_path = workspace_dir / "map_001_seed42.csv"
map_api = MapAPI(csv_path=str(csv_path), rng_seed=42)

# 2) Register one robot.
robot_id = "rover_1"
map_api.register_robot(robot_id=robot_id, robot_type=RobotType.ROVER)

# Fixed pose/command for this single-step example.
position = (3, 3)
command_velocity = 1.0
command_orientation = 0.0

# 3) Step once.
step_result = map_api.step(
    robot_id=robot_id,
    position=position,
    command_velocity=command_velocity,
    command_orientation=command_orientation,
)
print("Step result:", step_result)

# 4) Perceive local map around the same position.
observations = map_api.perceive(robot_id=robot_id, position=position)
print(f"Perceive returned {len(observations)} observations at position {position}:")
for obs in observations:
    print(obs)
