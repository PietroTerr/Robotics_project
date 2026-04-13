"""Minimal student workflow example for the student MapAPI.

Sequence:
1) load terrain from organizer-provided CSV
2) instantiate student map API
2) register robot
3) step once
4) perceive nearby cells
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.map_api import MapAPI


# 1) Load terrain from CSV provided in this same folder.
workspace_dir = Path(__file__).resolve().parent
csv_path = workspace_dir / "map_001_seed42.csv"

# 2) Instantiate student API with loaded terrain.
map_api = MapAPI(terrain=csv_path, rng_seed=42, time_step=0.2)

# 3) Register one robot.
robot_id = "rover_1"
map_api.register_robot(robot_id=robot_id, robot_type="rover")

# Fixed pose/command for this single-step example.
position = (3, 3)
command_velocity = 1.0
command_orientation = 0.0

# 4) Step once.
step_result = map_api.step(
    robot_id=robot_id,
    position=position,
    command_velocity=command_velocity,
    command_orientation=command_orientation,
)
print("Step result:")
print(f"  actual_velocity={step_result.actual_velocity:.3f}")
print(f"  is_stuck={step_result.is_stuck}")
print(f"  battery_value={step_result.battery_value:.3f}")

# 5) Perceive local map around the same position.
observations = map_api.perceive(robot_id=robot_id, position=position)
print(f"Perceive returned {len(observations)} observations at position {position}:")
for obs in observations:
    print(obs)

