from pathlib import Path
import math

from map_api import MapAPI

def load_map (csv_path: str, rng_seed: int):
    """Helper function to load a map from a CSV file."""
    workspace_dir = Path(__file__).resolve().parent
    csv_path = workspace_dir / "map_001_seed42.csv"
    map_api = MapAPI(csv_path=str(csv_path), rng_seed=42)
    return map_api

def simulate_and_perceive(map_api: MapAPI, robot_id: str, position: tuple, speed: float, orientation: float):
    """Helper function to perform a simulation step."""
    step_result = map_api.step(
        robot_id=robot_id,
        position=position,
        command_velocity=speed,
        command_orientation=orientation,
    )
    
    print("Step result:", step_result)
    observations = map_api.perceive(robot_id=robot_id, position=position)
    print(f"Perceive returned {len(observations)} observations at position {position}:")
    for obs in observations:
        print(obs)
    return step_result

