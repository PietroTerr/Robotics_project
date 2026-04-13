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

def move_rover_to_target(map_api: MapAPI, robot_id: str, start: tuple, target: tuple, speed: float, dt: float = 0.1):
    x, y = float(start[0]), float(start[1])
    tx, ty = float(target[0]), float(target[1])
    total_time = 0.0
    stuck_count = 0
 
    print(f"Start: ({x:.1f}, {y:.1f})  ->  Target: ({tx:.1f}, {ty:.1f})")
 
    while True:
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx**2 + dy**2)
 
        if dist < 0.001:
            print(f"REACHED target! Steps: {int(total_time/dt)}, Stuck events: {stuck_count}")
            break
 
        orientation = math.atan2(dy, dx)
 
        result = map_api.step(
            robot_id=robot_id,
            position=(x, y),
            command_velocity=speed,
            command_orientation=orientation,
        )

        if result.is_stuck:
            stuck_count += 1
            print(f"  t={total_time:.1f}s  STUCK at pos=({x:.2f}, {y:.2f})  dist={dist:.2f}")
        
        else:
            x += result.actual_velocity * dt * math.cos(orientation)
            y += result.actual_velocity * dt * math.sin(orientation)

        total_time += dt
 
        if total_time % 50 == 0:
            print(f"  t={total_time:.0f}s  pos=({x:.2f}, {y:.2f})  dist={dist:.2f}")
 
    return (x, y), total_time, stuck_count
 

