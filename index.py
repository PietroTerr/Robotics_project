from pathlib import Path

import map_api as m
import utils as u


def main():
    
    map = u.load_map(csv_path="map_001_seed42.csv", rng_seed=42)
    
    robot_id = "rover_1"
    map.register_robot(robot_id=robot_id, robot_type=m.RobotType.ROVER)
    
    position = (3, 3)
    speed = 0.1
    orientation = 0.2
    
    start    = (3, 3)
    target   = (10, 10) 
    speed    = 0.1        
    dt       = 0.01 
    final_pos, elapsed, stuck = u.move_rover_to_target(map, robot_id, start, target, speed, dt)
    print(f"\nDone. Final pos: ({final_pos[0]:.2f}, {final_pos[1]:.2f}), Time: {elapsed:.1f}s, Stuck: {stuck}")
    u.simulate_and_perceive(map, robot_id, final_pos, speed, orientation)

if __name__ == "__main__":
    main()
