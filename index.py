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
    u.simulate(map, robot_id, position, speed, orientation)
    
    # target_position = (10, 10)
    # u.move_to_position(map, robot_id, position, target_position, speed) 

if __name__ == "__main__":
    main()
