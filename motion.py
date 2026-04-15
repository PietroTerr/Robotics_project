import math
from typing import Tuple, List, Optional

from src.map_api import MapAPI

class RobotMovementBase:
    """Base class for handling robot behavior constraints."""
    
    def __init__(self, map_api: MapAPI, robot_id: str, robot_type: str, start_pos: Tuple[float, float]):
        self.map_api = map_api
        self.robot_id = robot_id
        self.robot_type = robot_type
        self.x, self.y = start_pos
        self.total_time_spent = 0.0
        
        self.map_api.register_robot(self.robot_id, self.robot_type)
        
    def perceive(self):
        """Standard perception, can be filtered by subclasses."""
        return self.map_api.perceive(self.robot_id, (self.x, self.y))


class Drone(RobotMovementBase):
    """
    Drone constraints:
    - Speed: 2 cells/s
    - TOF (Time Of Flight): 5 minutes (300 seconds)
    - Solar recharging time: 1 hr (3600 seconds)
    - Data: Collects only "observable" terrain data.
    """
    
    def __init__(self, map_api: MapAPI, robot_id: str, start_pos: Tuple[float, float]):
        super().__init__(map_api, robot_id, "drone", start_pos)
        self.speed = 2.0
        self.max_tof = 300.0
        self.recharge_time = 3600.0
        
        self.flight_clock = 0.0
        self.recharge_cycles = 0

    def perceive(self):
        """
        Drones only collect 'observable' data from the top.
        In this context, we return the full perceive block, but downstream memory
        should know that Drone measurements are purely visual (texture, slope).
        """
        return super().perceive()

    def step_towards(self, target_x: float, target_y: float, dt: float) -> Tuple[bool, bool]:
        """
        Moves the drone towards a target waypoint by dt seconds.
        Returns a tuple: (reached_target, forced_recharge)
        """
        forced_recharge = False
        # Check battery / TOF
        if self.flight_clock >= self.max_tof:
            # Must recharge before continuing
            self.total_time_spent += self.recharge_time
            self.flight_clock = 0.0
            self.recharge_cycles += 1
            forced_recharge = True
            
        dist = math.hypot(target_x - self.x, target_y - self.y)
        if dist < 0.1:
            return True, forced_recharge  # Reached target
            
        orientation = math.atan2(target_y - self.y, target_x - self.x)
        
        # Make the API step
        result = self.map_api.step(
            robot_id=self.robot_id,
            position=(self.x, self.y),
            command_velocity=self.speed,
            command_orientation=orientation
        )
        
        # Update physical coordinates
        self.x += result.actual_velocity * dt * math.cos(orientation)
        self.y += result.actual_velocity * dt * math.sin(orientation)
        
        # Advance clocks
        self.flight_clock += dt
        self.total_time_spent += dt
        
        return False, forced_recharge


class Scout(RobotMovementBase):
    """
    Scout: Ground vehicle, logs physical terrain data.
    - Speed: 0.5 cells/s
    - Affected by terrain traversal penalties.
    - Captures stuck events but forces movement through them to continue exploring.
    """
    def __init__(self, map_api: MapAPI, robot_id: str, start_pos: Tuple[float, float]):
        super().__init__(map_api, robot_id, "scout", start_pos)
        self.speed = 0.5
        self.stuck_cells: List[Tuple[int, int]] = []
        self.stuck_count = 0
        self.terrain_features = {}  # Stores {"slope": float, "uphill_angle": float} per cell

    def perceive(self):
        """
        Scout records slope and uphill_angle specifically to evaluate 
        directional slope factors and choose optimal traversal orientations.
        """
        observations = super().perceive()
        for obs in observations:
            cell_coords = (int(round(obs.x)), int(round(obs.y)))
            if cell_coords not in self.terrain_features:
                self.terrain_features[cell_coords] = {
                    "slope": obs.features.get("slope", 0.0),
                    "uphill_angle": obs.features.get("uphill_angle", 0.0)
                }
        return observations

    def step_towards(self, target_x: float, target_y: float, dt: float) -> Tuple[bool, bool, Optional[Tuple[int, int]]]:
        """
        Moves the scout towards a target waypoint.
        Returns a tuple: (reached_target, was_stuck, stuck_cell_coordinates)
        """
        dist = math.hypot(target_x - self.x, target_y - self.y)
        if dist < 0.1:
            return True, False, None  # Reached target
            
        orientation = math.atan2(target_y - self.y, target_x - self.x)
        
        result = self.map_api.step(
            robot_id=self.robot_id,
            position=(self.x, self.y),
            command_velocity=self.speed,
            command_orientation=orientation
        )
        
        was_stuck = False
        stuck_cell = None
        
        if result.is_stuck:
            self.stuck_count += 1
            stuck_cell = (int(round(self.x)), int(round(self.y)))
            if stuck_cell not in self.stuck_cells:
                self.stuck_cells.append(stuck_cell)
            was_stuck = True
            
            # Force movement regardless of being stuck
            self.x += self.speed * dt * math.cos(orientation)
            self.y += self.speed * dt * math.sin(orientation)
        else:
            # Move normally using actual velocity modulated by terrain
            self.x += result.actual_velocity * dt * math.cos(orientation)
            self.y += result.actual_velocity * dt * math.sin(orientation)
            
        self.total_time_spent += dt
        
        return False, was_stuck, stuck_cell


class Rover(RobotMovementBase):
    """
    Rover: Heavy ground vehicle, follows pre-optimized trajectories.
    - Speed: 0.1 cells/s
    - Can get immobilized by stuck events.
    - Does not `perceive` (runs blindly assuming the path is optimized).
    """
    def __init__(self, map_api: MapAPI, robot_id: str, start_pos: Tuple[float, float]):
        super().__init__(map_api, robot_id, "rover", start_pos)
        self.speed = 0.1
        self.status = "OPERATIONAL"

    def perceive(self):
        """
        Rover does not perceive the environment. It relies solely on the 
        global map already built by the Drone and Scout.
        """
        return []

    def step_towards(self, target_x: float, target_y: float, dt: float) -> Tuple[bool, bool]:
        """
        Moves the rover towards a target waypoint.
        If it hits a stuck event, it becomes permanently immobilized.
        Returns a tuple: (reached_target, is_stuck)
        """
        if self.status == "STUCK":
            return False, True
            
        dist = math.hypot(target_x - self.x, target_y - self.y)
        if dist < 0.1:
            return True, False  # Reached target
            
        orientation = math.atan2(target_y - self.y, target_x - self.x)
        
        result = self.map_api.step(
            robot_id=self.robot_id,
            position=(self.x, self.y),
            command_velocity=self.speed,
            command_orientation=orientation
        )
        
        if result.is_stuck:
            self.status = "STUCK"
            self.total_time_spent += dt
            return False, True
            
        self.x += result.actual_velocity * dt * math.cos(orientation)
        self.y += result.actual_velocity * dt * math.sin(orientation)
        
        self.total_time_spent += dt
        
        return False, False
