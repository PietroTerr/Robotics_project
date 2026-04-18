"""
Motion Module
=============

This module defines the movement capabilities and constraints for various types of robots 
(Drone, Scout, Rover) used in the exploration simulation. Each robot type inherits from 
the `RobotMovementBase` class and implements specific behaviors such as perception, 
step execution, and constraints (e.g., drone battery flight time, scout terrain interaction, 
and rover immobility on stuck events).

Classes:
- RobotMovementBase: Base class for common robot state and API registration.
- Drone: Fast moving aerial vehicle that requires periodic solar recharging and observes observable data.
- Scout: Ground vehicle that logs physical constraints and powers through stuck events.
- Rover: Heavy ground vehicle that relies on purely optimized paths and gets immobilized by stuck events.

Usage in a main() method:
-------------------------
To use these robot classes in your main orchestrator, initialize a MapAPI instance and pass it 
along with an ID and starting coordinates to the respective robot constructors. Then call 
`.perceive()` to gather data and `.step_towards()` iteratively to advance physically in the simulation.

Example:
    from src.map_api import MapAPI

    def main():
        map_api = MapAPI()
        start_pos = (5.0, 5.0)
        target_pos = (15.0, 20.0)
        dt = 1.0  # 1 second simulation step

        # Standard Initialization
        drone = Drone(map_api, "drone__01", start_pos)
        scout = Scout(map_api, "scout__01", start_pos)
        rover = Rover(map_api, "rover__01", start_pos)

        # 1. Perceive surroundings (Only Drone and Scout have perception)
        drone.perceive()
        scout.perceive()

        # 2. Step towards destination inside a simulation loop
        while True:
            # Move Drone
            d_reached, forced_recharge = drone.step_towards(target_pos[0], target_pos[1], dt)
            
            # Move Scout
            s_reached, was_stuck, stuck_pos = scout.step_towards(target_pos[0], target_pos[1], dt)
            
            # Move Rover
            r_reached, r_stuck = rover.step_towards(target_pos[0], target_pos[1], dt)
            
            if d_reached and s_reached and r_reached:
                print("All robots arrived!")
                break

    if __name__ == "__main__":
        main()
"""

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

        # ---  temporary
        self.dt = 0.1
        self.speed = 1.0

        
    def perceive(self):
        """Standard perception, can be filtered by subclasses."""
        return self.map_api.perceive(self.robot_id, (self.x, self.y))
    def step_towards_2(self, heading):
        result     = self.map_api.step(
            robot_id=self.robot_id,
            position=(self.x, self.y),
            command_velocity=self.speed,
            command_orientation=heading
        )
        # Update physical coordinates
        self.x += result.actual_velocity * self.dt * math.cos(heading)
        self.y += result.actual_velocity * self.dt * math.sin(heading)

        return result.is_stuck, result.actual_velocity

class Drone(RobotMovementBase):
    """
    Drone constraints:
    - Speed: 1.0 cells/s
    - TOF (Time Of Flight): 5 minutes (300 seconds)
    - Solar recharging time: 1 hr (3600 seconds)
    - Data: Collects only "observable" terrain data.
    """
    
    def __init__(self, map_api: MapAPI, robot_id: str, start_pos: Tuple[float, float]):
        super().__init__(map_api, robot_id, "drone", start_pos)
        self.speed = 1.0
        self.max_tof = 300.0
        self.recharge_time = 3600.0
        
        self.flight_clock = 0.0
        self.recharge_cycles = 0
        self.battery_state = 1 # 1.0 = fully charged, 0.0 = depleted

    def perceive(self):
        """
        Drones only collect 'observable' data from the top.
        In this context, we return the full perceive block, but downstream memory
        should know that Drone measurements are purely visual (texture, slope).
        """
        return super().perceive()
    def step_towards_2(self, heading):
        # if heading = None the drone must recharge
        if heading is None:
            heading = 0.0
            self.map_api.step(self.robot_id, (self.x, self.y), 0.0, heading)
            self.battery_state *= self.battery_state * 0.002 * 1/self.dt # battery recharges at 0.002 per second of rest (solar recharge)
            return (self.x, self.y), self.battery_state
        result = self.map_api.step(
            robot_id=self.robot_id,
            position=(self.x, self.y),
            command_velocity=self.speed,
            command_orientation=heading
        )
        # Update physical coordinates
        self.x += result.actual_velocity * self.dt * math.cos(heading)
        self.y += result.actual_velocity * self.dt * math.sin(heading)
        self.battery_state -= self.battery_state * 0.02 * 1/self.dt # battery depletes at 0.02 per second of fligh
        return None, result.actual_velocity


    def step_towards(self, target_x: float, target_y: float, dt: float) -> Tuple[bool, bool]:
        """
        Moves the drone towards a target waypoint by dt seconds.
        Returns a tuple: (reached_target, forced_recharge)
        """
        forced_recharge = getattr(self, "is_recharging", False)

        dist = math.hypot(target_x - self.x, target_y - self.y)
        if dist < 0.1:
            return True, forced_recharge  # Reached target

        orientation = math.atan2(target_y - self.y, target_x - self.x)

        # If currently recharging, just rest
        if forced_recharge:
            res = self.map_api.step(self.robot_id, (self.x, self.y), 0.0, orientation)
            self.total_time_spent += dt
            if hasattr(res, "battery_value") and res.battery_value >= 0.999:
                self.is_recharging = False
                self.flight_clock = 0.0
                self.recharge_cycles += 1
            return False, True

        # Make the API step
        result = self.map_api.step(
            robot_id=self.robot_id,
            position=(self.x, self.y),
            command_velocity=self.speed,
            command_orientation=orientation
        )
        
        # If API backend says battery is critically low (depleted), we must recharge
        if hasattr(result, "battery_value") and result.battery_value <= 0.02:
            self.is_recharging = True
            forced_recharge = True
            return False, True
        
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
    - Speed: 0.05 cells/s
    - Affected by terrain traversal penalties.
    - Captures stuck events but forces movement through them to continue exploring.
    """
    def __init__(self, map_api: MapAPI, robot_id: str, start_pos: Tuple[float, float]):
        super().__init__(map_api, robot_id, "scout", start_pos)
        self.speed = 0.05
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
    - Speed: 0.01 cells/s
    - Can get immobilized by stuck events.
    - Does not `perceive` (runs blindly assuming the path is optimized).
    """
    def __init__(self, map_api: MapAPI, robot_id: str, start_pos: Tuple[float, float]):
        super().__init__(map_api, robot_id, "rover", start_pos)
        self.speed = 0.01
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
