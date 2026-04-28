"""
Motion Module
=============

Movement and perception models for the three agents used in the simulation:

- `Drone`: aerial scout with battery and recharge constraints.
- `Scout`: ground robot that explores and logs terrain interaction data.
- `Rover`: ground robot used for conservative/goal-directed motion.

Design notes
------------
- All robots share a common API through `RobotMovementBase`.
- Positions are continuous floats `(x, y)` in map coordinates.
- Navigation controllers provide a heading (radians) to `step_towards`.
- The map backend (`MapAPI`) resolves dynamics (actual velocity, stuck state).

Typical loop
------------
1. Call `.perceive()` for sensing agents (drone/scout).
2. Ask a planner/governor for headings.
3. Call `.step_towards(heading)` on each robot.
4. Feed returned telemetry back into map/state estimators.
"""

import math
from typing import List, Tuple

from src.map_api import MapAPI


class RobotMovementBase:
    """
    Common robot interface and motion integration logic.

    Parameters
    ----------
    map_api : MapAPI
        External simulator/environment API.
    robot_id : str
        Unique identifier used by `MapAPI`.
    robot_type : str
        Semantic role ("drone", "scout", "rover"), used by planning logic.
    start_pos : tuple[float, float]
        Initial continuous position.
    """

    def __init__(
            self, map_api: MapAPI, robot_id: str, robot_type: str, start_pos: Tuple[float, float]
    ):
        self.map_api = map_api

        self.robot_type = robot_type
        self.x, self.y = start_pos

        self.robot_id = robot_id
        self.map_api.register_robot(self.robot_id, self.robot_type)

        # Simulation integration timestep (seconds).
        self.dt = 0.90
        # Commanded nominal speed (cells/s); subclasses override.
        self.speed = 1.0

        self.method_counts = {"step": 0, "perceive": 0}  # map_api.get_method_counts()
        self.path = [(self.x, self.y)]

    def perceive(self):
        """
        Sense nearby cells and return a normalized feature dict.

        Returns
        -------
        dict[tuple[int, int], dict[str, float | None]]
            Mapping from integer cell coordinates to observed features.
            Keys include: "texture", "color", "slope", "uphill_angle".
        """
        feature = {}
        obs = self.map_api.perceive(self.robot_id, (self.x, self.y))
        self.method_counts["perceive"] += 1

        for ob in obs:
            feature[(ob.x, ob.y)] = {
                "texture": ob.features.get("texture"),
                "color": ob.features.get("color"),
                "slope": ob.features.get("slope"),
                "uphill_angle": ob.features.get("uphill_angle"),
            }
        return feature

    def step_towards(self, heading):
        """
        Execute one movement step using a commanded heading.

        Parameters
        ----------
        heading : float
            Command orientation in radians.

        Returns
        -------
        dict[str, float | bool]
            Telemetry payload consumed by mapping/learning modules:
            - `heading`
            - `is_stuck`
            - `command_velocity`
            - `actual_velocity`
        """
        result = self.map_api.step(
            robot_id=self.robot_id,
            position=(self.x, self.y),
            command_velocity=self.speed,
            command_orientation=heading,
        )
        self.method_counts["step"] += 1
        # Integrate physical position from returned actual velocity.
        self.x += result.actual_velocity * self.dt * math.cos(heading)
        self.y += result.actual_velocity * self.dt * math.sin(heading)

        self.path.append((self.x, self.y))

        movement_information = {
            "heading": heading,
            "is_stuck": result.is_stuck,
            "command_velocity": self.speed,
            "actual_velocity": result.actual_velocity,
        }
        return movement_information

    def get_travel(self):
        """
        Calculate total travel distance from the path history.

        Returns
        -------
        float
            Total distance traveled along the path.
        """
        total_distance = 0.0
        for i in range(1, len(self.path)):
            x1, y1 = self.path[i - 1]
            x2, y2 = self.path[i]
            total_distance += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return total_distance


class Drone(RobotMovementBase):
    """
    Aerial robot with battery lifecycle and forced recharge behavior.

    Constraints
    -----------
    - Speed: 1.0 cells/s
    - Battery drains during movement, charges when paused
    - When battery reaches 0, drone enters recharge mode and must hold position

    Notes
    -----
    Battery rates in this implementation are scaled by `dt` both in coefficients
    and updates to preserve existing simulation behavior.
    """

    def __init__(self, map_api: MapAPI, robot_id: str, start_pos: Tuple[float, float]):
        super().__init__(map_api, robot_id, "drone", start_pos)
        self.speed = 1.0
        self._recharging = False

        # Internal battery model parameters.
        self._power_draw = 0.02  # per second
        self._battery_recharge = 0.002  # per second

        self.recharge_cycles = 0
        self.battery_state = 1  # 1.0 = full, 0.0 = empty

    @property
    def needs_pause(self) -> bool:
        """
        True when the drone should not receive movement commands.
        """
        return self.battery_state == 0.0 or (
                self.battery_state < 1.0 and self._recharging
        )

    def step_towards(self, heading):
        """
        Step drone motion or recharge cycle.

        Parameters
        ----------
        heading : float | None
            If `None`, drone remains stationary and recharges.

        Returns
        -------
        dict[str, float]
            `{"battery_state": ...}` after applying movement/recharge update.
        """
        # No heading means "hold position and recharge".
        if heading is None:
            heading = 0.0
            self.map_api.step(self.robot_id, (self.x, self.y), 0.0, heading)
            self.method_counts["step"] += 1

            # Recharge while idle.
            self.battery_state = min(
                1.0,
                self.battery_state + (self._battery_recharge * self.dt),
            )
            if self.battery_state == 1.0:
                self._recharging = False  # Fully recharged; can fly again.
            return {"battery_state": self.battery_state}

        super().step_towards(heading)

        # Drain battery during active flight.
        self.battery_state = max(
            0.0, self.battery_state - (self._power_draw * self.dt)
        )
        if self.battery_state == 0.0:
            print("Drone start recharging")
            self._recharging = True

        return {"battery_state": self.battery_state}


class Scout(RobotMovementBase):
    """
    Ground exploration robot.

    Characteristics
    ---------------
    - Speed: 0.05 cells/s
    - Exposes standard movement telemetry from base class
    - Keeps local bookkeeping for stuck events and terrain feature logs
    """

    def __init__(self, map_api: MapAPI, robot_id: str, start_pos: Tuple[float, float]):
        super().__init__(map_api, robot_id, "scout", start_pos)
        self.speed = 0.05

        # Tracking fields for analysis/metrics.
        self.stuck_cells: List[Tuple[int, int]] = []
        self.stuck_count = 0
        # Arbitrary per-cell feature cache, e.g. {"slope": ..., "uphill_angle": ...}.
        self.terrain_features = {}


class Rover(RobotMovementBase):
    """
    Slow, heavy ground robot used for robust target-reaching behavior.

    Characteristics
    ---------------
    - Speed: 0.01 cells/s
    - No custom perception override (uses base behavior if invoked)
    - Holds a coarse status string for mission state handling
    """

    def __init__(self, map_api: MapAPI, robot_id: str, start_pos: Tuple[float, float]):
        super().__init__(map_api, robot_id, "rover", start_pos)
        self.speed = 0.01
        self.stuck_cell = set()

    def step_towards(self, heading):
        movement_information = super().step_towards(heading)
        if movement_information["is_stuck"]:
            self.stuck_cell.add((int(self.x), int(self.y)))
        return movement_information

    def get_stuck_events_number(self):
        return self.stuck_cell.__len__()