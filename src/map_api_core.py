from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import Enum
import math
from os import PathLike
from pathlib import Path
import random
from typing import Dict, List, Tuple


class RobotType(str, Enum):
    DRONE = "drone"
    SCOUT = "scout"
    ROVER = "rover"


@dataclass(frozen=True)
class StepResult:
    actual_velocity: float
    is_stuck: bool | None
    battery_value: float


@dataclass(frozen=True)
class TerrainObservation:
    x: int
    y: int
    features: Dict[str, float]


@dataclass(frozen=True)
class StuckEventLog:
    step_index: int
    robot_id: str
    robot_type: str
    location: Tuple[float, float]
    cell_indices: Tuple[int, int]


@dataclass(frozen=True)
class StepCallLog:
    robot_id: str
    robot_type: str
    position: Tuple[float, float]
    command_velocity: float
    command_orientation: float


@dataclass(frozen=True)
class PerceiveCallLog:
    robot_id: str
    robot_type: str
    position: Tuple[float, float]


@dataclass(frozen=True)
class MapConfig:
    # motion
    permanent_stuck: bool = False
    stuck_duration_steps: int = 0
    disable_immobilization: bool = True
    slope_max_degrees_for_velocity: float = 30.0
    clamp_effective_traversability: bool = False
    uphill_penalty: float = 1.0
    downhill_boost: float = 0.2
    # perception
    perceive_radius: int = 2
    # when True (default) use pre-computed stuck_event flag from the cell;
    # set to False to restore legacy per-step Bernoulli sampling from stuck_probability.
    use_stuck_event_map: bool = True


@dataclass(frozen=True)
class HiddenTerrainCell:
    traversability: float
    stuck_probability: float
    slope: float
    uphill_angle: float
    stuck_event: bool = False
    appearance_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class RobotRuntimeState:
    robot_type: str
    stuck_steps_remaining: int = 0
    battery_value: float = 1.0
    power_draw: float = 0.1
    battery_recharge: float = 0.05
    min_battery_value: float = 0.0
    max_battery_value: float = 1.0
    max_velocity: float = 2.0
    affected_by_terrain: bool = True
    affected_by_stuck_events: bool = True
    immobilized_when_stuck: bool = True


@dataclass(frozen=True)
class RobotProfile:
    initial_battery_value: float
    power_draw: float
    battery_recharge: float
    min_battery_value: float
    max_battery_value: float
    max_velocity: float
    affected_by_terrain: bool
    affected_by_stuck_events: bool
    immobilized_when_stuck: bool


ROBOT_PROFILES: Dict[str, RobotProfile] = {
    "drone": RobotProfile(
        initial_battery_value=1.0,
        power_draw=0.02,
        battery_recharge=0.002,
        min_battery_value=0.0,
        max_battery_value=1.0,
        max_velocity=1.0,
        affected_by_terrain=False,
        affected_by_stuck_events=False,
        immobilized_when_stuck=False,
    ),
    "scout": RobotProfile(
        initial_battery_value=1.0,
        power_draw=0.00,
        battery_recharge=0.00,
        min_battery_value=0.0,
        max_battery_value=1.0,
        max_velocity=0.05,
        affected_by_terrain=True,
        affected_by_stuck_events=True,
        immobilized_when_stuck=False,
    ),
    "rover": RobotProfile(
        initial_battery_value=1.0,
        power_draw=0.00,
        battery_recharge=0.00,
        min_battery_value=0.0,
        max_battery_value=1.0,
        max_velocity=0.01,
        affected_by_terrain=True,
        affected_by_stuck_events=True,
        immobilized_when_stuck=True,
    ),
}


class MapAPICore:
    """Shared hidden-terrain runtime used by the student and scoring facades.

    The policy-facing surface is intentionally limited to two query methods:
    - step: converts command velocity to actual velocity and may trigger stuck state.
    - perceive: returns local appearance observations without hidden traversability data.
    """

    _FACADE_INIT_TOKEN = object()

    def __init__(
        self,
        terrain: List[List[HiddenTerrainCell]] | str | Path | PathLike[str],
        config: MapConfig | None = None,
        rng_seed: int | None = None,
        time_step: float = 0.2,
        _facade_token: object | None = None,
    ) -> None:
        """Create the map API from either terrain cells or a CSV path."""
        if type(self) is MapAPICore:
            raise TypeError("Do not instantiate MapAPICore directly")
        if _facade_token is not self._FACADE_INIT_TOKEN:
            raise TypeError("MapAPICore can only be initialized by approved facades")

        if isinstance(terrain, (str, Path, PathLike)):
            terrain_data = load_terrain_from_csv(terrain)
        else:
            terrain_data = terrain

        if not terrain_data or not terrain_data[0]:
            raise ValueError("terrain must be a non-empty 2D grid")
        if time_step <= 0.0 or time_step > 0.99:
            raise ValueError("time_step must be in (0, 0.4]")

        width = len(terrain_data[0])
        if any(len(row) != width for row in terrain_data):
            raise ValueError("terrain rows must all have the same length")

        self._height = len(terrain_data)
        self._width = width
        self._set_data(terrain_data)
        self._set_stuck_log_data([])
        self._set_method_counters({"register": 0, "step": 0, "perceive": 0})
        self._set_step_call_log_data([])
        self._set_perceive_call_log_data([])
        self._config = config or MapConfig()
        self._robots: Dict[str, RobotRuntimeState] = {}
        self._robot_step_index: Dict[str, int] = {}
        self._rng = random.Random(rng_seed)
        self._time_step = time_step

    def _set_data(self, terrain: List[List[HiddenTerrainCell]]) -> None:
        self._terrain = terrain

    def _get_data(self) -> List[List[HiddenTerrainCell]]:
        return self._terrain

    def _set_stuck_log_data(self, events: List[StuckEventLog]) -> None:
        self._stuck_log = events

    def _get_stuck_log_data(self) -> List[StuckEventLog]:
        return self._stuck_log

    def _append_stuck_event(self, event: StuckEventLog) -> None:
        self._get_stuck_log_data().append(event)

    def _set_method_counters(self, counters: Dict[str, int]) -> None:
        self._method_counters = counters

    def _get_method_counters(self) -> Dict[str, int]:
        return self._method_counters

    def _increment_method_count(self, method: str) -> None:
        counters = self._get_method_counters()
        counters[method] = counters.get(method, 0) + 1

    def _set_step_call_log_data(self, events: List[StepCallLog]) -> None:
        self._step_call_log = events

    def _get_step_call_log_data(self) -> List[StepCallLog]:
        return self._step_call_log

    def _append_step_call(self, event: StepCallLog) -> None:
        self._get_step_call_log_data().append(event)

    def _set_perceive_call_log_data(self, events: List[PerceiveCallLog]) -> None:
        self._perceive_call_log = events

    def _get_perceive_call_log_data(self) -> List[PerceiveCallLog]:
        return self._perceive_call_log

    def _append_perceive_call(self, event: PerceiveCallLog) -> None:
        self._get_perceive_call_log_data().append(event)

    def get_data(self) -> List[List[HiddenTerrainCell]]:
        return self._get_data()

    def register_robot(self, robot_id: str, robot_type: str) -> None:
        """Register a robot identity and its fixed type."""
        if not robot_id.strip():
            raise ValueError("robot_id must be non-empty")
        if robot_id in self._robots:
            raise ValueError(f"robot_id '{robot_id}' is already registered")
        self._increment_method_count("register")
        profile = ROBOT_PROFILES[robot_type]
        self._robots[robot_id] = RobotRuntimeState(
            robot_type=robot_type,
            battery_value=profile.initial_battery_value,
            power_draw=profile.power_draw,
            battery_recharge=profile.battery_recharge,
            min_battery_value=profile.min_battery_value,
            max_battery_value=profile.max_battery_value,
            max_velocity=profile.max_velocity,
            affected_by_terrain=profile.affected_by_terrain,
            affected_by_stuck_events=profile.affected_by_stuck_events,
            immobilized_when_stuck=profile.immobilized_when_stuck,
        )
        self._robot_step_index[robot_id] = 0

    def get_stuck_events(self) -> List[StuckEventLog]:
        return list(self._get_stuck_log_data())

    def get_robot_step_indices(self) -> Dict[str, int]:
        return dict(self._robot_step_index)

    def get_method_counts(self) -> Dict[str, int]:
        return dict(self._get_method_counters())

    def get_step_call_logs(self) -> List[StepCallLog]:
        return list(self._get_step_call_log_data())

    def get_perceive_call_logs(self) -> List[PerceiveCallLog]:
        return list(self._get_perceive_call_log_data())

    def step(
        self,
        robot_id: str,
        position: Tuple[float, float],
        command_velocity: float,
        command_orientation: float,
    ) -> StepResult:
        """Execute one mobility query and return actual velocity and stuck status, with battery logic.

        ``command_orientation`` is the commanded heading angle in radians,
        measured from the global +X axis.

        For DRONE, terrain is ignored and the robot never gets stuck.
        For SCOUT, velocity is modulated by terrain and stuck events are reported,
        but the robot is never immobilized.
        For ROVER, velocity is modulated by terrain and stuck events can occur.
        """
        self._increment_method_count("step")
        state = self._get_robot_state(robot_id)
        self._append_step_call(
            StepCallLog(
                robot_id=robot_id,
                robot_type=state.robot_type,
                position=(float(position[0]), float(position[1])),
                command_velocity=float(command_velocity),
                command_orientation=float(command_orientation),
            )
        )
        step_index = self._robot_step_index[robot_id]
        self._robot_step_index[robot_id] = step_index + 1
        commanded_speed = max(0.0, command_velocity)
        default_is_stuck = None if state.robot_type == "drone" else False

        # Battery logic
        if commanded_speed > 0:
            state.battery_value -= state.power_draw * commanded_speed * self._time_step
        else:
            state.battery_value += state.battery_recharge * self._time_step
        state.battery_value = max(state.min_battery_value, min(state.max_battery_value, state.battery_value))

        # Prevent movement if battery too low
        if state.battery_value <= state.min_battery_value + 1e-8:
            return StepResult(actual_velocity=0.0, is_stuck=default_is_stuck, battery_value=state.battery_value)

        if state.immobilized_when_stuck and self._is_currently_stuck(state):
            self._advance_stuck_timer(state)
            return StepResult(actual_velocity=0.0, is_stuck=True, battery_value=state.battery_value)

        capped_command_speed = min(commanded_speed, state.max_velocity)

        if not state.affected_by_terrain:
            return StepResult(actual_velocity=capped_command_speed, is_stuck=None, battery_value=state.battery_value)

        cell_x, cell_y = int(position[0]), int(position[1])
        cell = self._cell_at(position)
        slope_factor = self._directional_slope_factor(cell.slope, cell.uphill_angle, command_orientation)
        raw_effective_traversability = cell.traversability * slope_factor
        if self._config.clamp_effective_traversability:
            effective_traversability = _clamp(raw_effective_traversability, 0.0, 1.0)
        else:
            effective_traversability = max(0.0, raw_effective_traversability)
        actual_velocity = capped_command_speed * effective_traversability

        if not state.affected_by_stuck_events:
            return StepResult(actual_velocity=actual_velocity, is_stuck=default_is_stuck, battery_value=state.battery_value)

        if self._config.use_stuck_event_map:
            did_get_stuck = cell.stuck_event
        else:
            did_get_stuck = self._rng.random() < cell.stuck_probability

        if did_get_stuck:
            self._append_stuck_event(
                StuckEventLog(
                    step_index=step_index,
                    robot_id=robot_id,
                    robot_type=state.robot_type,
                    location=(position[0], position[1]),
                    cell_indices=(cell_x, cell_y),
                )
            )
            if state.immobilized_when_stuck and not self._config.disable_immobilization:
                if self._config.permanent_stuck:
                    state.stuck_steps_remaining = -1
                else:
                    state.stuck_steps_remaining = max(1, self._config.stuck_duration_steps)
                return StepResult(actual_velocity=0.0, is_stuck=True, battery_value=state.battery_value)
            return StepResult(actual_velocity=actual_velocity, is_stuck=True, battery_value=state.battery_value)

        return StepResult(actual_velocity=actual_velocity, is_stuck=default_is_stuck, battery_value=state.battery_value)

    def perceive(
        self,
        robot_id: str,
        position: Tuple[float, float],
    ) -> List[TerrainObservation]:
        """Return visible terrain features in a radius around the robot position.

        Output includes appearance features, slope, and uphill angle.
        Hidden fields such as traversability and stuck probability are excluded.
        """
        self._increment_method_count("perceive")
        state = self._get_robot_state(robot_id)
        self._append_perceive_call(
            PerceiveCallLog(
                robot_id=robot_id,
                robot_type=state.robot_type,
                position=(float(position[0]), float(position[1])),
            )
        )
        cx, cy = int(position[0]), int(position[1])
        radius = self._config.perceive_radius

        if radius < 0:
            raise ValueError("perceive_radius must be non-negative")

        observations: List[TerrainObservation] = []
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if not self._in_bounds(x, y):
                    continue
                if (x - cx) ** 2 + (y - cy) ** 2 > radius * radius:
                    continue

                cell = self.get_data()[y][x]
                features = dict(cell.appearance_features)
                features["slope"] = cell.slope
                features["uphill_angle"] = cell.uphill_angle
                observations.append(TerrainObservation(x=x, y=y, features=features))

        observations.sort(key=lambda obs: (obs.y, obs.x))
        return observations

    def _get_robot_state(self, robot_id: str) -> RobotRuntimeState:
        if robot_id not in self._robots:
            raise KeyError(f"robot_id '{robot_id}' is not registered")
        return self._robots[robot_id]

    def _is_currently_stuck(self, state: RobotRuntimeState) -> bool:
        return state.stuck_steps_remaining != 0

    def _advance_stuck_timer(self, state: RobotRuntimeState) -> None:
        if state.stuck_steps_remaining > 0:
            state.stuck_steps_remaining -= 1

    def _cell_at(self, position: Tuple[float, float]) -> HiddenTerrainCell:
        x, y = int(position[0]), int(position[1])
        if not self._in_bounds(x, y):
            raise ValueError("position is outside terrain bounds")
        return self.get_data()[y][x]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._width and 0 <= y < self._height

    def _directional_slope_factor(
        self,
        slope: float,
        uphill_angle: float,
        movement_orientation: float,
    ) -> float:
        if self._config.slope_max_degrees_for_velocity <= 0.0:
            raise ValueError("slope_max_degrees_for_velocity must be > 0")

        ux, uy = math.cos(uphill_angle), math.sin(uphill_angle)
        mx, my = math.cos(movement_orientation), math.sin(movement_orientation)
        alignment = ux * mx + uy * my
        slope_norm = _clamp(slope / self._config.slope_max_degrees_for_velocity, 0.0, 1.0)
        signed_grade = slope_norm * alignment

        if signed_grade >= 0:
            return 1.0 - self._config.uphill_penalty * signed_grade

        downhill_alignment = -signed_grade
        return 1.0 + self._config.downhill_boost * downhill_alignment


def load_terrain_from_csv(csv_path: str | Path) -> List[List[HiddenTerrainCell]]:
    """Load terrain cells from an organizer CSV export."""
    path = Path(csv_path)
    rows: Dict[Tuple[int, int], HiddenTerrainCell] = {}
    max_x = 0
    max_y = 0

    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = int(row["x"])
            y = int(row["y"])
            rows[(x, y)] = HiddenTerrainCell(
                traversability=float(row["traversability"]),
                stuck_probability=0.0,
                slope=float(row["slope"]),
                uphill_angle=float(row["uphill_angle"]),
                stuck_event=row["stuck_event"].strip().lower() == "true",
                appearance_features={
                    "texture": float(row["texture"]),
                    "color": float(row["color"]),
                },
            )
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    width = max_x + 1
    height = max_y + 1
    terrain: List[List[HiddenTerrainCell]] = [[None] * width for _ in range(height)]  # type: ignore[list-item]
    for (x, y), cell in rows.items():
        terrain[y][x] = cell

    return terrain

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))