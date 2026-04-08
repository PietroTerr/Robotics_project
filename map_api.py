from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import csv
import math
import random
from typing import Dict, List, Tuple


class RobotType(str, Enum):
    DRONE = "drone"
    SCOUT = "scout"
    ROVER = "rover"


@dataclass(frozen=True)
class StepResult:
    actual_velocity: float
    is_stuck: bool


@dataclass(frozen=True)
class TerrainObservation:
    x: int
    y: int
    features: Dict[str, float]


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
    feature_noise_std: float = 0.0
    feature_weight: float = 1.0


# ---------------------------------------------------------------------------
# Internal types — not part of the public API
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _MapCell:
    traversability: float
    slope: float
    uphill_angle: float
    stuck_event: bool = False
    appearance_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class _RobotState:
    robot_type: RobotType
    stuck_steps_remaining: int = 0


# ---------------------------------------------------------------------------

class MapAPI:
    """Map API for the hackathon.

    Allowed methods: ``register_robot``, ``step``, ``perceive``.
    Load a map from an organizer-provided CSV file and pass the instance to
    your governor.

    Example::

        api = MapAPI("maps/run_01.csv")
        api.register_robot("rover", RobotType.ROVER)
        result = api.step("rover", (2, 2), 1.0, 0.0)
    """

    def __init__(
        self,
        csv_path: str,
        config: MapConfig | None = None,
        rng_seed: int | None = None,
    ) -> None:
        self._params = config or MapConfig()
        self._rng_seed = rng_seed
        self._rng = random.Random(rng_seed)
        self._bots: Dict[str, _RobotState] = {}
        self._cells, self._nrows, self._ncols = _load_csv(csv_path)

    def register_robot(self, robot_id: str, robot_type: RobotType) -> None:
        """Register a robot identity and its fixed type."""
        if not robot_id.strip():
            raise ValueError("robot_id must be non-empty")
        if robot_id in self._bots:
            raise ValueError(f"robot_id '{robot_id}' is already registered")
        self._bots[robot_id] = _RobotState(robot_type=robot_type)

    def step(
        self,
        robot_id: str,
        position: Tuple[int, int],
        command_velocity: float,
        command_orientation: float,
    ) -> StepResult:
        """Execute one mobility step and return actual velocity and stuck status.

        ``command_orientation`` is the commanded heading angle in radians,
        measured from the global +X axis.

        For DRONE, terrain is ignored and the robot never gets stuck.
        For SCOUT, velocity is modulated by terrain and stuck events are reported,
        but the robot is never immobilized.
        For ROVER, velocity is modulated by terrain and stuck events can occur.
        """
        state = self._get_robot_state(robot_id)

        if self._is_currently_stuck(state):
            self._advance_stuck_timer(state)
            return StepResult(actual_velocity=0.0, is_stuck=True)

        robot_type = state.robot_type
        if robot_type == RobotType.DRONE:
            return StepResult(actual_velocity=max(0.0, command_velocity), is_stuck=False)

        cell = self._cell_at(position)
        slope_factor = self._directional_slope_factor(cell.slope, cell.uphill_angle, command_orientation)
        raw_eff = cell.traversability * slope_factor
        if self._params.clamp_effective_traversability:
            eff = _clamp(raw_eff, 0.0, 1.0)
        else:
            eff = max(0.0, raw_eff)
        actual_velocity = max(0.0, command_velocity) * eff

        if robot_type == RobotType.SCOUT:
            scout_stuck = cell.stuck_event
            return StepResult(actual_velocity=actual_velocity, is_stuck=scout_stuck)

        did_get_stuck = cell.stuck_event

        if did_get_stuck:
            if not self._params.disable_immobilization:
                if self._params.permanent_stuck:
                    state.stuck_steps_remaining = -1
                else:
                    state.stuck_steps_remaining = max(1, self._params.stuck_duration_steps)
                return StepResult(actual_velocity=0.0, is_stuck=True)
            return StepResult(actual_velocity=actual_velocity, is_stuck=True)

        return StepResult(actual_velocity=actual_velocity, is_stuck=False)

    def perceive(
        self,
        robot_id: str,
        position: Tuple[int, int],
    ) -> List[TerrainObservation]:
        """Return visible terrain features in a radius around the robot position.

        Output includes appearance features (texture, color), slope, and
        uphill angle.  Hidden fields such as traversability and stuck
        probability are excluded.
        """
        self._get_robot_state(robot_id)
        cx, cy = int(position[0]), int(position[1])
        radius = self._params.perceive_radius

        if radius < 0:
            raise ValueError("perceive_radius must be non-negative")

        observations: List[TerrainObservation] = []
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if not self._in_bounds(x, y):
                    continue
                if (x - cx) ** 2 + (y - cy) ** 2 > radius * radius:
                    continue

                cell = self._cells[y][x]
                features = dict(cell.appearance_features)
                features["slope"] = cell.slope
                features["uphill_angle"] = cell.uphill_angle
                features = self._apply_observation_transform(features, x, y)
                observations.append(TerrainObservation(x=x, y=y, features=features))

        observations.sort(key=lambda obs: (obs.y, obs.x))
        return observations

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_robot_state(self, robot_id: str) -> _RobotState:
        if robot_id not in self._bots:
            raise KeyError(f"robot_id '{robot_id}' is not registered")
        return self._bots[robot_id]

    def _is_currently_stuck(self, state: _RobotState) -> bool:
        return state.stuck_steps_remaining != 0

    def _advance_stuck_timer(self, state: _RobotState) -> None:
        if state.stuck_steps_remaining > 0:
            state.stuck_steps_remaining -= 1

    def _cell_at(self, position: Tuple[int, int]) -> _MapCell:
        x, y = int(position[0]), int(position[1])
        if not self._in_bounds(x, y):
            raise ValueError("position is outside terrain bounds")
        return self._cells[y][x]

    def _in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self._ncols and 0 <= y < self._nrows

    def _directional_slope_factor(
        self,
        slope: float,
        uphill_angle: float,
        movement_orientation: float,
    ) -> float:
        if self._params.slope_max_degrees_for_velocity <= 0.0:
            raise ValueError("slope_max_degrees_for_velocity must be > 0")

        ux, uy = math.cos(uphill_angle), math.sin(uphill_angle)
        mx, my = math.cos(movement_orientation), math.sin(movement_orientation)
        alignment = ux * mx + uy * my
        slope_norm = _clamp(slope / self._params.slope_max_degrees_for_velocity, 0.0, 1.0)
        signed_grade = slope_norm * alignment

        if signed_grade >= 0:
            return 1.0 - self._params.uphill_penalty * signed_grade

        downhill_alignment = -signed_grade
        return 1.0 + self._params.downhill_boost * downhill_alignment

    def _apply_observation_transform(self, features: Dict[str, float], x: int, y: int) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for key, value in features.items():
            transformed = value * self._params.feature_weight
            if self._params.feature_noise_std > 0:
                transformed += self._deterministic_noise(x, y, key)
            out[key] = transformed
        return out

    def _deterministic_noise(self, x: int, y: int, key: str) -> float:
        if self._rng_seed is None:
            return self._rng.gauss(0.0, self._params.feature_noise_std)

        key_value = sum(ord(c) for c in key)
        seed = self._rng_seed + 73856093 * x + 19349663 * y + 83492791 * key_value
        local_rng = random.Random(seed)
        return local_rng.gauss(0.0, self._params.feature_noise_std)


# ---------------------------------------------------------------------------
# Module-level private utilities
# ---------------------------------------------------------------------------

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _load_csv(
    csv_path: str,
) -> Tuple[List[List[_MapCell]], int, int]:
    """Load a map CSV produced by the organizer into a 2-D cell grid."""
    rows_dict: Dict[Tuple[int, int], _MapCell] = {}
    max_x = 0
    max_y = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x = int(row["x"])
            y = int(row["y"])
            cell = _MapCell(
                traversability=float(row["traversability"]),
                slope=float(row["slope"]),
                uphill_angle=float(row["uphill_angle"]),
                stuck_event=row["stuck_event"].strip().lower() == "true",
                appearance_features={
                    "texture": float(row["texture"]),
                    "color": float(row["color"]),
                },
            )
            rows_dict[(x, y)] = cell
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

    ncols = max_x + 1
    nrows = max_y + 1
    grid: List[List[_MapCell]] = [[None] * ncols for _ in range(nrows)]  # type: ignore[list-item]
    for (x, y), cell in rows_dict.items():
        grid[y][x] = cell

    return grid, nrows, ncols
