from __future__ import annotations

import uuid
from os import PathLike
from pathlib import Path
from typing import List

from .map_api_core import HiddenTerrainCell, MapAPICore, MapConfig, StepResult, TerrainObservation


class _StudentBackend(MapAPICore):
    """Internal backend for student facade.

    Uses randomized attribute names to make trivial hardcoding harder.
    """

    def __init__(
        self,
        terrain: List[List[HiddenTerrainCell]] | str | Path | PathLike[str],
        config: MapConfig | None = None,
        rng_seed: int | None = None,
        time_step: float = 0.2,
    ) -> None:
        map_key = f"_m_{uuid.uuid4().hex}"
        step_key = f"_s_{uuid.uuid4().hex}"
        log_key = f"_l_{uuid.uuid4().hex}"
        method_key = f"_c_{uuid.uuid4().hex}"
        step_log_key = f"_sl_{uuid.uuid4().hex}"
        perceive_log_key = f"_pl_{uuid.uuid4().hex}"
        object.__setattr__(self, "_map_storage_key", map_key)
        object.__setattr__(self, "_step_storage_key", step_key)
        object.__setattr__(self, "_log_storage_key", log_key)
        object.__setattr__(self, "_method_storage_key", method_key)
        object.__setattr__(self, "_step_log_storage_key", step_log_key)
        object.__setattr__(self, "_perceive_log_storage_key", perceive_log_key)
        super().__init__(
            terrain=terrain,
            config=config,
            rng_seed=rng_seed,
            time_step=time_step,
            _facade_token=MapAPICore._FACADE_INIT_TOKEN,
        )

    def _set_data(self, terrain: List[List[HiddenTerrainCell]]) -> None:
        object.__setattr__(self, self._map_storage_key, terrain)

    def _get_data(self) -> List[List[HiddenTerrainCell]]:
        return object.__getattribute__(self, self._map_storage_key)

    def _set_stuck_log_data(self, events):
        object.__setattr__(self, self._log_storage_key, events)

    def _get_stuck_log_data(self):
        return object.__getattribute__(self, self._log_storage_key)

    def _set_method_counters(self, counters):
        object.__setattr__(self, self._method_storage_key, counters)

    def _get_method_counters(self):
        return object.__getattribute__(self, self._method_storage_key)

    def _set_step_call_log_data(self, events):
        object.__setattr__(self, self._step_log_storage_key, events)

    def _get_step_call_log_data(self):
        return object.__getattribute__(self, self._step_log_storage_key)

    def _set_perceive_call_log_data(self, events):
        object.__setattr__(self, self._perceive_log_storage_key, events)

    def _get_perceive_call_log_data(self):
        return object.__getattribute__(self, self._perceive_log_storage_key)

    def register_robot(self, robot_id: str, robot_type: str) -> None:
        return super().register_robot(robot_id, robot_type)

    def step(
        self,
        robot_id: str,
        position,
        command_velocity: float,
        command_orientation: float,
    ) -> StepResult:
        return super().step(robot_id, position, command_velocity, command_orientation)

    def perceive(self, robot_id: str, position) -> List[TerrainObservation]:
        return super().perceive(robot_id, position)


class MapAPI:
    """Student-facing facade exposing only register_robot, step, and perceive."""

    _PUBLIC_API = {"register_robot", "step", "perceive"}

    def __init__(
        self,
        terrain: List[List[HiddenTerrainCell]] | str | Path | PathLike[str],
        config: MapConfig | None = None,
        rng_seed: int | None = None,
        time_step: float = 0.2,
    ) -> None:
        backend_name = f"_{uuid.uuid4().hex}"
        object.__setattr__(self, "_backend_name", backend_name)
        backend = _StudentBackend(terrain=terrain, config=config, rng_seed=rng_seed, time_step=time_step)
        object.__setattr__(self, backend_name, backend)

    def _backend(self) -> _StudentBackend:
        backend_name = object.__getattribute__(self, "_backend_name")
        return object.__getattribute__(self, backend_name)

    def register_robot(self, robot_id: str, robot_type: str) -> None:
        self._backend().register_robot(robot_id, robot_type)

    def step(self, robot_id: str, position, command_velocity: float, command_orientation: float) -> StepResult:
        return self._backend().step(robot_id, position, command_velocity, command_orientation)

    def perceive(self, robot_id: str, position) -> List[TerrainObservation]:
        return self._backend().perceive(robot_id, position)

    def __getattribute__(self, name: str):
        if name in {
            "_PUBLIC_API",
            "_backend_name",
            "_backend",
            "__class__",
            "__dict__",
            "__repr__",
            "__str__",
            "__dir__",
            "__getattribute__",
            "__setattr__",
            "__delattr__",
            "__init__",
        }:
            return object.__getattribute__(self, name)
        if name in object.__getattribute__(self, "_PUBLIC_API"):
            return object.__getattribute__(self, name)
        raise AttributeError(f"'{name}' is not accessible")

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("Cannot set attributes on student MapAPI facade")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Cannot delete attributes on student MapAPI facade")

    def __dir__(self):
        return sorted(self._PUBLIC_API)

