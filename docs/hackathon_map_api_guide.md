# Hackathon MapAPI Guide (Student-Facing)

This guide describes the student-facing API in `src/map_api.py`.

## What You Can Use
The simulation exposes a `MapAPI` object with three relevant methods:
- `register_robot(robot_id, robot_type)`
- `step(robot_id, position, command_velocity, command_orientation)`
- `perceive(robot_id, position)`

`command_orientation` is in radians, measured from global +X.

> **Position semantics:** `position` in both `step` and `perceive` is always the robot's continuous world coordinate `(x, y)` as floats — the same coordinate your controller tracks. The API handles all internal terrain cell lookups. You never need to think in terms of grid indices.

## Constructor (Student Version)
Student code receives a CSV map file prepared by organizers and passes the path directly into `MapAPI`.

- `MapAPI(terrain, config=None, rng_seed=None, time_step=0.2)`
- `terrain` can be either:
  - a `str` / `Path` pointing to an organizer-supplied CSV file — the terrain is loaded automatically, **or**
  - a rectangular grid of `HiddenTerrainCell` objects.
- `time_step` is configured once at construction and must satisfy `(0, 0.4]`.

Example (recommended — CSV path):
```python
from pathlib import Path
from map_api import MapAPI

csv_path = Path("teams_workspace") / "map_001_seed42.csv"
api = MapAPI(csv_path, rng_seed=42, time_step=0.2)
api.register_robot("rover_1", "rover")
```

Example (manual terrain construction):
```python
from map_api_core import HiddenTerrainCell
from map_api import MapAPI

terrain = [[
    HiddenTerrainCell(
        traversability=0.8,
        stuck_probability=0.1,
        slope=5.0,
        uphill_angle=0.0,
        stuck_event=False,
        appearance_features={"texture": 0.3, "color": 0.6},
    )
]]

api = MapAPI(terrain, rng_seed=42, time_step=0.2)
api.register_robot("rover_1", "rover")
```

## Robot Types
Pass the robot type as a plain string to `register_robot`:
- `"drone"`: not slowed by terrain, never immobilized by stuck behavior. `is_stuck` is always `None`. Current profile: `max_velocity=1.0`, `power_draw=0.02`, `battery_recharge=0.002`, `battery_value in [0.0, 1.0]`.
- `"scout"`: terrain affects velocity; it can report stuck through `is_stuck`, but it is not immobilized. Current profile: `max_velocity=0.05`, no battery draw or recharge.
- `"rover"`: terrain affects velocity and it can be immobilized by stuck behavior. Current profile: `max_velocity=0.01`, no battery draw or recharge.

Useful current drone timing numbers:
- flight time from full to empty at max speed: `50 s`
- recharge time from empty to full: `500 s`

## Minimal Usage Pattern
Typical control loop per timestep:
1. Call `perceive(...)` at current position.
2. Compute your action from returned observations.
3. Call `step(...)` with commanded velocity and heading.
4. Use `StepResult.actual_velocity` and `StepResult.is_stuck` to update your planner/controller.

## API Inputs/Outputs
### `register_robot(robot_id, robot_type)`
Registers one robot in the simulation.
- `robot_type`: plain string — `"drone"`, `"scout"`, or `"rover"`.

### `step(robot_id, position, command_velocity, command_orientation) -> StepResult`
Inputs:
- `robot_id`: string used at registration.
- `position`: `(x, y)` tuple of **floats** — the robot's current continuous world coordinates. The API maps this to the appropriate terrain cell internally. You do not need to know anything about the map layout or cell indices.
- `command_velocity`: desired speed.
- `command_orientation`: heading in radians from global +X.

Output (`StepResult`):
- `actual_velocity`: executed speed after terrain and robot effects.
- `is_stuck`: stuck signal for the timestep. `None` for drones (stuck state is not tracked).
- `battery_value`: current battery value after the timestep's battery update.

### `perceive(robot_id, position) -> list[TerrainObservation]`
The perception radius is configured in `MapConfig.perceive_radius`.

Inputs:
- `robot_id`: string used at registration.
- `position`: `(x, y)` tuple of **floats** — the robot's current continuous world coordinates. The API determines which cells fall within the perception radius internally.

Returns local observations around `position`.
Each observation contains:
- `x`, `y`
- `features` with only observable values:
  - `texture`
  - `color`
  - `slope`
  - `uphill_angle`

## Important Constraints
- Hidden fields like traversability and stuck probability are not returned by `perceive`.
- Use only `perceive` and `step` outputs in your policy logic.
- Scoring uses a different facade with different internal storage names.
- Competition runs currently use `maximumTime = 1000000` seconds.

## Telemetry & Logging
The runtime maintains append-only logs for analysis and validation.

Important:
- Student policies should rely only on `register_robot`, `step`, and `perceive`.
- Detailed telemetry accessors are intended for scoring/organizer workflows.

### Stuck Event Log
Access via `api.get_stuck_events()`:
- Returns list of `StuckEventLog` entries
- Each entry contains: `step_index`, `robot_id`, `robot_type`, `location (x, y)`, `cell_indices`
- Logged when terrain stuck event triggers, not during stuck-timer continuation
- Drones never generate stuck events

### Per-Robot Step Index
Access via `api.get_robot_step_indices()`:
- Returns dict mapping `robot_id` to step count
- Each robot has independent counter starting at 0
- Incremented at start of each `step()` call
- Useful for correlating events with mission timeline

### Anti-Tampering Note
Do not attempt to modify logs or counters directly. The `MapAPI` instance uses internal obfuscation to prevent tampering. Only read through the public accessor methods.

## Working Example in This Repository
See `src/student_workflow_example.py`.

Run from project root:
```bash
python src/student_workflow_example.py
```

It loads `src/map_001_seed42.csv`, converts it to terrain cells, registers a rover, performs one `step`, and prints `perceive` output.

To view stuck event logs and telemetry after a run (scoring/organizer runtime):
```python
events = api.get_stuck_events()
print(f"Total stuck events: {len(events)}")
for event in events:
    print(f"  Robot {event.robot_id} stuck at step {event.step_index} at {event.location}")

step_indices = api.get_robot_step_indices()
print(f"Robot step indices: {step_indices}")

method_counts = api.get_method_counts()
print(f"Method calls: {method_counts}")

step_call_logs = api.get_step_call_logs()
print(f"Step call logs: {len(step_call_logs)}")

perceive_call_logs = api.get_perceive_call_logs()
print(f"Perceive call logs: {len(perceive_call_logs)}")
```

## Note About Organizer Utilities
Organizer-only generation lives in `src/generate_maps.py` via `OrganizerMapGenerator` and `GenerationConfig`.
Those utilities are intentionally separate from the student-facing `MapAPI` module.
