# Hackathon Rules

In this competition, each team must design an exploration strategy to guide robots across an unknown terrain map and reach the target zone as efficiently as possible. The strategy shall be implemented by a _governor_ function. The map contains areas with different mobility properties, and teams must rely only on the allowed API calls to perceive local information and execute motion commands. Final ranking is determined by mission success and execution time across multiple runs, rewarding both performance and robustness.

## Instructions
- The governor should be packaged as a function callable as: `runGovernor(mapApi_instance, startPoint, targetPoint, maximumTime)`.
- For the current competition setup, use `maximumTime = 1000000` seconds.
- The governor should return the following tuple: `(reached_target: boolean, last_distance_from_target: double, time_elapsed: double, drone_travel: double, scout_travel: double, rover_travel: double, perceive_calls: int, step_calls: int, stuck_events: int)`.
- Metrics can be queried from the `MapAPI` instance:
  - `get_method_counts()` returns count of `register`, `step`, `perceive` calls
  - `get_stuck_events()` returns list of stuck event logs with step index and position
  - `get_robot_step_indices()` returns per-robot step counters
	- `get_step_call_logs()` returns step input logs (`robot_id`, `robot_type`, `position`, `command_velocity`, `command_orientation`)
	- `get_perceive_call_logs()` returns perceive input logs (`robot_id`, `robot_type`, `position`)

## Rules
- Teams may use only the public `MapAPI` interface (`register_robot`, `step`, `perceive`).
- Direct access to hidden terrain internals is not allowed.
- Each team is allowed to choose a suitable timestep for the simulation, but this has to make sure no agent can skip map cells.
- Any exploit that bypasses intended API boundaries will lead to disqualification.

## Agents Setup
- **Drone**:
	unaffected by terrain traversability, unstuckable, moves at constant speed of **1.0 cells/s**, omnidirectionally.
	Current runtime profile: `initial_battery_value=1.0`, `power_draw=0.02`, `battery_recharge=0.002`, `min_battery_value=0.0`, `max_battery_value=1.0`, `max_velocity=1.0`.
	At max speed this corresponds to **50 s** flight time from full to empty and **500 s** recharge time from empty to full.
- **Scout**:
	affected by traversability, can *sense* stuck events, does not get stuck, moves at **0.05 cells/s**.
	Current runtime profile: `initial_battery_value=1.0`, `power_draw=0.0`, `battery_recharge=0.0`, `min_battery_value=0.0`, `max_battery_value=1.0`, `max_velocity=0.05`.
- **Rover**:
	affected by traversability, can get stuck (and receives a score penalty each time), moves at **0.01 cells/s**.
	Current runtime profile: `initial_battery_value=1.0`, `power_draw=0.0`, `battery_recharge=0.0`, `min_battery_value=0.0`, `max_battery_value=1.0`, `max_velocity=0.01`.

## Competition
- Teams have to provide the organizers with their codebase, formatted as requested in the instructions, detailed above.
- The organizers will provide the `MapAPI` class to avoid class tampering.
- A series ($n\geq3$, depending on the duration of the runs...) of datasets, consisting of a map, starting point and target point, will be prepared for the teams. Each team governor will run the same $n$ datasets.
- Scores are based on a cost function that penalizes mission time, stuck events, and final distance to target.
- The competition will consist of multiple runs to evaluate robustness.

## Scoring


Each run $i$ is assigned a cost:

$$J_i = T_i + \lambda_s \cdot N_i^{\text{stuck}} + \lambda_d \cdot D_i^{\text{final}}$$

where:
- $T_i$ is the total mission time for run $i$
- $N_i^{\text{stuck}}$ is the number of stuck events during run $i$
- $D_i^{\text{final}}$ is the cell distance between the robot's final cell and the target cell (CELL DISTANCE)
- $\lambda_s = 50,000$ (penalty per stuck event)
- $\lambda_d = 7,000$ (penalty per cell of distance to target)

**Success criterion:** The mission is considered successful if the rover is in the same cell as the target at the end of the run (i.e., $D_i^{\text{final}} = 0$). If the rover is not in the target cell, a distance penalty is applied proportional to the cell distance.

The final score is the average cost over all $N$ runs ($N \geq 3$, ideally $N = 30$):

$$J = \frac{1}{N} \sum_{i=1}^{N} J_i$$

The team with the **lowest** $J$ wins.

## Warnings
- Several automated checks will be made to ensure the `map_api` is not tampered with, that all rules are followed, and that the outputs of the governor are legit.
- The `map_api.py` file at your disposal differs from the one used during scoring in many aspects, i.e. the names of properties, in order to make exploits harder. These changes will never affect the performance of strategies that use only the provided methods.
- Organizer-generated traversability is floored at the configured `trav_low_baseline`, so zero traversability cells are not intended in the current setup.