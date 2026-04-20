"""
Autonomous Drone Exploration System

This module handles the exploration logic for airborne robots (drones) across an unknown terrain. 
Key features:
- Intelligent heuristic pathfinding that balances moving towards a target with exploring unvisited areas.
- Shared persistent `TerrainMemory` that can be maintained across multiple short-lived drone flights.
- Visual terrain analysis: Uses observed "texture" values to estimate ground traversability.
- Avoids low-traversability areas, creating a safer mapping strategy.
- Includes plotting functions tailored for drone flight results (trajectories, memory state, observation density).
"""
from pathlib import Path
import math
import random

import matplotlib.pyplot as plt
import numpy as np

from src.map_api import MapAPI


# ---------------------------------------------------------------------------
# Map loading
# ---------------------------------------------------------------------------

def load_map(csv_path: str, rng_seed: int) -> MapAPI:
    """Load a map from a CSV file located in the src/ folder."""
    workspace_dir = Path(__file__).resolve().parent
    full_path = workspace_dir / "src" / csv_path
    return MapAPI(terrain=str(full_path), rng_seed=rng_seed)


# ---------------------------------------------------------------------------
# Single step helper
# ---------------------------------------------------------------------------

def simulate_and_perceive(
    map_api: MapAPI,
    robot_id: str,
    position: tuple,
    speed: float,
    orientation: float,
):
    """Perform one step + perceive and print the results."""
    step_result = map_api.step(
        robot_id=robot_id,
        position=position,
        command_velocity=speed,
        command_orientation=orientation,
    )
    print("Step result:", step_result)

    observations = map_api.perceive(robot_id=robot_id, position=position)
    print(f"Perceive returned {len(observations)} observations at {position}:")
    for obs in observations:
        print(obs)
    return step_result, observations


# ---------------------------------------------------------------------------
# Drone – single intelligent exploration flight
# ---------------------------------------------------------------------------

# Number of candidate movement directions evaluated at each step
_N_DIRECTIONS = 16


def _estimate_traversability(texture: float) -> float:
    """Map observed *texture* value to a [0, 1] traversability estimate.

    The heuristic assumes that low texture correlates with difficult terrain
    and high texture correlates with easy terrain.  The mapping is kept
    simple so that it works as a first-order signal.
    """
    return max(0.0, min(1.0, texture))


class TerrainMemory:
    """Persistent memory of terrain observations gathered during flights.

    Stores the average observed features for each discrete (x, y) cell,
    together with visit counts.  This lets the drone build an increasingly
    accurate picture of the map across multiple steps and flights.
    """

    def __init__(self):
        # (x, y) -> {"texture_sum": float, "slope_sum": float, "count": int}
        self._data: dict[tuple[int, int], dict] = {}

    def update(self, observations):
        """Incorporate a list of TerrainObservation into memory."""
        for obs in observations:
            key = (int(obs.x), int(obs.y))
            if key not in self._data:
                self._data[key] = {"texture_sum": 0.0, "slope_sum": 0.0, "count": 0, "stuck": False}
            entry = self._data[key]
            entry["texture_sum"] += obs.features.get("texture", 0.5)
            entry["slope_sum"] += obs.features.get("slope", 0.0)
            entry["count"] += 1

    def update_stuck(self, x: int, y: int):
        key = (x, y)
        if key not in self._data:
            self._data[key] = {"texture_sum": 0.0, "slope_sum": 0.0, "count": 1, "stuck": True}
        else:
            self._data[key]["stuck"] = True

    def get_estimated_traversability(self, x: int, y: int) -> float | None:
        """Return estimated traversability for cell *(x, y)*, or None if unknown."""
        key = (x, y)
        if key not in self._data:
            return None
        entry = self._data[key]
        if entry.get("stuck", False):
            return -10.0
        if entry["count"] == 0:
            return None
        avg_texture = entry["texture_sum"] / entry["count"]
        return _estimate_traversability(avg_texture)

    def is_known(self, x: int, y: int) -> bool:
        return (x, y) in self._data

    def visit_count(self, x: int, y: int) -> int:
        entry = self._data.get((x, y))
        return entry["count"] if entry else 0


def smart_drone_exploration(
    map_api: MapAPI,
    robot_id: str = "drone_1",
    start: tuple = (3, 3),
    target: tuple | None = None,
    speed: float = 2.0,
    dt: float = 0.2,
    map_width: int = 50,
    map_height: int = 50,
    battery_safety_margin: float = 0.05,
    rng_seed: int = 42,
    verbose: bool = True,
    terrain_memory: TerrainMemory | None = None,
    traversability_threshold: float = 0.3,
):
    """Run one drone flight with intelligent, observation-based movement.

    The drone heads toward *target* but deviates when it observes cells
    with low estimated traversability along the direct path.  All terrain
    observations are stored in a ``TerrainMemory`` so that knowledge
    persists across steps (and across flights if the same memory object
    is reused).

    The drone is registered with ``robot_type="drone"`` using the public API.

    Parameters
    ----------
    target : tuple | None
        (x, y) cell to look for in perceive results.  When the target cell
        appears in any observation the flight is marked as successful.
    terrain_memory : TerrainMemory | None
        Shared memory of terrain observations.  If *None* a fresh one is
        created.  Pass the same instance across flights to accumulate
        knowledge.
    traversability_threshold : float
        Cells with estimated traversability below this value are treated as
        obstacles that the drone will try to route around.

    Returns
    -------
    path : list[tuple]
    all_observations : list[list[TerrainObservation]]
    battery_log : list[float]
    target_found : bool
        True if the *target* cell was observed during this flight.
    terrain_memory : TerrainMemory
        The (possibly updated) terrain memory.
    """
    rng = random.Random(rng_seed)

    if terrain_memory is None:
        terrain_memory = TerrainMemory()

    # Register drone via public API (string type, no map_api_core import)
    map_api.register_robot(robot_id=robot_id, robot_type="drone")

    tx, ty = (int(target[0]), int(target[1])) if target else (None, None)

    x, y = float(start[0]), float(start[1])
    path = [(x, y)]
    battery_log = []
    all_observations = []
    step_count = 0
    target_found = False

    if verbose:
        print(f"  [{robot_id}] Taking off from ({x:.1f}, {y:.1f}), speed={speed}")

    # --- Pre-perceive at start position ---
    initial_obs = map_api.perceive(robot_id=robot_id, position=(x, y))
    all_observations.append(initial_obs)
    terrain_memory.update(initial_obs)

    is_recharging = False

    while True:
        # Prevent infinite flights
        if step_count > 1500:
            if verbose:
                print(f"  [{robot_id}] Flight limit reached (5 mins / 1500 steps) – landing.")
            break

        # --- Choose best direction ------------------------------------------
        best_orientation = 0.0
        best_score = -math.inf

        if target is not None:
            dist_to_target = math.hypot(tx - x, ty - y)
            direct_angle = math.atan2(ty - y, tx - x)
        else:
            dist_to_target = None
            direct_angle = None

        for i in range(_N_DIRECTIONS):
            angle = -math.pi + 2 * math.pi * i / _N_DIRECTIONS

            nx = x + speed * math.cos(angle) * dt
            ny = y + speed * math.sin(angle) * dt
            nx = max(0.0, min(float(map_width - 1), nx))
            ny = max(0.0, min(float(map_height - 1), ny))

            if target is not None and dist_to_target > 0.01:
                new_dist = math.hypot(tx - nx, ty - ny)
                approach_score = (dist_to_target - new_dist) / (speed * dt + 1e-9)
            else:
                approach_score = 0.0

            cell_x, cell_y = int(round(nx)), int(round(ny))
            est_trav = terrain_memory.get_estimated_traversability(cell_x, cell_y)

            if est_trav is not None:
                # Strong penalty for bad terrain to act as a proper pathfinder for the rover
                if est_trav < traversability_threshold:
                    terrain_score = -3.0
                else:
                    terrain_score = est_trav
            else:
                terrain_score = 0.6

            visits = terrain_memory.visit_count(cell_x, cell_y)
            exploration_bonus = 6.0 / (1.0 + visits)
            if visits > 5:
                exploration_bonus -= (visits - 5) * 0.5 

            lookahead_x = x + 2 * speed * math.cos(angle) * dt
            lookahead_y = y + 2 * speed * math.sin(angle) * dt
            lookahead_x = max(0.0, min(float(map_width - 1), lookahead_x))
            lookahead_y = max(0.0, min(float(map_height - 1), lookahead_y))
            la_cx, la_cy = int(round(lookahead_x)), int(round(lookahead_y))
            la_trav = terrain_memory.get_estimated_traversability(la_cx, la_cy)
            if la_trav is not None and la_trav < traversability_threshold:
                lookahead_penalty = -1.0
            else:
                lookahead_penalty = 0.0

            score = (
                2.5 * approach_score
                + 1.0 * terrain_score
                + exploration_bonus
                + 0.5 * lookahead_penalty
            )
            score += rng.uniform(0, 0.5)

            if score > best_score:
                best_score = score
                best_orientation = angle

        # Execute step
        speed_command = speed
        orientation_command = best_orientation

        result = map_api.step(
            robot_id=robot_id,
            position=(x, y),
            command_velocity=speed_command,
            command_orientation=orientation_command,
        )
        battery_log.append(result.battery_value)

        vx = result.actual_velocity * math.cos(orientation_command) * dt
        vy = result.actual_velocity * math.sin(orientation_command) * dt
        x = max(0.0, min(float(map_width - 1), x + vx))
        y = max(0.0, min(float(map_height - 1), y + vy))
        path.append((x, y))

        # Perceive local terrain and update memory
        observations = map_api.perceive(robot_id=robot_id, position=(x, y))
        all_observations.append(observations)
        terrain_memory.update(observations)

        if result.battery_value <= battery_safety_margin:
            if verbose:
                print(f"  [{robot_id}] Battery depleted – landing for solar recharge.")
            break

        # Check if target cell was observed
        if target is not None and not target_found:
            for obs in observations:
                if int(obs.x) == tx and int(obs.y) == ty:
                    target_found = True
                    if verbose:
                        print(
                            f"  [{robot_id}] *** Target cell ({tx}, {ty}) "
                            f"OBSERVED at step {step_count}! ***"
                        )
                    break

        step_count += 1

    if verbose:
        print(
            f"  [{robot_id}] Flight done: {step_count} steps, "
            f"target_found={target_found}, "
            f"cells_mapped={len(terrain_memory._data)}"
        )
    return path, all_observations, battery_log, target_found, terrain_memory
# ---------------------------------------------------------------------------
# Scout (Rover) – following drone's path step by step
# ---------------------------------------------------------------------------

def scout_follow_and_explore(
    map_api: MapAPI,
    robot_id: str,
    path: list[tuple],
    target: tuple,
    speed: float = 0.5,
    dt: float = 0.1,
    exploration_time: float = 3600.0,
    terrain_memory: TerrainMemory | None = None,
    verbose: bool = True,
    map_width: int = 50,
    map_height: int = 50,
):
    """Scout follows drone path, then explores straight towards target during solar charge time."""
    if len(path) == 0:
        return path, 0.0, 0, False

    map_api.register_robot(robot_id=robot_id, robot_type="scout")
    x, y = path[0]
    total_time = 0.0
    stuck_events = 0
    scout_path = [(x, y)]
    scout_observations = []
    target_found = False

    if verbose:
        print(f"  [{robot_id}] Scout starting from ({x:.1f}, {y:.1f}) to follow drone path.")

    # Phase 1: Follow drone path
    for target_pt in path[1:]:
        tx, ty = target_pt
        while True:
            dx = tx - x
            dy = ty - y
            dist = math.hypot(dx, dy)
            if dist < 0.05:
                x, y = tx, ty
                break
            
            orientation = math.atan2(dy, dx)
            result = map_api.step(robot_id=robot_id, position=(x, y), command_velocity=speed, command_orientation=orientation)
            
            if result.is_stuck:
                stuck_events += 1
                if terrain_memory is not None:
                    terrain_memory.update_stuck(int(round(x)), int(round(y)))
                # Force movement
                x += speed * dt * math.cos(orientation)
                y += speed * dt * math.sin(orientation)
            else:
                x += result.actual_velocity * dt * math.cos(orientation)
                y += result.actual_velocity * dt * math.sin(orientation)
            
            x = max(0.0, min(float(map_width - 1), x))
            y = max(0.0, min(float(map_height - 1), y))
            
            total_time += dt
            scout_path.append((x, y))
            
            obs = map_api.perceive(robot_id=robot_id, position=(x, y))
            scout_observations.extend(obs)
            if terrain_memory is not None:
                terrain_memory.update(obs)

    # Phase 2: Explore straight towards target
    if verbose:
        print(f"  [{robot_id}] Drone path finished. Exploring towards target for {exploration_time}s.")
        
    explore_time = 0.0
    tx, ty = float(target[0]), float(target[1])
    
    while explore_time < exploration_time:
        dx = tx - x
        dy = ty - y
        dist = math.hypot(dx, dy)
        if dist < 0.05:
            target_found = True
            if verbose:
                print(f"  [{robot_id}] Scout reached target!")
            break
            
        orientation = math.atan2(dy, dx)
        result = map_api.step(robot_id=robot_id, position=(x, y), command_velocity=speed, command_orientation=orientation)
        
        if result.is_stuck:
            stuck_events += 1
            if terrain_memory is not None:
                terrain_memory.update_stuck(int(round(x)), int(round(y)))
            # Force movement ignoring stuck
            x += speed * dt * math.cos(orientation)
            y += speed * dt * math.sin(orientation)
        else:
            x += result.actual_velocity * dt * math.cos(orientation)
            y += result.actual_velocity * dt * math.sin(orientation)
            
        x = max(0.0, min(float(map_width - 1), x))
        y = max(0.0, min(float(map_height - 1), y))
        
        explore_time += dt
        total_time += dt
        scout_path.append((x, y))
        
        obs = map_api.perceive(robot_id=robot_id, position=(x, y))
        scout_observations.extend(obs)
        if terrain_memory is not None:
            terrain_memory.update(obs)
            
    if verbose:
        print(f"  [{robot_id}] Scout exploration finished at ({x:.1f}, {y:.1f}). Time: {total_time:.1f}s, Stuck: {stuck_events}")
    
    return scout_path, scout_observations, total_time, stuck_events, target_found


# ---------------------------------------------------------------------------
# Drone – repeated flights until the target is explored
# ---------------------------------------------------------------------------

def explore_until_target_found(
    map_api: MapAPI,
    start: tuple,
    target: tuple,
    speed: float = 2.0,
    dt: float = 0.2,
    map_width: int = 50,
    map_height: int = 50,
    battery_safety_margin: float = 0.05,
    max_flights: int = 500,
    rng_seed: int = 42,
    traversability_threshold: float = 0.3,
):
    """Launch drone flights repeatedly until the target cell is observed.

    Each flight registers a new drone (``drone_1``, ``drone_2``, …) with a
    fresh battery.  A shared ``TerrainMemory`` is passed across flights so
    that each successive drone benefits from what previous flights have
    already mapped — it progressively learns which areas to avoid.

    Returns
    -------
    all_paths : list[list[tuple]]
    all_observations : list[TerrainObservation]   (flat list, merged)
    all_battery_logs : list[list[float]]
    total_flights : int
    """
    all_paths = []
    all_observations = []       # flat list of TerrainObservation
    all_battery_logs = []
    target_found = False

    # Shared terrain memory across all flights
    terrain_memory = TerrainMemory()

    print("=" * 60)
    print(f"  Drone exploration — target: {target}")
    print("=" * 60)

    current_start = start

    for flight in range(1, max_flights + 1):
        drone_id = f"drone_{flight}"
        flight_seed = rng_seed + flight      # vary RNG per flight

        path, obs, battery, found, terrain_memory = smart_drone_exploration(
            map_api,
            robot_id=drone_id,
            start=current_start,
            target=target,
            speed=speed,
            dt=dt,
            map_width=map_width,
            map_height=map_height,
            battery_safety_margin=battery_safety_margin,
            rng_seed=flight_seed,
            verbose=False,
            terrain_memory=terrain_memory,
            traversability_threshold=traversability_threshold,
        )

        all_paths.append(path)
        # obs is list-of-lists; flatten to individual observations
        for step_obs in obs:
            all_observations.extend(step_obs)
        all_battery_logs.append(battery)

        scout_id = f"scout_{flight}"
        print(f"  [flight {flight}] Drone landed to solar charge. Scout '{scout_id}' departing to validate path and explore towards target...")
        scout_path, scout_observations, scout_time, scout_stuck, s_found = scout_follow_and_explore(
            map_api=map_api,
            robot_id=scout_id,
            path=path,
            target=target,
            speed=0.5,
            dt=0.1,
            exploration_time=3600.0,
            terrain_memory=terrain_memory,
            verbose=False,
            map_width=map_width,
            map_height=map_height,
        )
        
        all_paths.append(scout_path)
        all_observations.extend(scout_observations)

        current_start = path[-1]

        if found or s_found:
            target_found = True
            print(
                f"\n>>> Target ({target[0]}, {target[1]}) found on flight "
                f"{flight}! (cells mapped: {len(terrain_memory._data)}) <<<\n"
            )
            break

        # Progress report every 10 flights
        if flight % 10 == 0:
            print(
                f"  ... {flight} flights completed, "
                f"cells mapped: {len(terrain_memory._data)}, "
                f"target not yet found."
            )

    if not target_found:
        print(
            f"\n[WARNING] Target NOT found after {max_flights} flights. "
            f"Consider increasing max_flights.\n"
        )

    # Plot combined results
    _plot_drone_results(
        all_paths, all_battery_logs, all_observations,
        battery_safety_margin, map_width, map_height, target,
    )

    return all_paths, all_observations, all_battery_logs, len(all_paths)


# ---------------------------------------------------------------------------
# Rover – drive to a specific target
# ---------------------------------------------------------------------------

def move_rover_to_target(
    map_api: MapAPI,
    robot_id: str,
    start: tuple,
    target: tuple,
    speed: float,
    dt: float = 0.1,
):
    """Drive the rover from *start* towards *target*.

    The rover is registered via public API with ``robot_type="rover"``.

    Returns
    -------
    final_pos, total_time, stuck_count
    """
    map_api.register_robot(robot_id=robot_id, robot_type="rover")

    x, y = float(start[0]), float(start[1])
    tx, ty = float(target[0]), float(target[1])
    total_time = 0.0
    stuck_count = 0

    print(f"[rover] Start: ({x:.1f}, {y:.1f})  ->  Target: ({tx:.1f}, {ty:.1f})")

    while True:
        dx = tx - x
        dy = ty - y
        dist = math.sqrt(dx**2 + dy**2)

        if dist < 0.001:
            print(
                f"[rover] REACHED target! Steps: {int(total_time / dt)}, "
                f"Stuck events: {stuck_count}"
            )
            break

        orientation = math.atan2(dy, dx)

        result = map_api.step(
            robot_id=robot_id,
            position=(x, y),
            command_velocity=speed,
            command_orientation=orientation,
        )

        if result.is_stuck:
            stuck_count += 1
            print(f"[rover]  t={total_time:.1f}s  STUCK at ({x:.2f}, {y:.2f})  dist={dist:.2f}")
        else:
            x += result.actual_velocity * dt * math.cos(orientation)
            y += result.actual_velocity * dt * math.sin(orientation)

        total_time += dt

        if total_time % 50 == 0:
            print(f"[rover]  t={total_time:.0f}s  pos=({x:.2f}, {y:.2f})  dist={dist:.2f}")

    return (x, y), total_time, stuck_count


# ---------------------------------------------------------------------------
# Plotting helper – supports multiple flights
# ---------------------------------------------------------------------------

def _plot_drone_results(
    all_paths, all_battery_logs, all_observations,
    battery_safety_margin, map_width, map_height,
    target=None,
):
    """Produce a 4-panel figure from one or more drone flights."""

    # Aggregate observations into grids
    texture_sum = np.zeros((map_height, map_width))
    visit_count = np.zeros((map_height, map_width))

    for obs in all_observations:
        ox, oy = int(obs.x), int(obs.y)
        if 0 <= ox < map_width and 0 <= oy < map_height:
            visit_count[oy, ox] += 1
            texture_sum[oy, ox] += obs.features.get("texture", 0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        texture_mean = np.where(visit_count > 0, texture_sum / visit_count, np.nan)

    n_flights = len(all_paths)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_flights, 1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f"Drone Exploration Results ({n_flights} flight{'s' if n_flights != 1 else ''})",
        fontsize=16, fontweight="bold",
    )

    # 1 — Trajectories (one colour per flight)
    ax = axes[0, 0]
    for i, path in enumerate(all_paths):
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        label = f"Flight {i + 1}" if n_flights <= 20 else (f"Flight {i + 1}" if i == 0 or i == n_flights - 1 else None)
        ax.plot(xs, ys, lw=0.7, alpha=0.7, color=colors[i % len(colors)], label=label)
        ax.scatter(xs[0], ys[0], marker="o", s=40, c=[colors[i % len(colors)]],
                   edgecolors="k", zorder=5)
        ax.scatter(xs[-1], ys[-1], marker="X", s=40, c=[colors[i % len(colors)]],
                   edgecolors="k", zorder=5)
    if target:
        ax.scatter(target[0], target[1], marker="*", s=250, c="red",
                   edgecolors="k", zorder=6, label="Target")
    ax.set_xlim(-0.5, map_width - 0.5)
    ax.set_ylim(-0.5, map_height - 0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Drone Trajectories")
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.5)

    # 2 — Battery over time (one curve per flight)
    ax = axes[0, 1]
    for i, blog in enumerate(all_battery_logs):
        label = f"Flight {i + 1}" if n_flights <= 20 else None
        ax.plot(blog, lw=1.0, alpha=0.8, color=colors[i % len(colors)], label=label)
    ax.axhline(battery_safety_margin, color="red", ls="--", lw=0.8,
               label="Safety margin")
    ax.set_xlabel("Step"); ax.set_ylabel("Battery level")
    ax.set_title("Battery Over Time")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, lw=0.3, alpha=0.5)

    # 3 — Mean texture heatmap
    ax = axes[1, 0]
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgrey")
    im = ax.imshow(texture_mean, origin="lower", cmap=cmap, aspect="equal")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean texture")
    if target:
        ax.scatter(target[0], target[1], marker="*", s=200, c="red",
                   edgecolors="k", zorder=6)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Observed Mean Texture")

    # 4 — Observation density
    ax = axes[1, 1]
    dcmap = plt.cm.hot_r.copy()
    dcmap.set_under(color="white")
    im2 = ax.imshow(visit_count, origin="lower", cmap=dcmap, aspect="equal", vmin=1)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="Visit count")
    if target:
        ax.scatter(target[0], target[1], marker="*", s=200, c="cyan",
                   edgecolors="k", zorder=6)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Observation Density")

    plt.tight_layout()
    plt.savefig("drone_exploration_results.png", dpi=150)
    plt.show()
    print("[drone] Plot saved to drone_exploration_results.png")
