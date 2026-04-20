"""
Autonomous Scout Exploration System

This module handles the ground-based exploration and mapping logic for the scout robot.
Key features:
- Two-phase execution with "frontier mop-up" coverage guarantee:
    1. Phase 1: Straight-to-target path, accumulating physical traversal data.
    2. Phase 2: Zigzag return sweep to observe surrounding boundaries.
    3. Frontier Sweep: Dynamically chases nearest unexplored areas until 75% coverage is reached.
- Physical traversability estimation: By recording the real time spent crossing each physical cell, 
  the module learns a data-driven mapping from visual "texture" to actual delay/speed.
- Obstacle ("Stuck") recording: Logs impassable coordinates while continuing simulation flow.
- Includes a sophisticated 4-panel plotting system that translates raw time data into an 
  estimated traversability heat-map for the entire observed environment.
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
# Traversability estimation heuristic
# ---------------------------------------------------------------------------

_N_DIRECTIONS = 16


def _estimate_traversability(texture: float) -> float:
    """Map observed *texture* value to a [0, 1] traversability estimate."""
    return max(0.0, min(1.0, texture))


# ---------------------------------------------------------------------------
# Terrain Memory
# ---------------------------------------------------------------------------

class TerrainMemory:
    """Persistent memory of terrain observations."""

    def __init__(self):
        self._data: dict[tuple[int, int], dict] = {}

    def update(self, observations):
        for obs in observations:
            key = (int(obs.x), int(obs.y))
            if key not in self._data:
                self._data[key] = {
                    "texture_sum": 0.0,
                    "slope_sum": 0.0,
                    "count": 0,
                    "stuck": False,
                    "time_sum": 0.0,
                    "pass_count": 0,
                }
            entry = self._data[key]
            entry["texture_sum"] += obs.features.get("texture", 0.5)
            entry["slope_sum"] += obs.features.get("slope", 0.0)
            entry["count"] += 1

    def record_cell_time(self, x: int, y: int, time_spent: float):
        key = (x, y)
        if key not in self._data:
            self._data[key] = {
                "texture_sum": 0.0,
                "slope_sum": 0.0,
                "count": 0,
                "stuck": False,
                "time_sum": 0.0,
                "pass_count": 0,
            }
        self._data[key]["time_sum"] += time_spent
        self._data[key]["pass_count"] += 1

    def update_stuck(self, x: int, y: int):
        key = (x, y)
        if key not in self._data:
            self._data[key] = {
                "texture_sum": 0.0,
                "slope_sum": 0.0,
                "count": 1,
                "stuck": True,
                "time_sum": 0.0,
                "pass_count": 0,
            }
        else:
            self._data[key]["stuck"] = True

    def get_estimated_traversability(self, x: int, y: int) -> float | None:
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

    def known_cells_count(self) -> int:
        return len(self._data)

    def get_coverage(self, map_width: int, map_height: int) -> float:
        total_cells = map_width * map_height
        if total_cells == 0:
            return 0.0
        return self.known_cells_count() / total_cells


# ---------------------------------------------------------------------------
# Zigzag waypoint generation around a straight path
# ---------------------------------------------------------------------------

def _generate_return_zigzag(
    start: tuple,
    target: tuple,
    amplitude: float,
    spacing: float,
    map_width: int,
    map_height: int,
) -> list[tuple[float, float]]:
    """Generate zigzag waypoints from *target* back to *start*.

    The waypoints oscillate left/right of the straight line connecting
    target -> start, with the given *amplitude* (perpendicular offset)
    and *spacing* (distance along the path between consecutive zigs).
    """
    tx, ty = float(target[0]), float(target[1])
    sx, sy = float(start[0]), float(start[1])

    dx = sx - tx
    dy = sy - ty
    path_len = math.hypot(dx, dy)

    if path_len < 0.01:
        return [(sx, sy)]

    # Normalised direction (target -> start) and perpendicular
    dir_x, dir_y = dx / path_len, dy / path_len
    perp_x, perp_y = -dir_y, dir_x

    waypoints: list[tuple[float, float]] = []
    num_points = max(1, int(path_len / spacing))

    for i in range(num_points + 1):
        t = i / num_points
        px = tx + t * dx
        py = ty + t * dy

        side = 1 if i % 2 == 0 else -1
        wx = px + side * amplitude * perp_x
        wy = py + side * amplitude * perp_y

        # Clamp to map bounds
        wx = max(0.0, min(float(map_width - 1), wx))
        wy = max(0.0, min(float(map_height - 1), wy))

        waypoints.append((wx, wy))

    # Always finish at start
    waypoints.append((sx, sy))
    return waypoints


# ---------------------------------------------------------------------------
# Core helper: navigate toward a goal one step at a time
# ---------------------------------------------------------------------------

def _pick_best_direction(
    x: float, y: float,
    goal_x: float, goal_y: float,
    speed: float, dt: float,
    map_width: int, map_height: int,
    terrain_memory: TerrainMemory,
    traversability_threshold: float,
    rng: random.Random,
    approach_weight: float = 4.0,
) -> float:
    """Return the best orientation (radians) among 16 candidates."""
    best_orientation = math.atan2(goal_y - y, goal_x - x)
    best_score = -math.inf
    dist_to_goal = math.hypot(goal_x - x, goal_y - y)

    for i in range(_N_DIRECTIONS):
        angle = -math.pi + 2 * math.pi * i / _N_DIRECTIONS

        nx = x + speed * math.cos(angle) * dt
        ny = y + speed * math.sin(angle) * dt
        nx = max(0.0, min(float(map_width - 1), nx))
        ny = max(0.0, min(float(map_height - 1), ny))

        # Approach score
        if dist_to_goal > 0.01:
            new_dist = math.hypot(goal_x - nx, goal_y - ny)
            approach = (dist_to_goal - new_dist) / (speed * dt + 1e-9)
        else:
            approach = 0.0

        cell_x, cell_y = int(round(nx)), int(round(ny))

        # Terrain quality
        est_trav = terrain_memory.get_estimated_traversability(cell_x, cell_y)
        if est_trav is not None:
            terrain_score = -3.0 if est_trav < traversability_threshold else est_trav
        else:
            terrain_score = 0.6

        # Exploration bonus
        visits = terrain_memory.visit_count(cell_x, cell_y)
        exploration_bonus = 3.0 / (1.0 + visits)

        score = (
            approach_weight * approach
            + 1.0 * terrain_score
            + exploration_bonus
            + rng.uniform(0, 0.2)
        )

        if score > best_score:
            best_score = score
            best_orientation = angle

    return best_orientation


# ---------------------------------------------------------------------------
# Scout – straight to target, then zigzag return with 75 % coverage
# ---------------------------------------------------------------------------

def scout_coverage_exploration(
    map_api: MapAPI,
    robot_id: str = "scout_explorer",
    start: tuple = (3, 3),
    target: tuple = (49, 49),
    speed: float = 0.5,
    dt: float = 0.5,
    map_width: int = 50,
    map_height: int = 50,
    coverage_goal: float = 0.75,
    rng_seed: int = 42,
    verbose: bool = True,
    terrain_memory: TerrainMemory | None = None,
    traversability_threshold: float = 0.3,
    max_steps: int = 200_000,
    zigzag_amplitude: float = 20.0,
    zigzag_spacing: float = 4.0,
):
    """Scout exploration in two phases.

    **Phase 1 -- Straight to target**
    Navigate directly toward *target*, calling ``perceive`` at every step
    and recording any stuck events.

    **Phase 2 -- Zigzag return to start**
    Generate zigzag waypoints that oscillate around the straight outbound
    path (with configurable *zigzag_amplitude*) and navigate back to
    *start*.  Continue zigzagging until map coverage >= *coverage_goal*
    **and** the scout has returned to *start*.

    At the end, a summary is printed with:
    - Total stuck events and their cell coordinates
    - Total exploration time (steps * dt)
    - Final coverage percentage

    Returns
    -------
    path : list[tuple]
    all_observations : list[TerrainObservation]
    terrain_memory : TerrainMemory
    stuck_cells : list[tuple[int,int]]
    """
    rng = random.Random(rng_seed)

    if terrain_memory is None:
        terrain_memory = TerrainMemory()

    map_api.register_robot(robot_id=robot_id, robot_type="scout")

    x, y = float(start[0]), float(start[1])
    tx, ty = float(target[0]), float(target[1])
    sx, sy = float(start[0]), float(start[1])

    path: list[tuple] = [(x, y)]
    all_observations: list = []
    stuck_cells: list[tuple[int, int]] = []
    stuck_count = 0
    step_count = 0

    total_cells = map_width * map_height
    needed = int(total_cells * coverage_goal)

    current_cell = (int(round(x)), int(round(y)))
    time_in_current_cell = 0.0

    if verbose:
        print(f"  [{robot_id}] Starting at ({x:.1f}, {y:.1f}), speed={speed}, dt={dt}")
        print(f"  [{robot_id}] Phase 1: straight to target ({tx:.0f}, {ty:.0f})")

    # --- Perceive at start ---
    obs = map_api.perceive(robot_id=robot_id, position=(x, y))
    all_observations.extend(obs)
    terrain_memory.update(obs)

    # ==================================================================
    # PHASE 1 — Straight to target
    # ==================================================================
    while step_count < max_steps:
        dist_to_target = math.hypot(tx - x, ty - y)
        if dist_to_target < 0.5:
            if verbose:
                cov = terrain_memory.get_coverage(map_width, map_height)
                print(
                    f"  [{robot_id}] Reached target at step {step_count}. "
                    f"Coverage so far: {cov*100:.1f}%"
                )
            break

        orientation = math.atan2(ty - y, tx - x)

        result = map_api.step(
            robot_id=robot_id,
            position=(x, y),
            command_velocity=speed,
            command_orientation=orientation,
        )

        if result.is_stuck:
            stuck_count += 1
            cell = (int(round(x)), int(round(y)))
            if cell not in stuck_cells:
                stuck_cells.append(cell)
            terrain_memory.update_stuck(cell[0], cell[1])
            # Force movement
            x += speed * dt * math.cos(orientation)
            y += speed * dt * math.sin(orientation)
        else:
            x += result.actual_velocity * dt * math.cos(orientation)
            y += result.actual_velocity * dt * math.sin(orientation)

        x = max(0.0, min(float(map_width - 1), x))
        y = max(0.0, min(float(map_height - 1), y))
        path.append((x, y))

        # Always perceive
        obs = map_api.perceive(robot_id=robot_id, position=(x, y))
        all_observations.extend(obs)
        terrain_memory.update(obs)

        # Track time spent in cell
        new_cell = (int(round(x)), int(round(y)))
        if new_cell == current_cell:
            time_in_current_cell += dt
        else:
            terrain_memory.record_cell_time(current_cell[0], current_cell[1], time_in_current_cell)
            current_cell = new_cell
            time_in_current_cell = dt

        step_count += 1

    phase1_steps = step_count

    # ==================================================================
    # PHASE 2 — Zigzag return + frontier exploration until coverage goal
    # ==================================================================
    if verbose:
        print(
            f"  [{robot_id}] Phase 2: zigzag return to start ({sx:.0f}, {sy:.0f}), "
            f"amplitude={zigzag_amplitude}, coverage goal={coverage_goal*100:.0f}%"
        )

    # Step 2a: One zigzag pass from current position back to start
    zigzag_wps = _generate_return_zigzag(
        start=start,
        target=(x, y),
        amplitude=zigzag_amplitude,
        spacing=zigzag_spacing,
        map_width=map_width,
        map_height=map_height,
    )

    if verbose:
        print(f"  [{robot_id}] Zigzag waypoints: {len(zigzag_wps)}")

    wp_idx = 0
    coverage_met = False
    mode = "zigzag"  # "zigzag" → "frontier" → "return"

    while step_count < max_steps:
        # --- Check coverage ---
        cov = terrain_memory.get_coverage(map_width, map_height)
        if cov >= coverage_goal and not coverage_met:
            coverage_met = True
            mode = "return"
            if verbose:
                print(
                    f"  [{robot_id}] Coverage goal reached: {cov*100:.1f}% "
                    f"({terrain_memory.known_cells_count()}/{total_cells}) "
                    f"at step {step_count}. Returning to start..."
                )

        # --- Determine goal ---
        if mode == "return":
            dist_to_start = math.hypot(sx - x, sy - y)
            if dist_to_start < 0.5:
                if verbose:
                    print(f"  [{robot_id}] Returned to start at step {step_count}")
                break
            goal_x, goal_y = sx, sy

        elif mode == "zigzag":
            # Advance waypoints
            while wp_idx < len(zigzag_wps):
                wp = zigzag_wps[wp_idx]
                if math.hypot(wp[0] - x, wp[1] - y) < 1.5:
                    wp_idx += 1
                else:
                    break

            if wp_idx < len(zigzag_wps):
                goal_x, goal_y = zigzag_wps[wp_idx]
            else:
                # Zigzag done — switch to frontier if coverage not met
                if not coverage_met:
                    mode = "frontier"
                    if verbose:
                        print(
                            f"  [{robot_id}] Zigzag done, coverage={cov*100:.1f}%. "
                            f"Switching to frontier exploration..."
                        )
                    continue
                else:
                    mode = "return"
                    continue

        elif mode == "frontier":
            # Navigate to nearest unvisited cell
            fx, fy = _find_nearest_unvisited(
                terrain_memory, x, y, map_width, map_height,
            )
            goal_x, goal_y = fx, fy

        # --- Pick direction (16-direction scoring) ---
        orientation = _pick_best_direction(
            x, y, goal_x, goal_y,
            speed, dt, map_width, map_height,
            terrain_memory, traversability_threshold, rng,
            approach_weight=4.0,
        )

        # --- Execute step ---
        result = map_api.step(
            robot_id=robot_id,
            position=(x, y),
            command_velocity=speed,
            command_orientation=orientation,
        )

        if result.is_stuck:
            stuck_count += 1
            cell = (int(round(x)), int(round(y)))
            if cell not in stuck_cells:
                stuck_cells.append(cell)
            terrain_memory.update_stuck(cell[0], cell[1])
            x += speed * dt * math.cos(orientation)
            y += speed * dt * math.sin(orientation)
        else:
            x += result.actual_velocity * dt * math.cos(orientation)
            y += result.actual_velocity * dt * math.sin(orientation)

        x = max(0.0, min(float(map_width - 1), x))
        y = max(0.0, min(float(map_height - 1), y))
        path.append((x, y))

        # --- Always perceive ---
        obs = map_api.perceive(robot_id=robot_id, position=(x, y))
        all_observations.extend(obs)
        terrain_memory.update(obs)

        step_count += 1

        # Progress report
        if verbose and step_count % 2000 == 0:
            cov = terrain_memory.get_coverage(map_width, map_height)
            print(
                f"  [{robot_id}] step {step_count}: "
                f"coverage {cov*100:.1f}% "
                f"({terrain_memory.known_cells_count()}/{total_cells}), "
                f"mode={mode}, stucks: {stuck_count}"
            )

    # ==================================================================
    # SUMMARY
    # ==================================================================
    # Record the time for the final cell
    terrain_memory.record_cell_time(current_cell[0], current_cell[1], time_in_current_cell)

    total_time = step_count * dt
    final_cov = terrain_memory.get_coverage(map_width, map_height)

    print()
    print(f"  [{robot_id}] === EXPLORATION SUMMARY ===")
    print(f"  [{robot_id}] Total steps    : {step_count} (Phase 1: {phase1_steps}, Phase 2: {step_count - phase1_steps})")
    print(f"  [{robot_id}] Total time     : {total_time:.1f}s")
    print(f"  [{robot_id}] Final coverage : {final_cov*100:.1f}% ({terrain_memory.known_cells_count()}/{total_cells})")
    print(f"  [{robot_id}] Stuck events   : {stuck_count}")
    print(f"  [{robot_id}] Unique stuck cells ({len(stuck_cells)}):")
    for cell in stuck_cells:
        print(f"      -> cell ({cell[0]}, {cell[1]})")
    print(f"  [{robot_id}] Final position : ({x:.2f}, {y:.2f})")
    print()

    return path, all_observations, terrain_memory, stuck_cells


# ---------------------------------------------------------------------------
# Helper: find nearest unvisited cell
# ---------------------------------------------------------------------------

def _find_nearest_unvisited(
    terrain_memory: TerrainMemory,
    x: float,
    y: float,
    map_width: int,
    map_height: int,
) -> tuple[float, float]:
    """Return the nearest cell that has not been observed yet.

    Falls back to the map centre if every cell is already known.
    """
    best: tuple[float, float] | None = None
    best_dist = math.inf

    # Check cells at the frontier of known territory
    for (kx, ky) in list(terrain_memory._data.keys()):
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = kx + dx, ky + dy
            if 0 <= nx < map_width and 0 <= ny < map_height:
                if not terrain_memory.is_known(nx, ny):
                    d = math.hypot(nx - x, ny - y)
                    if d < best_dist:
                        best_dist = d
                        best = (float(nx), float(ny))

    if best is not None:
        return best
    return (float(map_width // 2), float(map_height // 2))


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------

def plot_scout_results(
    path: list[tuple],
    terrain_memory: TerrainMemory,
    map_width: int = 50,
    map_height: int = 50,
    target: tuple | None = None,
    start: tuple | None = None,
    phase1_steps: int = 0,
    stuck_cells: list[tuple[int, int]] | None = None,
    save_path: str = "scout_exploration_results.png",
):
    """Produce a 4-panel figure for the scout exploration run."""

    texture_sum = np.zeros((map_height, map_width))
    visit_count = np.zeros((map_height, map_width))
    
    # 1. Learn mapping from texture to time using physically traversed cells
    bins = np.linspace(0, 1, 11)
    bin_time_sums = np.zeros(10)
    bin_time_counts = np.zeros(10)
    
    for (cx, cy), data in terrain_memory._data.items():
        if data.get("pass_count", 0) > 0 and data.get("count", 0) > 0:
            avg_time = data["time_sum"] / data["pass_count"]
            mean_tex = data["texture_sum"] / data["count"]
            bin_idx = int(max(0.0, min(mean_tex, 0.999)) * 10)
            bin_time_sums[bin_idx] += avg_time
            bin_time_counts[bin_idx] += 1
            
    # Interpolate empty bins
    valid_bins = np.where(bin_time_counts > 0)[0]
    bin_means = np.full(10, 2.0) # default 2.0s
    if len(valid_bins) > 0:
        for i in range(10):
            if bin_time_counts[i] > 0:
                bin_means[i] = bin_time_sums[i] / bin_time_counts[i]
            else:
                nearest = valid_bins[np.argmin(np.abs(valid_bins - i))]
                bin_means[i] = bin_time_sums[nearest] / bin_time_counts[nearest]
    
    # 2. Render traversability based on time (actual or estimated)
    # 2.0 s is ideal for speed=0.5 (1 unit distance covered in 2s)
    # 1.0 score is perfectly traversable, 0.0 is very bad, -1.0 is stuck
    trav_grid = np.full((map_height, map_width), np.nan)

    for (cx, cy), data in terrain_memory._data.items():
        if 0 <= cx < map_width and 0 <= cy < map_height:
            visit_count[cy, cx] = data.get("count", 0)
            texture_sum[cy, cx] = data.get("texture_sum", 0.0)
            
            if data.get("count", 0) > 0:
                if data.get("stuck", False):
                    trav_grid[cy, cx] = -1.0  # Special value for stuck
                else:
                    if data.get("pass_count", 0) > 0:
                        # We have physical data
                        avg_time = data["time_sum"] / data["pass_count"]
                    else:
                        # Estimate from texture
                        mean_tex = data["texture_sum"] / data["count"]
                        bin_idx = int(max(0.0, min(mean_tex, 0.999)) * 10)
                        avg_time = bin_means[bin_idx]
                        
                    # Convert time to a score. Ideal time is ~2.0
                    score = min(2.0 / max(avg_time, 0.1), 1.0)
                    trav_grid[cy, cx] = max(0.0, score)

    with np.errstate(divide="ignore", invalid="ignore"):
        texture_mean = np.where(visit_count > 0, texture_sum / visit_count, np.nan)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Scout Exploration: Straight + Zigzag Return", fontsize=16, fontweight="bold")

    # ---- Panel 1: Trajectory ----
    ax = axes[0, 0]
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]

    if phase1_steps > 0 and phase1_steps < len(path):
        ax.plot(
            xs[:phase1_steps], ys[:phase1_steps],
            lw=0.8, alpha=0.7, color="dodgerblue", label="Phase 1 (to target)",
        )
        ax.plot(
            xs[phase1_steps:], ys[phase1_steps:],
            lw=0.5, alpha=0.5, color="orangered", label="Phase 2 (zigzag return)",
        )
    else:
        ax.plot(xs, ys, lw=0.5, alpha=0.6, color="dodgerblue", label="Scout path")

    if start:
        ax.scatter(start[0], start[1], marker="o", s=100, c="lime",
                   edgecolors="k", zorder=6, label="Start")
    if target:
        ax.scatter(target[0], target[1], marker="*", s=250, c="red",
                   edgecolors="k", zorder=6, label="Target")
    if stuck_cells:
        sx_list = [c[0] for c in stuck_cells]
        sy_list = [c[1] for c in stuck_cells]
        ax.scatter(sx_list, sy_list, marker="x", s=50, c="magenta",
                   zorder=5, label=f"Stuck cells ({len(stuck_cells)})")

    ax.set_xlim(-0.5, map_width - 0.5)
    ax.set_ylim(-0.5, map_height - 0.5)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Scout Trajectory")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.grid(True, lw=0.3, alpha=0.5)

    # ---- Panel 2: Coverage over time ----
    ax = axes[0, 1]
    # Reconstruct coverage curve from terrain memory
    cov = terrain_memory.get_coverage(map_width, map_height)
    # We can just plot a simple bar chart or final stat since we don't have chronological obs
    ax.bar(["Coverage"], [cov * 100], color="teal")
    ax.axhline(75, color="red", ls="--", lw=0.8, label="75% goal")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Coverage (%)")
    ax.set_title(f"Final Map Coverage: {cov*100:.1f}%")
    ax.legend(fontsize=8)
    ax.grid(True, lw=0.3, alpha=0.5, axis="y")

    # ---- Panel 3: Mean texture heatmap ----
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

    # ---- Panel 4: Estimated Traversability (Time-based) ----
    ax = axes[1, 1]
    tcmap = plt.cm.RdYlGn.copy()
    tcmap.set_under(color="black")  # Black for stuck
    tcmap.set_bad(color="lightgrey")  # Grey for unvisited
    
    im2 = ax.imshow(trav_grid, origin="lower", cmap=tcmap, aspect="equal", vmin=-0.01, vmax=1.0)
    fig.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label="Traversability (Green=Good, Red=Bad, Black=Stuck)")
    if target:
        ax.scatter(target[0], target[1], marker="*", s=200, c="cyan",
                   edgecolors="k", zorder=6)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title("Estimated Traversability")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"[scout] Plot saved to {save_path}")
