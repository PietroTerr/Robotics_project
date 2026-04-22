"""
Governor Module
===============

Master orchestrator for the three-agent coordinated terrain-exploration mission.

Mission Phases
--------------
Phase 1 — Forward Trip (home → target):
    Drone  : Flies in a straight beeline toward the target; calls perceive() at every
             step to fill the terrain map with visual observations (texture, color,
             slope!!!!!). Battery-constrained: if the battery drops below threshold the
             drone idles and recharges automatically (handled by Drone.step_towards).
    Scout  : Follows a zigzag pattern in the neighbourhood of the drone's straight-
             line path, sweeping a corridor ±ZIG_WIDTH cells wide perpendicular to
             the home→target axis.  Perceives and records real traversability at
             every step.

Phase 2 — Return Trip (target → home):
    Drone  : Flies back home but now biases its direction toward the nearest
             unvisited cells (prefer_unexplored=True), maximising new coverage.
    Scout  : Same zigzag strategy, but when choosing which side to sweep it picks
             the side that still has more unvisited cells.

Phase 3 — Graph Construction + A*:
    TerrainMap.refresh_estimation() runs IDW to propagate traversability estimates
    to cells that were only perceived visually.  build_weighted_graph() lifts the
    grid into a NetworkX DiGraph.  A* (networkx.astar_path) finds the cheapest path
    from home to target.

Phase 4 — Rover Traversal:
    Rover follows the A* waypoints cell-by-cell via Rover.step_towards().
    It is immobilised on the first stuck event (heavy ground vehicle).

Usage
-----
    python governor.py
"""

import math
import sys
import time
from pathlib import Path
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.map_api import MapAPI
from data_management import TerrainMap, build_weighted_graph
from motion import Drone, Scout, Rover

# ─── Mission Parameters ───────────────────────────────────────────────────────
START      = (5, 5)     # Shared home position for all agents
TARGET     = (40, 40)   # Exploration target

DT         = 0.2        # Simulation time-step (seconds)
ZIG_WIDTH  = 10          # Half-width of the scout zigzag corridor (cells)
MAX_STEPS  = 120_000    # Safety cap per phase (avoids infinite loops)

MAP_CSV    = PROJECT_ROOT / "src" / "map_001_seed42.csv"


# ─── Geometry Helpers ─────────────────────────────────────────────────────────

def _axis_perp(p_from: tuple, p_to: tuple):
    """
    Return (unit_axis, unit_perp) for the straight line p_from → p_to.
    unit_perp is 90° CCW from unit_axis.
    """
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    L = math.hypot(dx, dy)
    if L < 1e-9:
        return (1.0, 0.0), (0.0, 1.0)
    ax, ay = dx / L, dy / L
    return (ax, ay), (-ay, ax)


def _axial_projection(point: tuple, origin: tuple, axis: tuple) -> float:
    """Signed scalar projection of `point - origin` onto `axis`."""
    return (point[0] - origin[0]) * axis[0] + (point[1] - origin[1]) * axis[1]


def _nearest_unexplored(pos: tuple, axis: tuple, perp: tuple,
                         terrain_map: TerrainMap,
                         look: int = 6, spread: int = 2) -> tuple:
    """
    Scan a look × (2·spread+1) grid ahead of `pos` along the flight axis and
    return the nearest unvisited cell as a float waypoint.
    Falls back to pos + look*axis if every candidate is already explored.
    """
    for k in range(1, look + 1):
        for s in range(-spread, spread + 1):
            cx = int(round(pos[0] + k * axis[0] + s * perp[0]))
            cy = int(round(pos[1] + k * axis[1] + s * perp[1]))
            if (cx, cy) not in terrain_map.grid:
                return float(cx), float(cy)
    # Fallback: look cells straight ahead
    return pos[0] + look * axis[0], pos[1] + look * axis[1]


def _scout_side_prefer_unexplored(base_x: float, base_y: float,
                                   perp: tuple, zig_width: float,
                                   current_side: int,
                                   terrain_map: TerrainMap) -> int:
    """
    Return the zigzag side (+1 or -1) that has more unvisited cells.
    Checks zig_width probes in each perpendicular direction from (base_x, base_y).
    Falls back to the opposite of current_side (normal flip) if counts are equal.
    """
    def _count(s: int) -> int:
        n = 0
        for k in range(1, int(zig_width) + 1):
            cc = (int(round(base_x + s * k * perp[0])),
                  int(round(base_y + s * k * perp[1])))
            if cc not in terrain_map.grid:
                n += 1
        return n

    count_pos = _count(1)
    count_neg = _count(-1)
    if count_pos == count_neg:
        return -current_side   # simple alternation when tied
    return 1 if count_pos > count_neg else -1


# ─── Phase Runner ─────────────────────────────────────────────────────────────

def _run_phase(name: str,
               drone: Drone,
               scout: Scout,
               goal: tuple,
               origin: tuple,
               terrain_map: TerrainMap,
               prefer_unexplored: bool = False) -> tuple:
    """
    Advance drone and scout concurrently (one DT tick per iteration) until
    both reach `goal` or MAX_STEPS is exhausted.

    Drone behaviour
    ~~~~~~~~~~~~~~~
    - Always calls drone.step_towards() then drone.perceive().
    - When prefer_unexplored=False: target is the straight-line goal.
    - When prefer_unexplored=True:  target is a 40/60 blend of the true goal
      and the nearest unvisited cell ahead, so the drone drifts toward new cells
      while still converging on home.

    Scout behaviour
    ~~~~~~~~~~~~~~~
    - Computes a zigzag waypoint at ±ZIG_WIDTH cells perpendicular to the
      axis formed by (origin → goal), referenced to the drone's axial progress.
    - When within 2×ZIG_WIDTH cells of the goal, aims straight at it.
    - When prefer_unexplored=True: chooses the perpendicular side with the
      most unvisited cells at every waypoint flip.
    - Records real traversability from actual vs commanded velocity.

    Returns
    -------
    tuple: (steps, drone_traj, scout_traj)
        steps       — number of DT steps consumed
        drone_traj  — list of (x, y) sampled from the drone every step
        scout_traj  — list of (x, y) sampled from the scout every step
    """
    axis, perp = _axis_perp(origin, goal)

    print(f"\n{'='*62}")
    print(f"  {name}")
    print(f"  Origin={origin}  Goal={goal}  prefer_unexplored={prefer_unexplored}")
    print(f"{'='*62}")

    drone_done  = False
    scout_done  = False
    scout_side  = 1          # +1 or -1 for current zigzag side
    scout_wp    = None       # current scout zigzag waypoint (x, y)
    step        = 0
    drone_traj: list = []    # recorded (x, y) positions
    scout_traj: list = []

    while not (drone_done and scout_done) and step < MAX_STEPS:
        drone_traj.append((drone.x, drone.y))
        scout_traj.append((scout.x, scout.y))

        # ── Drone ─────────────────────────────────────────────────────
        if not drone_done:
            if prefer_unexplored:
                ux, uy = _nearest_unexplored(
                    (drone.x, drone.y), axis, perp, terrain_map, look=5, spread=2)
                # Blend: 40% true goal + 60% nearest unexplored cell
                tx = 0.4 * goal[0] + 0.6 * ux
                ty = 0.4 * goal[1] + 0.6 * uy
            else:
                tx, ty = float(goal[0]), float(goal[1])

            # step_towards handles battery / recharge internally
            drone.step_towards(tx, ty, DT)

            # Perceive the neighbourhood and store observations
            obs = drone.perceive()
            terrain_map.store_observation(obs)

            if math.hypot(drone.x - goal[0], drone.y - goal[1]) < 1.5:
                drone_done = True
                print(f"  [step {step:6d}] 🛰  Drone reached goal {goal}")

        # ── Scout ─────────────────────────────────────────────────────
        if not scout_done:
            dist_to_goal = math.hypot(scout.x - goal[0], scout.y - goal[1])

            if dist_to_goal < ZIG_WIDTH * 2.0:
                # Close to goal: go straight
                scout_tx, scout_ty = float(goal[0]), float(goal[1])
            else:
                # ─ Zigzag waypoint computation ─
                # Base point: drone's progress projected on axis from origin
                proj = _axial_projection((drone.x, drone.y), origin, axis)
                base_x = origin[0] + proj * axis[0]
                base_y = origin[1] + proj * axis[1]

                # Recompute waypoint when None or when scout has arrived at it
                if scout_wp is None or \
                   math.hypot(scout.x - scout_wp[0], scout.y - scout_wp[1]) < 1.5:
                    if prefer_unexplored:
                        scout_side = _scout_side_prefer_unexplored(
                            base_x, base_y, perp, ZIG_WIDTH,
                            scout_side, terrain_map)
                    else:
                        scout_side = -scout_side  # simple alternation

                    scout_wp = (
                        base_x + scout_side * ZIG_WIDTH * perp[0],
                        base_y + scout_side * ZIG_WIDTH * perp[1],
                    )

                scout_tx, scout_ty = scout_wp

            # Record position before step to compute actual velocity
            pre_x, pre_y = scout.x, scout.y

            s_reached, was_stuck, _ = scout.step_towards(scout_tx, scout_ty, DT)

            # Perceive from current scout position
            obs = scout.perceive()
            terrain_map.store_observation(obs)

            # Store real traversability derived from actual vs commanded displacement
            cx, cy = int(round(scout.x)), int(round(scout.y))
            cell = terrain_map.get_cell(cx, cy)
            if cell.real_traversability is None:
                actual_dist   = math.hypot(scout.x - pre_x, scout.y - pre_y)
                expected_dist = scout.speed * DT
                if expected_dist > 1e-9:
                    trav = actual_dist / expected_dist
                else:
                    trav = 1.0
                cell.real_traversability = max(0.01, min(1.0, trav))

            if was_stuck:
                cell.is_stuck = True
                # Override with a low traversability on explicit stuck events
                cell.real_traversability = min(cell.real_traversability, 0.1)

            if math.hypot(scout.x - goal[0], scout.y - goal[1]) < 1.5:
                scout_done = True
                print(f"  [step {step:6d}] 🚙  Scout reached goal {goal}")

        step += 1

        # ── Progress log every 1000 steps ────────────────────────────
        if step % 1000 == 0:
            d_dist = math.hypot(drone.x - goal[0], drone.y - goal[1])
            s_dist = math.hypot(scout.x - goal[0], scout.y - goal[1])
            print(f"  step={step:6d} | 🛰 drone dist={d_dist:6.2f} | "
                  f"🚙 scout dist={s_dist:6.2f} | 📦 cells={len(terrain_map.grid)}")

    if step >= MAX_STEPS:
        print(f"  ⚠  Phase reached MAX_STEPS ({MAX_STEPS:,}). Forcing next phase.")

    print(f"\n  {name} completed in {step:,} steps. "
          f"TerrainMap coverage: {len(terrain_map.grid)} cells.")
    return step, drone_traj, scout_traj


# ─── Governor ─────────────────────────────────────────────────────────────────

def governor():
    """
    Top-level mission orchestrator.

    1. Initialises MapAPI, TerrainMap, Drone, Scout, and Rover.
    2. Runs the forward trip (Phase 1) and return trip (Phase 2).
    3. Builds the weighted terrain graph and runs A* (Phase 3).
    4. Drives the Rover along the optimal path (Phase 4).
    5. Prints a final mission summary.
    """
    print("\n" + "█" * 62)
    print("  THREE-AGENT TERRAIN EXPLORATION MISSION")
    print("█" * 62)
    print(f"  Home:       {START}")
    print(f"  Target:     {TARGET}")
    print(f"  DT:         {DT} s")
    print(f"  ZIG_WIDTH:  {ZIG_WIDTH} cells")
    print(f"  MAX_STEPS:  {MAX_STEPS:,}")

    # ── Initialise simulation environment ─────────────────────────
    map_api     = MapAPI(terrain=MAP_CSV, rng_seed=42, time_step=DT)
    terrain_map = TerrainMap(width=50, height=50)

    drone = Drone(map_api, "drone_1", START)
    scout = Scout(map_api, "scout_1", START)
    rover = Rover(map_api, "rover_1", START)

    total_steps = 0
    wall_start  = time.perf_counter()
    drone_traj_fwd: list = []
    drone_traj_ret: list = []
    scout_traj_fwd: list = []
    scout_traj_ret: list = []

    # ─────────────────────────────────────────────────────────────
    # PHASE 1: Forward Trip — home → target
    # Drone flies straight, perceiving along the path.
    # Scout zigzags; both build the map together.
    # ─────────────────────────────────────────────────────────────
    steps1, drone_traj_fwd, scout_traj_fwd = _run_phase(
        "[PHASE 1] FORWARD TRIP — home → target",
        drone, scout,
        goal   = TARGET,
        origin = START,
        terrain_map      = terrain_map,
        prefer_unexplored = False,
    )
    total_steps += steps1

    # ─────────────────────────────────────────────────────────────
    # PHASE 2: Return Trip — target → home
    # Drone biases toward unexplored cells while returning.
    # Scout picks the zigzag side with more unvisited cells.
    # ─────────────────────────────────────────────────────────────
    steps2, drone_traj_ret, scout_traj_ret = _run_phase(
        "[PHASE 2] RETURN TRIP — target → home",
        drone, scout,
        goal   = START,
        origin = TARGET,
        terrain_map      = terrain_map,
        prefer_unexplored = True,
    )
    total_steps += steps2

    # ─────────────────────────────────────────────────────────────
    # PHASE 3: Graph Construction + A* Path Planning
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  [PHASE 3] GRAPH CONSTRUCTION + A* PATH PLANNING")
    print(f"{'='*62}")
    print(f"  Running IDW estimation over {len(terrain_map.grid)} cells …")

    terrain_map.refresh_estimation()

    G = build_weighted_graph(terrain_map)
    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges.")

    rover_start = (int(round(rover.x)), int(round(rover.y)))
    rover_goal  = TARGET

    # Ensure start / goal nodes exist in the graph
    for node in [rover_start, rover_goal]:
        if node not in G:
            print(f"  ⚠  Node {node} missing from graph — stitching it in.")
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nb = (node[0] + dx, node[1] + dy)
                    if nb in G:
                        G.add_edge(node, nb, weight=1.0)
                        G.add_edge(nb, node, weight=1.0)

    try:
        heuristic = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])

        optimal_path = nx.astar_path(
            G,
            source    = rover_start,
            target    = rover_goal,
            heuristic = heuristic,
            weight    = "weight",
        )
        total_cost = sum(
            G[optimal_path[i]][optimal_path[i + 1]]["weight"]
            for i in range(len(optimal_path) - 1)
        )

        print(f"  ✅ A* path found: {len(optimal_path)} waypoints, "
              f"total cost = {total_cost:.2f}")
        print(f"  Path preview: {optimal_path[:4]} … {optimal_path[-3:]}")

    except nx.NetworkXNoPath:
        print("  ❌ A* found no path between start and target.")
        print("     Exploration coverage may be too low. Aborting.")
        return
    except nx.NodeNotFound as exc:
        print(f"  ❌ Graph node not found: {exc}. Aborting.")
        return

    # ─────────────────────────────────────────────────────────────
    # PHASE 4: Rover Traversal of Optimal Path
    # ─────────────────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  [PHASE 4] ROVER TRAVERSAL OF OPTIMAL PATH")
    print(f"{'='*62}")

    mission_success = False
    rover_max_wp_steps = 50_000   # per-waypoint safety cap
    for i, wp in enumerate(optimal_path[1:], start=1):
        cell  = terrain_map.grid.get(wp)
        trav  = (f"{cell.traversability_estimate:.2f}"
                 if cell and cell.traversability_estimate is not None else "N/A")
        sp    = (f"{cell.stuck_probability_estimate:.2f}"
                 if cell else "N/A")

        # Drive the rover in a tight loop until it reaches this waypoint
        wp_steps = 0
        r_stuck  = False
        while math.hypot(rover.x - wp[0], rover.y - wp[1]) >= 0.15 \
              and wp_steps < rover_max_wp_steps:
            r_reached, r_stuck = rover.step_towards(wp[0], wp[1], DT)
            wp_steps += 1
            if r_stuck:
                break

        flag = "🔴 STUCK" if r_stuck else "🟢 OK"
        print(f"  [{i:3d}/{len(optimal_path)-1}] → {wp} | "
              f"trav_est={trav}  stuck_p={sp} | {flag}  ({wp_steps} steps)")

        if r_stuck:
            print("\n  ⚠  Rover immobilized. Mission failed at waypoint "
                  f"{i}/{len(optimal_path)-1}.")
            break
    else:
        mission_success = True
        print(f"\n  ✅ MISSION COMPLETE — Rover arrived at {TARGET}!")

    # ─────────────────────────────────────────────────────────────
    # Final summary
    # ─────────────────────────────────────────────────────────────
    wall_elapsed = time.perf_counter() - wall_start
    sim_time     = total_steps * DT   # total simulated seconds

    print(f"\n{'─'*62}")
    print("  MISSION SUMMARY")
    print(f"{'─'*62}")
    print(f"  Outcome               : {'SUCCESS ✅' if mission_success else 'FAILED ❌'}")
    print(f"  Total simulation steps: {total_steps:,}")
    print(f"  Total simulated time  : {sim_time:,.1f} s  ({sim_time/3600:.2f} h)")
    print(f"  Wall-clock time       : {wall_elapsed:.1f} s")
    print(f"  Terrain map cells     : {len(terrain_map.grid)}")
    print(f"  Drone recharge cycles : {drone.recharge_cycles}")
    print(f"  Scout stuck events    : {scout.stuck_count}")
    print(f"  Rover final position  : ({rover.x:.2f}, {rover.y:.2f})")
    print(f"  A* waypoints          : {len(optimal_path)}")
    print(f"  A* total cost         : {total_cost:.2f}")
    print(f"{'─'*62}\n")

    # ── Trajectory + path plots ──────────────────────────────────
    _plot_mission(
        terrain_map, optimal_path,
        drone_traj_fwd, drone_traj_ret,
        scout_traj_fwd, scout_traj_ret,
        map_csv = MAP_CSV,
    )


# ─── Plot ─────────────────────────────────────────────────────────────────────

def _plot_mission(terrain_map: TerrainMap,
                  optimal_path: list,
                  drone_traj_fwd: list,
                  drone_traj_ret: list,
                  scout_traj_fwd: list,
                  scout_traj_ret: list,
                  map_csv,
                  grid_size: int = 50):
    """
    Produce a 3-panel dark-theme figure:
      Left   — Drone trajectories (forward + return) on traversability background.
      Centre — Scout trajectories (forward zigzag + return zigzag).
      Right  — Rover A* path on traversability background.
    """
    # ── Load raw terrain traversability as background ─────────────
    try:
        df = pd.read_csv(map_csv)
        trav_grid = np.full((grid_size, grid_size), np.nan)
        for _, row in df.iterrows():
            xi, yi = int(row["x"]), int(row["y"])
            if 0 <= xi < grid_size and 0 <= yi < grid_size:
                trav_grid[yi, xi] = row["traversability"]
    except Exception:
        trav_grid = np.zeros((grid_size, grid_size))

    cmap_trav = LinearSegmentedColormap.from_list(
        "trav", ["#d62728", "#ff7f0e", "#2ca02c"])

    # ── Helper: draw a single axis ────────────────────────────────
    def _setup_ax(ax, title):
        ax.set_facecolor("#0d0d1a")
        im = ax.imshow(trav_grid, origin="lower", aspect="equal",
                       cmap=cmap_trav, vmin=0, vmax=1,
                       interpolation="nearest", alpha=0.55)
        ax.set_title(title, fontsize=11, color="white", pad=8)
        ax.tick_params(colors="gray", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")
        ax.set_xticks(np.arange(0, grid_size + 1, 5))
        ax.set_yticks(np.arange(0, grid_size + 1, 5))
        ax.set_xticklabels(range(0, grid_size + 1, 5), fontsize=6, color="gray")
        ax.set_yticklabels(range(0, grid_size + 1, 5), fontsize=6, color="gray")
        ax.grid(color="#2a2a3a", linewidth=0.3)
        ax.set_xlabel("x (cells)", color="gray", fontsize=8)
        ax.set_ylabel("y (cells)", color="gray", fontsize=8)
        return im

    def _mark_home_target(ax):
        ax.plot(*START,  marker="*", markersize=14, color="#f0e68c",
                zorder=10, label="Home",   linestyle="none")
        ax.plot(*TARGET, marker="D", markersize=9,  color="#00ffcc",
                zorder=10, label="Target", linestyle="none")

    def _traj_xy(traj):
        """Unzip list of (x,y) tuples into two arrays, down-sample for clarity."""
        if not traj:
            return np.array([]), np.array([])
        # Down-sample to at most 4000 points so the plot isn't too heavy
        step = max(1, len(traj) // 4000)
        xs = np.array([p[0] for p in traj[::step]])
        ys = np.array([p[1] for p in traj[::step]])
        return xs, ys

    # ── Build figure ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Three-Agent Mission — Trajectories & Optimal Path",
                 fontsize=14, fontweight="bold", color="white", y=1.01)

    # ── Panel 1: Drone ───────────────────────────────────────────
    ax1 = axes[0]
    _setup_ax(ax1, "🛰  Drone Trajectory")
    _mark_home_target(ax1)

    xs, ys = _traj_xy(drone_traj_fwd)
    if len(xs):
        ax1.plot(xs, ys, color="#4fc3f7", linewidth=1.0, alpha=0.85, label="Forward")
        ax1.plot(xs[0], ys[0], "o", color="#4fc3f7", markersize=5)

    xs, ys = _traj_xy(drone_traj_ret)
    if len(xs):
        ax1.plot(xs, ys, color="#ce93d8", linewidth=1.0, alpha=0.85,
                 linestyle="--", label="Return")
        ax1.plot(xs[-1], ys[-1], "o", color="#ce93d8", markersize=5)

    ax1.legend(fontsize=7, facecolor="#222", labelcolor="white",
               loc="upper left", framealpha=0.7)

    # ── Panel 2: Scout ───────────────────────────────────────────
    ax2 = axes[1]
    _setup_ax(ax2, "🚙  Scout Zigzag Trajectory")
    _mark_home_target(ax2)

    xs, ys = _traj_xy(scout_traj_fwd)
    if len(xs):
        ax2.plot(xs, ys, color="#ffb74d", linewidth=0.8, alpha=0.80, label="Forward")

    xs, ys = _traj_xy(scout_traj_ret)
    if len(xs):
        ax2.plot(xs, ys, color="#ef9a9a", linewidth=0.8, alpha=0.80,
                 linestyle="--", label="Return")

    # Mark scout stuck events
    stuck_coords = Scout.stuck_cells
    if stuck_coords:
        sx = [c[0] for c in stuck_coords]
        sy = [c[1] for c in stuck_coords]
        ax2.scatter(sx, sy, c="#e63946", s=12, zorder=8,
                    label=f"Stuck ({len(stuck_coords)})", alpha=0.7)

    ax2.legend(fontsize=7, facecolor="#222", labelcolor="white",
               loc="upper left", framealpha=0.7)

    # ── Panel 3: Rover A* path ────────────────────────────────────
    ax3 = axes[2]
    _setup_ax(ax3, "🚜  Rover — A* Optimal Path")
    _mark_home_target(ax3)

    if optimal_path:
        rx = [p[0] for p in optimal_path]
        ry = [p[1] for p in optimal_path]
        ax3.plot(rx, ry, color="#69f0ae", linewidth=2.0,
                 marker="o", markersize=3.5, zorder=9, label="A* path")
        # Colour each node by its traversability estimate
        for node in optimal_path:
            cell = terrain_map.grid.get(node)
            if cell and cell.traversability_estimate is not None:
                c = cmap_trav(cell.traversability_estimate)
                ax3.plot(node[0], node[1], "o", color=c,
                         markersize=6, zorder=10, alpha=0.9)

    ax3.legend(fontsize=7, facecolor="#222", labelcolor="white",
               loc="upper left", framealpha=0.7)

    # ── Shared colourbar (traversability) ────────────────────────
    sm = plt.cm.ScalarMappable(cmap=cmap_trav,
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, fraction=0.015, pad=0.02)
    cb.set_label("Terrain traversability  (0=blocked → 1=free)",
                 color="gray", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="gray", labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="gray")

    out_path = Path(__file__).resolve().parent / "governor_mission.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"  📊 Plot saved → {out_path}")
    plt.show()


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    governor()
