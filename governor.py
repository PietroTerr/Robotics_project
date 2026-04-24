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
from matplotlib.colors import LinearSegmentedColormap
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.map_api import MapAPI
# Nuove importazioni modulari
from TerrainMap import TerrainMap
from motion import Drone, Scout, Rover

# ─── Parametri Missione ───────────────────────────────────────────────────────
START      = (5, 5)
TARGET     = (30, 30)
DT         = 0.2
ZIG_WIDTH  = 10
ZIG_LOOKAHEAD = 8.0
MAX_STEPS  = float('inf')
MAP_CSV    = PROJECT_ROOT / "src" / "map_001_seed1.csv"

# ─── Geometry Helpers ─────────────────────────────────────────────────────────

def _axis_perp(p_from, p_to):
    dx, dy = p_to[0] - p_from[0], p_to[1] - p_from[1]
    L = math.hypot(dx, dy)
    if L < 1e-9: return (1.0, 0.0), (0.0, 1.0)
    return (dx/L, dy/L), (-dy/L, dx/L)

def _axial_projection(point, origin, axis):
    return (point[0] - origin[0]) * axis[0] + (point[1] - origin[1]) * axis[1]

# ─── Phase Runner ─────────────────────────────────────────────────────────────

def _run_phase(name, drone, scout, goal, origin, terrain_map, prefer_unexplored=False, zig_width=ZIG_WIDTH, zig_lookahead=ZIG_LOOKAHEAD):
    axis, perp = _axis_perp(origin, goal)
    goal_proj = _axial_projection(goal, origin, axis)   # Total axial distance to goal
    FINAL_APPROACH_DIST = zig_lookahead * 2             # Scout switches to direct nav inside this
    print(f"\n{'='*60}\n  {name}\n{'='*60}")

    drone_done = scout_done = False
    scout_side, scout_wp, step = 1, None, 0
    drone_traj, scout_traj = [], []

    while not (drone_done and scout_done) and step < MAX_STEPS:
        drone_traj.append((drone.x, drone.y))
        scout_traj.append((scout.x, scout.y))

        # ── Drone ─────────────────────────────────────────────────────
        if not drone_done:
            tx, ty = float(goal[0]), float(goal[1])
            if prefer_unexplored:
                d_ang = math.atan2(goal[1] - drone.y, goal[0] - drone.x)
                best_unobs = None
                best_dist = float('inf')
                for d_dist in range(4, 15, 2):
                    for ang_off in [-0.6, 0.0, 0.6]:
                        cx = int(drone.x + d_dist * math.cos(d_ang + ang_off))
                        cy = int(drone.y + d_dist * math.sin(d_ang + ang_off))
                        if 0 <= cx < 50 and 0 <= cy < 50:
                            if (cx, cy) not in terrain_map.grid or not terrain_map.grid[(cx, cy)].is_observed:
                                if d_dist < best_dist:
                                    best_dist = d_dist
                                    best_unobs = (cx, cy)
                if best_unobs:
                    tx, ty = float(best_unobs[0]), float(best_unobs[1])

            drone.step_towards(tx, ty, DT)

            # Ingestione Osservazioni Drone
            obs = drone.perceive()
            obs_dict = {(int(o.x), int(o.y)): o.features for o in obs}
            terrain_map.update_map(obs_dict, {})  # Solo osservazioni visive

            # ✅ Tighter threshold + snap to exact cell centre
            if math.hypot(drone.x - goal[0], drone.y - goal[1]) < 0.5:
                drone.x, drone.y = float(goal[0]), float(goal[1])
                drone_done = True
                print(f"  ✔ Drone arrived at exact goal {goal}")

        # ── Scout ─────────────────────────────────────────────────────
        if not scout_done:
            scout_proj = _axial_projection((scout.x, scout.y), origin, axis)
            remaining_to_goal = goal_proj - scout_proj

            # ✅ Final-approach mode: drive straight to goal when close
            if remaining_to_goal <= FINAL_APPROACH_DIST:
                scout_wp = (float(goal[0]), float(goal[1]))
            else:
                # Normal zigzag waypoint
                base_proj = min(scout_proj + zig_lookahead, goal_proj)
                if scout_wp is None or math.hypot(scout.x - scout_wp[0], scout.y - scout_wp[1]) < 1.5:
                    if prefer_unexplored:
                        left_unobs = 0
                        right_unobs = 0
                        for d_dist in range(2, int(zig_lookahead * 1.5) + 1):
                            for w in range(1, int(zig_width)):
                                lx, ly = int(scout.x + d_dist*axis[0] - w*perp[0]), int(scout.y + d_dist*axis[1] - w*perp[1])
                                rx, ry = int(scout.x + d_dist*axis[0] + w*perp[0]), int(scout.y + d_dist*axis[1] + w*perp[1])
                                if 0 <= lx < 50 and 0 <= ly < 50 and ((lx, ly) not in terrain_map.grid or not terrain_map.grid[(lx, ly)].is_observed):
                                    left_unobs += 1
                                if 0 <= rx < 50 and 0 <= ry < 50 and ((rx, ry) not in terrain_map.grid or not terrain_map.grid[(rx, ry)].is_observed):
                                    right_unobs += 1
                        if left_unobs > right_unobs:
                            scout_side = -1
                        elif right_unobs > left_unobs:
                            scout_side = 1
                        else:
                            scout_side = -scout_side
                    else:
                        scout_side = -scout_side
                    
                    scout_wp = (
                        origin[0] + base_proj * axis[0] + scout_side * zig_width * perp[0],
                        origin[1] + base_proj * axis[1] + scout_side * zig_width * perp[1],
                    )

            pre_x, pre_y = scout.x, scout.y
            orientation = math.atan2(scout_wp[1] - scout.y, scout_wp[0] - scout.x)

            _, was_stuck, _ = scout.step_towards(scout_wp[0], scout_wp[1], DT)

            # Calcolo Telemetria per Ground Truth
            actual_dist = math.hypot(scout.x - pre_x, scout.y - pre_y)
            actual_vel = actual_dist / DT

            # Ingestione Dati Movimento + Percezione Scout
            obs = scout.perceive()
            obs_dict = {(int(o.x), int(o.y)): o.features for o in obs}
            movement_dict = {
                (int(round(scout.x)), int(round(scout.y))): {
                    "is_stuck": was_stuck,
                    "heading": orientation,
                    "command_velocity": scout.speed,
                    "actual_velocity": actual_vel,
                }
            }
            terrain_map.update_map(obs_dict, movement_dict)  # Ingestione completa

            # ✅ Tighter threshold + snap to exact cell centre
            if math.hypot(scout.x - goal[0], scout.y - goal[1]) < 0.5:
                scout.x, scout.y = float(goal[0]), float(goal[1])
                scout_done = True
                print(f"  ✔ Scout arrived at exact goal {goal}")

        step += 1
        if step % 2000 == 0:
            print(f"  Step {step}: Mappa copre {len(terrain_map.grid)} celle.")

    return step, drone_traj, scout_traj, step * DT  # also return simulated seconds


def _run_target_sweep(name, scout, target, terrain_map, size=8):
    print(f"\n{'='*60}\n  {name}\n{'='*60}")
    
    # Generate lawnmower pattern around target
    min_x = max(0, int(target[0]) - size // 2)
    max_x = min(49, int(target[0]) + size // 2 - 1)
    min_y = max(0, int(target[1]) - size // 2)
    max_y = min(49, int(target[1]) + size // 2 - 1)

    waypoints = []
    go_up = True
    for x in range(min_x, max_x + 1):
        y_range = range(min_y, max_y + 1) if go_up else range(max_y, min_y - 1, -1)
        for y in y_range:
            waypoints.append((float(x), float(y)))
        go_up = not go_up

    scout_traj = []
    step = 0
    scout_wp_idx = 0
    
    # We want to visit all waypoints
    while scout_wp_idx < len(waypoints) and step < MAX_STEPS:
        scout_traj.append((scout.x, scout.y))
        wp = waypoints[scout_wp_idx]
        
        # Check if reached waypoint
        if math.hypot(scout.x - wp[0], scout.y - wp[1]) < 0.5:
            scout.x, scout.y = float(wp[0]), float(wp[1])
            scout_wp_idx += 1
            if scout_wp_idx >= len(waypoints):
                break
            wp = waypoints[scout_wp_idx]

        pre_x, pre_y = scout.x, scout.y
        orientation = math.atan2(wp[1] - scout.y, wp[0] - scout.x)
        
        _, was_stuck, _ = scout.step_towards(wp[0], wp[1], DT)
        
        actual_dist = math.hypot(scout.x - pre_x, scout.y - pre_y)
        actual_vel = actual_dist / DT
        
        obs = scout.perceive()
        obs_dict = {(int(o.x), int(o.y)): o.features for o in obs}
        movement_dict = {
            (int(round(scout.x)), int(round(scout.y))): {
                "is_stuck": was_stuck,
                "heading": orientation,
                "command_velocity": scout.speed,
                "actual_velocity": actual_vel,
            }
        }
        terrain_map.update_map(obs_dict, movement_dict)

        step += 1
        if step % 2000 == 0:
            print(f"  Sweep Step {step}: Mappa copre {len(terrain_map.grid)} celle.")

    return step, scout_traj, step * DT


# ─── Governor Main ────────────────────────────────────────────────────────────

def governor():
    map_api = MapAPI(terrain=MAP_CSV, rng_seed=42, time_step=DT)
    terrain_map = TerrainMap(width=50, height=50)

    drone = Drone(map_api, "drone_1", START)
    scout = Scout(map_api, "scout_1", START)
    rover = Rover(map_api, "rover_1", START)

    wall_start = time.perf_counter()

    # Fase 1: Esplorazione
    steps1, drone_fwd, scout_fwd, sim1 = _run_phase(
        "PHASE 1: FORWARD", drone, scout, TARGET, START, terrain_map, prefer_unexplored=False, zig_width=ZIG_WIDTH, zig_lookahead=4.0)
        
    # Fase 1.5: Target Sweep
    steps15, scout_sweep, sim15 = _run_target_sweep("PHASE 1.5: 8x8 TARGET SWEEP", scout, TARGET, terrain_map, size=8)
    scout_fwd.extend(scout_sweep) # append sweep to fwd traj for plotting

    steps2, drone_ret, scout_ret, sim2 = _run_phase(
        "PHASE 2: RETURN",  drone, scout, START,  TARGET, terrain_map, True)

    # Fase 3: Path Planning
    print("\n--- [PHASE 3] PATH PLANNING ---")
    observed = terrain_map.get_observed_cells()
    terrain_map.terrain_predictor.update_prediction(observed)

    # 1. Otteniamo il dizionario delle adiacenze dal TerrainGraph
    G_dict = terrain_map.terrain_graph.get_graph("rover")
    
    # 2. Lo convertiamo in un vero grafo NetworkX
    G = nx.DiGraph(G_dict)
    
    rover_start = (int(round(rover.x)), int(round(rover.y)))
    
    # 3. Stitching: Se START o TARGET mancano, li forziamo agganciandoli ai vicini
    for node in [rover_start, TARGET]:
        if node not in G:
            print(f"  ⚠ Nodo {node} non esplorato. Lo aggancio alla mappa...")
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: continue
                    nb = (node[0] + dx, node[1] + dy)
                    if nb in G:
                        G.add_edge(node, nb, weight=1.0)
                        G.add_edge(nb, node, weight=1.0)

    # 4. Eseguiamo l'algoritmo A*
    try:
        path = nx.astar_path(G, rover_start, TARGET, 
                             heuristic=lambda a, b: math.hypot(a[0]-b[0], a[1]-b[1]),
                             weight="weight")
        print(f"  ✅ Percorso A* trovato: {len(path)} nodi.")
    except Exception as e:
        print(f"  ❌ Errore Pathfinding: {e}")
        path = []

    # Fase 4: Traversal Rover
    print("\n--- [PHASE 4] ROVER TRAVERSAL ---")
    rover_stuck = False
    if path:
        for wp in path[1:]:
            if rover_stuck:
                break
        wp_steps = 0
        while math.hypot(rover.x - wp[0], rover.y - wp[1]) >= 0.15 and wp_steps < 5000:
            _, r_stuck = rover.step_towards(wp[0], wp[1], DT)
            if r_stuck:
                print(f"  ⚠️ Rover bloccato a {wp}!")
                rover_stuck = True
                break
            wp_steps += 1
        # ✅ Snap rover to exact waypoint centre after each leg
        if not rover_stuck:
            rover.x, rover.y = float(wp[0]), float(wp[1])

    # ── Final position report ────────────────────────────────────────────────
    elapsed = time.perf_counter() - wall_start
    sim_total = sim1 + sim15 + sim2 + len(path) * 5000 * DT  # rough rover sim-time upper bound
    rover_sim_time = rover.total_time_spent

    print(f"\n{'='*60}")
    print(f"  MISSION LOG")
    print(f"  {'─'*54}")
    print(f"  Phase 1 — steps: {steps1:>6}   sim time: {sim1:>8.1f} s")
    print(f"  Phase 1.5 — steps: {steps15:>4}   sim time: {sim15:>8.1f} s")
    print(f"  Phase 2 — steps: {steps2:>6}   sim time: {sim2:>8.1f} s")
    print(f"  Rover traversal sim time:         {rover_sim_time:>8.1f} s")
    print(f"  Total simulated time:             {sim1+sim15+sim2+rover_sim_time:>8.1f} s")
    print(f"  Wall-clock time:                  {elapsed:>8.1f} s")
    print(f"  {'─'*54}")
    print(f"  Posizioni finali:")
    print(f"    Drone  → ({drone.x:.4f}, {drone.y:.4f})")
    print(f"    Scout  → ({scout.x:.4f}, {scout.y:.4f})")
    print(f"    Rover  → ({rover.x:.4f}, {rover.y:.4f})")
    if path and not rover_stuck:
        print(f"  ✅ Tutti e tre gli agenti hanno raggiunto il centro esatto di {TARGET}")
    elif path:
        print(f"  ⚠️  Rover bloccato — non ha raggiunto {TARGET}")
    else:
        print(f"  ⚠️  Rover non ha percorso — Pathfinding fallito")
    print(f"  Celle esplorate totali: {len(terrain_map.grid)}")
    print(f"  Stuck events scout: {len(scout.stuck_cells)}")
    print(f"  Nodi A* path: {len(path)}")
    print(f"{'='*60}")

    _plot_mission(
        terrain_map, path, scout.stuck_cells,
        drone_fwd, drone_ret, scout_fwd, scout_ret,
    )

# ─── Plotting ────────────────────────────────────────────────────────────────

# Custom RdYlGn-like colormap identical to reference image
_TRAV_CMAP = LinearSegmentedColormap.from_list(
    "trav",
    [(0.0, "#8B0000"), (0.35, "#CC4400"), (0.55, "#AAAA00"), (0.75, "#228B22"), (1.0, "#006400")],
)
_BG = "#1a1e2e"   # dark navy background matching the reference


def _load_terrain_img():
    df = pd.read_csv(MAP_CSV)
    img = np.zeros((50, 50))
    for _, r in df.iterrows():
        img[int(r["y"]), int(r["x"])] = r["traversability"]
    return img


def _style_ax(ax, title):
    """Apply dark-theme styling and grid to an axes."""
    ax.set_facecolor(_BG)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel("x (cells)", color="#aaaaaa", fontsize=9)
    ax.set_ylabel("y (cells)", color="#aaaaaa", fontsize=9)
    ax.tick_params(colors="#aaaaaa", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444455")
    ax.grid(color="#33334455", linewidth=0.4, linestyle="-")
    ax.set_xlim(-0.5, 49.5)
    ax.set_ylim(-0.5, 49.5)
    ax.set_aspect("equal")


def _draw_bg(ax, img):
    ax.imshow(img, origin="lower", cmap=_TRAV_CMAP, vmin=0, vmax=1,
              extent=(-0.5, 49.5, -0.5, 49.5), aspect="auto", alpha=0.85)


def _draw_markers(ax, home, target):
    ax.plot(*home,   marker="*", markersize=14, color="#FFD700",
            zorder=6, label="Home",   linestyle="none")
    ax.plot(*target, marker="D", markersize=10, color="#00CED1",
            zorder=6, label="Target", linestyle="none",
            markeredgecolor="white", markeredgewidth=0.6)


def _legend(ax):
    leg = ax.legend(facecolor="#22263a", edgecolor="#555566",
                    labelcolor="white", fontsize=8, loc="upper left",
                    framealpha=0.85)


def _plot_mission(terrain_map, path, scout_stuck,
                  drone_fwd, drone_ret, scout_fwd, scout_ret):
    img = _load_terrain_img()

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor(_BG)
    fig.suptitle("Three-Agent Mission — Trajectories & Optimal Path",
                 color="white", fontsize=14, fontweight="bold", y=1.01)

    # ── Panel 1: Drone ────────────────────────────────────────────────────────
    ax = axes[0]
    _draw_bg(ax, img)
    _style_ax(ax, "✈  Drone Trajectory")
    _draw_markers(ax, START, TARGET)

    if drone_fwd:
        dx, dy = zip(*drone_fwd)
        ax.plot(dx, dy, color="#00BFFF", linewidth=1.4, linestyle="-",  label="Forward")
    if drone_ret:
        dx, dy = zip(*drone_ret)
        ax.plot(dx, dy, color="#87CEEB", linewidth=1.0, linestyle="--", label="Return")
    _legend(ax)

    # ── Panel 2: Scout ────────────────────────────────────────────────────────
    ax = axes[1]
    _draw_bg(ax, img)
    _style_ax(ax, "Scout Zigzag Trajectory")
    _draw_markers(ax, START, TARGET)

    if scout_fwd:
        sx, sy = zip(*scout_fwd)
        ax.plot(sx, sy, color="#DAA520", linewidth=1.2, linestyle="-",  label="Forward")
    if scout_ret:
        sx, sy = zip(*scout_ret)
        ax.plot(sx, sy, color="#F0E68C", linewidth=0.8, linestyle="--", label="Return")
    if scout_stuck:
        ex, ey = zip(*scout_stuck)
        ax.scatter(ex, ey, c="#FF4444", marker="o", s=18, zorder=5,
                   label=f"Stuck ({len(scout_stuck)})")
    _legend(ax)

    # ── Panel 3: Rover ────────────────────────────────────────────────────────
    ax = axes[2]
    im = ax.imshow(img, origin="lower", cmap=_TRAV_CMAP, vmin=0, vmax=1,
                   extent=(-0.5, 49.5, -0.5, 49.5), aspect="auto", alpha=0.85)
    _style_ax(ax, "Rover — A* Optimal Path")
    _draw_markers(ax, START, TARGET)

    if path:
        px, py = zip(*path)
        ax.plot(px, py, color="#00CED1", linewidth=1.5, linestyle="--",
                marker="o", markersize=4, markerfacecolor="#FF4444",
                markeredgewidth=0, zorder=5, label="A* path")
    _legend(ax)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Terrain traversability  (0=blocked, 1=free)",
                   color="white", fontsize=8, rotation=270, labelpad=14)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    plt.tight_layout()
    out_path = PROJECT_ROOT / "governor_mission_output.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    print(f"\n  📊 Plot salvato in: {out_path}")
    plt.show()

if __name__ == "__main__":
    governor()