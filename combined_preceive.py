"""
Multi-Agent Cooperative Terrain Exploration and Mapping.

This script coordinates the movement and perception of two robotic agents: a Drone and a Scout.
- The Drone flies directly towards the target, mapping terrain features.
- The Scout follows a dynamic zigzag path to maximize exploration coverage, and
  performs local exploration maneuvers whenever the Drone pauses to recharge.
Perception data from both agents is combined into a shared TerrainMap to estimate
traversability, with the final results plotted on a 2D grid view.

"""
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from src.map_api import MapAPI
from motion import Scout, Drone
from data_management import TerrainMap, build_weighted_graph

def main():
    print("Loading MapAPI & Components...")
    csv_path = "src/map_001_seed42.csv"
    map_api = MapAPI(terrain=csv_path, rng_seed=42)
    terrain_map = TerrainMap(width=50, height=50)
    
    start_pos = (5.0, 5.0)
    target_pos = (45.0, 45.0)
    
    print("Initializing Drone & Scout...")
    drone = Drone(map_api, "drone_01", start_pos)
    scout = Scout(map_api, "scout_01", start_pos)
    
    # dt MUST align with internal map_api timestep (0.1) so Drone battery draws properly!
    dt = 0.1  
    
    zigzag_width = 25.0
    zigzag_period = 5.0
    
    drone_path_x, drone_path_y = [drone.x], [drone.y]
    scout_path_x, scout_path_y = [scout.x], [scout.y]
    
    explore_target = None
    
    print(f"Starting journey towards {target_pos}...")
    
    step = 0
    drone_finished = False
    
    # Calculate geometric trajectory for projection
    dx_path = target_pos[0] - start_pos[0]
    dy_path = target_pos[1] - start_pos[1]
    path_len = math.hypot(dx_path, dy_path)
    ux_path = dx_path / path_len
    uy_path = dy_path / path_len
    lx_path = -uy_path
    ly_path = ux_path

    while True:
        step += 1
        
        # --- PERCEIVE ---
        d_obs = drone.perceive()
        s_obs = scout.perceive()
        
        # --- UPDATE MAP ---
        for agent_id, observations in [("drone", d_obs), ("scout", s_obs)]:
            for obs in observations:
                cell = terrain_map.get_cell(obs.x, obs.y)
                cell.texture = obs.features.get("texture", cell.texture)
                cell.color = obs.features.get("color", cell.color)
                cell.slope = obs.features.get("slope", cell.slope)
                cell.uphill_angle = obs.features.get("uphill_angle", cell.uphill_angle)
                
                if cell.texture is not None:
                    cell.traversability_estimate = max(0.1, cell.texture)
                    cell.confidence = 0.9
                    
                cell.observed_by.add(agent_id)
                cell.visit_count += 1
                cell.last_updated = step
                
        # --- MOVE DRONE ---
        drone_reached, drone_recharging = drone.step_towards(target_pos[0], target_pos[1], dt)
        if drone_reached and not drone_finished:
            print(f"-> Drone reached the target at step {step}! Scout is continuing to catch up...")
            drone_finished = True
        
        # --- MOVE SCOUT ---
        if drone_recharging:
            # Random exploration around the drone's resting location
            if explore_target is None or math.hypot(explore_target[0]-scout.x, explore_target[1]-scout.y) < 0.5:
                ang = random.uniform(0, 2*math.pi)
                rad = random.uniform(1.0, 5.0)
                explore_target = (drone.x + rad*math.cos(ang), drone.y + rad*math.sin(ang))
                explore_target = (
                    max(0.0, min(49.0, explore_target[0])),
                    max(0.0, min(49.0, explore_target[1]))
                )
            
            s_reached, s_stuck, stuck_cell = scout.step_towards(explore_target[0], explore_target[1], dt)
        else:
            explore_target = None
            
            # Project scout onto the ideal path to find where the zigzag should center
            proj_dist = (scout.x - start_pos[0])*ux_path + (scout.y - start_pos[1])*uy_path
            
            if proj_dist >= path_len - 1.0:
                s_reached, s_stuck, stuck_cell = scout.step_towards(target_pos[0], target_pos[1], dt)
                if s_reached:
                    print(f"-> Scout reached the target at step {step}! Simulation complete.")
                    break
            else:
                lookahead = 1.0
                center_x = start_pos[0] + (proj_dist + lookahead) * ux_path
                center_y = start_pos[1] + (proj_dist + lookahead) * uy_path
                
                wave_offset = math.sin((scout.total_time_spent / zigzag_period) * 2 * math.pi) * zigzag_width
                
                scout_target_x = center_x + lx_path * wave_offset
                scout_target_y = center_y + ly_path * wave_offset
                
                _, s_stuck, stuck_cell = scout.step_towards(scout_target_x, scout_target_y, dt)

        # Record stuck events
        if s_stuck and stuck_cell:
            cell = terrain_map.get_cell(*stuck_cell)
            cell.stuck_probability_estimate = 1.0
            cell.traversability_estimate = 0.001
            
        # Logging traces for plot
        if step % 5 == 0:  # Sample less dense for rendering
            drone_path_x.append(drone.x)
            drone_path_y.append(drone.y)
            scout_path_x.append(scout.x)
            scout_path_y.append(scout.y)

    # --- PLOTTING ---
    print("\nGenerating Orchestration Plots...")
    
    grid_view = np.full((terrain_map.height, terrain_map.width), np.nan)
    for (x, y), cell in terrain_map.grid.items():
        if 0 <= x < terrain_map.width and 0 <= y < terrain_map.height:
            if cell.stuck_probability_estimate > 0.5:
                grid_view[y, x] = -0.1
            else:
                grid_view[y, x] = cell.traversability_estimate if cell.traversability_estimate is not None else np.nan
                
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 10))
    fig.suptitle("Drone (Straight) & Scout (ZigZag/Explore) Orchestration")
    
    cmap = plt.cm.viridis.copy() 
    cmap.set_bad(color="lightgrey") 
    cmap.set_under(color="red")
    im1 = ax1.imshow(grid_view, origin="lower", cmap=cmap, vmin=0, vmax=1.0)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Traversability / Stuck (Red)")
    
    ax1.plot(drone_path_x, drone_path_y, 'w--', linewidth=2, label="Drone Path")
    ax1.plot(scout_path_x, scout_path_y, 'c-', linewidth=1.5, alpha=0.8, label="Scout Path")
    
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label="Start")
    ax1.plot(target_pos[0], target_pos[1], 'r*', markersize=12, label="Target")

    ax1.set_xlim(-0.5, terrain_map.width - 0.5)
    ax1.set_ylim(-0.5, terrain_map.height - 0.5)
    ax1.legend()
    
    try:
        plt.show()
    except Exception as e:
        print(f"Plot error: {e}")

if __name__ == "__main__":
    main()
