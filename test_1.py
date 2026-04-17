"""
Integration Test Script for Terrain Mapping and Path Planning.

This script tests the integration between the robotic perception agents (`motion.py`) 
and the spatial memory components (`data_management.py`). It simulates a selected 
robot (Drone, Scout, or Rover) exploring the terrain using the `MapAPI`.

Crucially, this script demonstrates how the accumulated `TerrainMap` data is used
to build a Directed Weighted Graph (via NetworkX) for path planning. The generated 
graph models reachable nodes and traversable edges, ensuring that subsequent pathfinding 
algorithms can avoid obstacles and areas with a high stuck probability.

"""
import time
from src.map_api import MapAPI
from motion import Scout, Drone, Rover
from data_management import TerrainMap, build_weighted_graph

def main():

    print("Loading MapAPI...")
    csv_path = "src/map_001_seed42.csv"
    map_api = MapAPI(terrain=csv_path, rng_seed=42)
    
    print("Initializing TerrainMap (from data_management.py)...")
    terrain_map = TerrainMap(width=50, height=50)
    
    case = "drone"  #  "scout" or "rover" 
    start_pos = (2.0, 2.0)
    target_pos = (45.0, 45.0)
    
    print(f"Initializing {case.capitalize()} (from motion.py) at {start_pos}...")
    if case == "scout":
        robot = Scout(map_api, "scout_integration_01", start_pos)
    elif case == "drone":
        robot = Drone(map_api, "drone_integration_01", start_pos)
    elif case == "rover":
        robot = Rover(map_api, "rover_integration_01", start_pos)
    else:
        raise ValueError("Invalid case specified")
    
    dt = 0.1 
    
    print(f"Starting {case.capitalize()} movement towards {target_pos}...")
    
    step = 0
    while True:
        step += 1
        
        observations = robot.perceive()
        
        for obs in observations:
            cell = terrain_map.get_cell(obs.x, obs.y)
            
            cell.texture = obs.features.get("texture", cell.texture)
            cell.color = obs.features.get("color", cell.color)
            cell.slope = obs.features.get("slope", cell.slope)
            cell.uphill_angle = obs.features.get("uphill_angle", cell.uphill_angle)
            
            if cell.texture is not None:
                cell.traversability_estimate = max(0.1, cell.texture)
                cell.confidence = 0.9
                
            cell.observed_by.add(case)
            cell.visit_count += 1
            cell.last_updated = step
            
        if case == "scout":
            reached, was_stuck, stuck_pos = robot.step_towards(target_pos[0], target_pos[1], dt)
            if was_stuck and stuck_pos:
                stuck_cell = terrain_map.get_cell(stuck_pos[0], stuck_pos[1])
                stuck_cell.stuck_probability_estimate = 1.0
                stuck_cell.traversability_estimate = 0.001
            
            if reached:
                print(f"-> Target reached at step {step}!")
                break
                
        elif case == "drone":
            reached, forced_recharge = robot.step_towards(target_pos[0], target_pos[1], dt)
            if reached:
                print(f"-> Target reached at step {step}!")
                break
                
        elif case == "rover":
            reached, was_stuck = robot.step_towards(target_pos[0], target_pos[1], dt)
            if was_stuck:
                stuck_cell = terrain_map.get_cell(robot.x, robot.y)
                stuck_cell.stuck_probability_estimate = 1.0
                stuck_cell.traversability_estimate = 0.001
                print(f"-> Rover experienced a FATAL STUCK event at step {step}. Exploration aborted!")
                break
            
            if reached:
                print(f"-> Target reached at step {step}!")
                break

    print("\n--- RESULTS ---")
    print(f"{case.capitalize()} ended at position: ({robot.x:.2f}, {robot.y:.2f})")
    print(f"TerrainMap has gathered {len(terrain_map.grid)} unique mapped cells.")
    
    print("\nBuilding Directed Weighted Graph for Path Planning...")
    graph = build_weighted_graph(terrain_map)
    print(f"Graph generated with {graph.number_of_nodes()} reachable nodes and {graph.number_of_edges()} traversable edges.")
    
    if graph.number_of_edges() > 0:
        print("Success! `data_management.py` and `motion.py` are properly integrated and fully usable together.")

    print("\nGenerating visual plots...")
    import numpy as np
    import matplotlib.pyplot as plt
    import networkx as nx
    
    grid_view = np.full((terrain_map.height, terrain_map.width), np.nan)
    for (x, y), cell in terrain_map.grid.items():
        if 0 <= x < terrain_map.width and 0 <= y < terrain_map.height:
            if cell.stuck_probability_estimate > 0.5:
                grid_view[y, x] = -0.1
            else:
                grid_view[y, x] = cell.traversability_estimate if cell.traversability_estimate is not None else np.nan                
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle("Layered Grid Map Integration & Path Planning Graph")
    
    ax1 = axes[0]
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgrey")
    cmap.set_under(color="red")
    im1 = ax1.imshow(grid_view, origin="lower", cmap=cmap, vmin=0, vmax=1.0)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label="Traversability / Stuck (Red)")
    
    ax1.plot(start_pos[0], start_pos[1], 'go', markersize=10, label="Start")
    ax1.plot(target_pos[0], target_pos[1], 'r*', markersize=12, label="Target")
    ax1.plot(robot.x, robot.y, 'cx', markersize=8, label="Final Pos")
    ax1.set_title("Layered Grid Map: Layer 2 Estimates")
    ax1.legend()
    
    ax2 = axes[1]
    
    ax2.imshow(grid_view, origin="lower", cmap=cmap, vmin=0, vmax=1.0, alpha=0.3)
    
    pos = {node: (node[0], node[1]) for node in graph.nodes()}
    if len(pos) > 0:
        nx.draw(graph, pos, ax=ax2, node_size=10, node_color="blue", alpha=0.6, edge_color="gray", width=0.5)
        
    ax2.plot(start_pos[0], start_pos[1], 'go', markersize=10, label="Start")
    ax2.plot(target_pos[0], target_pos[1], 'r*', markersize=12, label="Target")
    
    ax2.set_xlim(-0.5, terrain_map.width - 0.5)
    ax2.set_ylim(-0.5, terrain_map.height - 0.5)
    ax2.set_title("Generated NetworkX Edge Connectivity")
    ax2.legend()
    
    plt.tight_layout()
    try:
        plt.show()
    except Exception as e:
        print(f"Could not show plot interactively: {e}")

if __name__ == "__main__":
    main()
