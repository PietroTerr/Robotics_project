import queue
import threading

from Governor import Governor
from TerrainMap import TerrainMap
from motion import Drone, Scout, Rover
from real_time_plot import MapPlotter
from src.map_api import MapAPI


def simulation(data_queue):

    map_api = get_map_api()
    start_pos = (5,5)

    drone = Drone(map_api, "drone_01", start_pos)
    scout = Scout(map_api, "scout_01", start_pos)
    rover = Rover(map_api, "rover_01", start_pos)

    terrain_map = TerrainMap()
    governor = Governor(terrain_map,rover,scout,drone)

    while True:
        # -- Get heading for each agent
        (rover_heading, scout_heading, drone_heading) = governor.get_heading()
        if rover_heading is None:
            data_queue.put(None)  # Signal: simulation done
            print("Simulation Complete.")
            
            # --- EVALUATION --------------------------------------------------
            from TerrainPredictorEvaluator import TerrainPredictorEvaluator
            visited = terrain_map.get_visited_cells()
            if len(visited) >= 5:
                print("\n[EVALUATION] Generando il report del TerrainPredictor a fine missione...")
                evaluator = TerrainPredictorEvaluator(terrain_map.terrain_predictor, visited)
                report = evaluator.full_report(k_folds=5)
                print(report.summary())
                
                try:
                    evaluator.plot_diagnostics()
                except ImportError as exc:
                    print(f"[WARN] Impossibile mostrare i grafici: {exc}")
            else:
                print("\n[EVALUATION] Non ci sono abbastanza celle esplorate per valutare il modello.")
            # -----------------------------------------------------------------
            
            return

        observations = {}
        observations.update(scout.perceive())
        observations.update(drone.perceive())

        movement_information = {}
        step_rover_result = rover.step_towards(rover_heading)
        movement_information[rover.x, rover.y] = step_rover_result
        step_scout_result = scout.step_towards(scout_heading)
        movement_information[scout.x, scout.y] = step_scout_result
        drone.step_towards(drone_heading)
        terrain_map.update_map(observations, movement_information)

        agents_positions = [
            (rover.x, rover.y),
            (scout.x, scout.y),
            (drone.x, drone.y)
        ]
        # 2. Package the snapshot using your method!
        snapshot = {
            "grid": terrain_map.get_grid_snapshot(),
            "agents": agents_positions
        }

        # 3. Put it in the queue so the GUI can render it
        data_queue.put(snapshot)
        print(agents_positions)

def get_map_api():
    print("Loading MapAPI & Components...")
    csv_path = "src/map_001_seed42.csv"
    map_api = MapAPI(terrain=csv_path, rng_seed=42)
    return map_api


import threading
import queue
import matplotlib.pyplot as plt


def main():
    # maxsize=1 prevents the simulation from running too far ahead of the plot
    data_queue = queue.Queue(maxsize=1)

    # 1. Start the simulation in the background
    thread = threading.Thread(target=simulation, args=(data_queue,), daemon=True)
    thread.start()

    # 2. Initialize the plotter in the main thread
    plotter = MapPlotter(grid_size=50)

    # 3. The GUI Loop
    while True:
        try:
            # Wait up to 100ms for a new step from the simulation
            snapshot = data_queue.get(timeout=0.1)

            # If the simulation thread sends 'None', it's finished
            if snapshot is None:
                print("Simulation Complete.")
                break

            # Note: Ensure snapshot["grid"] is a dict mapping (x,y) -> CellData
            plotter.update(snapshot["grid"], snapshot["agents"])

        except queue.Empty:
            # No new data yet, just pass and let the GUI refresh
            pass

        # This yields control to matplotlib to draw the frame.
        # IT PREVENTS THE WHITE SCREEN.
        plt.pause(0.05)

        # Keep the final plot open when done
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
