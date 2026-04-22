import math
import sys
from pathlib import Path
import networkx as nx

# Setup percorsi
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.map_api import MapAPI
from data_management import TerrainMap, build_weighted_graph

def run_simulation_completa():
    # 1. Inizializzazione API (Modifica time_step qui)
    csv_path = PROJECT_ROOT / "src" / "map_001_seed42.csv"
    map_api = MapAPI(terrain=csv_path, rng_seed=42, time_step=0.2) # <--- TIME_STEP
    
    terrain_map = TerrainMap(width=50, height=50)
    
    # Registrazione Robot
    drone_id = "drone_1"
    scout_id = "scout_1"
    rover_id = "rover_1"
    map_api.register_robot(robot_id=drone_id, robot_type="drone")
    map_api.register_robot(robot_id=scout_id, robot_type="scout")
    map_api.register_robot(robot_id=rover_id, robot_type="rover")

    # Posizioni iniziali
    drone_pos = (5, 5)
    scout_pos = (5, 5)
    rover_pos = (5, 5)
    
    print("--- 🛰️ FASE 1: ESPLORAZIONE COMBINATA (DRONE + SCOUT) ---")
    
    # Simuliamo 5 step di esplorazione per raccogliere dati
    for i in range(5):
        # A) Il Drone vola e percepisce aree vaste (dati visivi)
        drone_pos = (drone_pos[0] + 2, drone_pos[1] + 2)
        obs_drone = map_api.perceive(robot_id=drone_id, position=drone_pos)
        terrain_map.store_observation(obs_drone)
        
        # B) Lo Scout avanza e testa fisicamente il terreno (dati reali/stuck)
        scout_vel_command = 1.0
        step_res = map_api.step(robot_id=scout_id, position=scout_pos, 
                                command_velocity=scout_vel_command, command_orientation=0.0)
        
        # Salviamo i dati reali nella mappa
        cell = terrain_map.get_cell(scout_pos[0], scout_pos[1])
        cell.is_stuck = step_res.is_stuck
        cell.real_traversability = max(0.01, step_res.actual_velocity / scout_vel_command)
        
        scout_pos = (scout_pos[0] + 1, scout_pos[1])
        print(f"Step {i+1}: Scout a {scout_pos}, Bloccato: {step_res.is_stuck}")

    # 2. Aggiornamento Stime (Layer 2)
    print("\n--- 🧠 FASE 2: ELABORAZIONE DATI E PREDIZIONE ---")
    terrain_map.refresh_estimation() # Propaga la conoscenza dello scout sui dati del drone

    # 3. Pianificazione Percorso Rover
    print("\n--- 🗺️ FASE 3: CALCOLO PERCORSO SICURO PER IL ROVER ---")
    target_rover = (15, 10) # Punto di destinazione
    
    # Costruiamo il grafo pesato che include le penalità per gli stuck events
    G = build_weighted_graph(terrain_map)
    
    try:
        # A* o Dijkstra per trovare il percorso che minimizza i costi (evitando i pericoli)
        percorso_sicuro = nx.shortest_path(G, source=rover_pos, target=target_rover, weight='weight')
        print(f"Percorso trovato! Il Rover dovrà attraversare {len(percorso_sicuro)} celle.")
        
        # 4. Moto del Rover lungo il percorso sicuro
        print("\n--- 🚜 FASE 4: MOVIMENTO DEL ROVER ---")
        for next_node in percorso_sicuro[1:]:
            # Calcoliamo l'orientamento necessario per raggiungere il prossimo nodo
            dx = next_node[0] - rover_pos[0]
            dy = next_node[1] - rover_pos[1]
            angle = math.atan2(dy, dx)
            
            # Eseguiamo lo step del Rover
            res = map_api.step(robot_id=rover_id, position=rover_pos, 
                               command_velocity=1.0, command_orientation=angle)
            
            print(f"  Rover si sposta verso {next_node}. Velocità reale: {res.actual_velocity:.2f}, Stuck: {res.is_stuck}")
            
            if res.is_stuck:
                print("  ⚠️ ATTENZIONE: Il Rover si è bloccato nonostante la pianificazione!")
                break
                
            rover_pos = next_node
            
        if rover_pos == target_rover:
            print(f"\n✅ Missione Compiuta! Rover arrivato a {target_rover}")
            
    except nx.NetworkXNoPath:
        print("❌ Impossibile trovare un percorso sicuro verso il target scelto.")

if __name__ == "__main__":
    run_simulation_completa()