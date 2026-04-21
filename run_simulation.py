from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.map_api import MapAPI
from data_management import TerrainMap, build_weighted_graph

def run_combined_exploration():
    # 1. Inizializzazione API e Mappa
    csv_path = PROJECT_ROOT / "src" / "map_001_seed42.csv"
    map_api = MapAPI(terrain=csv_path, rng_seed=42, time_step=0.2)
    terrain_map = TerrainMap(width=50, height=50)
    
    # Registriamo ENTRAMBI i robot
    drone_id = "drone_1"
    scout_id = "scout_1"
    map_api.register_robot(robot_id=drone_id, robot_type="drone")
    map_api.register_robot(robot_id=scout_id, robot_type="scout")
    
    # Posizioni iniziali
    drone_pos = (5, 5)
    scout_pos = (5, 5)
    
    target_pos = (45, 45)
    print(f"--- INIZIO ESPLORAZIONE COMBINATA → Target: {target_pos} ---")
    
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))
    
    def step_toward(pos, target, step_size):
        """Move pos toward target by at most step_size per axis."""
        dx = target[0] - pos[0]
        dy = target[1] - pos[1]
        # Normalise to step_size
        dist = (dx**2 + dy**2) ** 0.5
        if dist == 0:
            return pos
        scale = min(step_size, dist) / dist
        nx = clamp(int(round(pos[0] + dx * scale)), 0, terrain_map.width - 1)
        ny = clamp(int(round(pos[1] + dy * scale)), 0, terrain_map.height - 1)
        return (nx, ny)

    import math
    drone_done = False
    scout_done = False
    step = 0
    MAX_STEPS = 200  # safety cap

    # 2. Ciclo di simulazione dinamico verso il target
    while not (drone_done and scout_done) and step < MAX_STEPS:
        step += 1
        print(f"\n🔄 STEP {step}")

        # ==========================================
        # AZIONI DEL DRONE (Vola veloce, 3 celle per step)
        # ==========================================
        if not drone_done:
            next_drone_pos = step_toward(drone_pos, target_pos, step_size=3)

            # Percezione dell'area
            drone_obs = map_api.perceive(robot_id=drone_id, position=next_drone_pos)
            terrain_map.store_observation(drone_obs)

            # Calcola orientamento verso il target
            ddx = next_drone_pos[0] - drone_pos[0]
            ddy = next_drone_pos[1] - drone_pos[1]
            d_orientation = math.atan2(ddy, ddx) if (ddx != 0 or ddy != 0) else 0.0

            map_api.step(robot_id=drone_id, position=drone_pos,
                         command_velocity=2.0, command_orientation=d_orientation)
            drone_pos = next_drone_pos

            if drone_pos == target_pos:
                drone_done = True
                print(f"  🚁 Drone ha RAGGIUNTO il target {target_pos}! Scansionate {len(drone_obs)} celle.")
            else:
                print(f"  🚁 Drone volato a {drone_pos} → {target_pos}. Scansionate {len(drone_obs)} celle.")

        # ==========================================
        # AZIONI DELLO SCOUT (Testa il terreno, 1 cella per step)
        # ==========================================
        if not scout_done:
            next_scout_pos = step_toward(scout_pos, target_pos, step_size=1)
            scout_vel = 1.0

            # Percezione della cella corrente
            scout_obs = map_api.perceive(robot_id=scout_id, position=scout_pos)
            terrain_map.store_observation(scout_obs)

            # Calcola orientamento verso il target
            sdx = next_scout_pos[0] - scout_pos[0]
            sdy = next_scout_pos[1] - scout_pos[1]
            s_orientation = math.atan2(sdy, sdx) if (sdx != 0 or sdy != 0) else 0.0

            # Movimento fisico
            step_result = map_api.step(
                robot_id=scout_id, position=scout_pos,
                command_velocity=scout_vel, command_orientation=s_orientation
            )

            # Ground Truth SOLO per lo scout
            scout_cell = terrain_map.get_cell(scout_pos[0], scout_pos[1])
            scout_cell.is_stuck = step_result.is_stuck
            real_trav = max(0.01, step_result.actual_velocity / scout_vel)
            scout_cell.real_traversability = real_trav

            scout_pos = next_scout_pos

            if scout_pos == target_pos:
                scout_done = True
                print(f"  🤖 Scout ha RAGGIUNTO il target {target_pos}! Trav. reale: {real_trav:.2f}")
            else:
                print(f"  🤖 Scout avanzato a {scout_pos} → {target_pos}. Traversabilità reale: {real_trav:.2f}")

    if step >= MAX_STEPS:
        print(f"\n⚠️  Simulazione interrotta al limite massimo di {MAX_STEPS} step.")
    else:
        print(f"\n✅ Esplorazione completata in {step} step.")


    # ==========================================
    # 3. FUSIONE DEI DATI (Magia Predittiva)
    # ==========================================
    print("\n--- 🧠 AVVIO PREDIZIONE ---")
    # Qui l'algoritmo prende i pochi dati fisici dello Scout e li mappa
    # su tutte le decine di celle viste dall'alto dal Drone!
    terrain_map.refresh_estimation()
    
    # Analisi del target dopo l'esplorazione
    target_cell = terrain_map.get_cell(*target_pos)
    
    print(f"Analisi della cella target {target_pos}:")
    if target_cell.texture is not None:
        print(f"  > Traversabilità stimata: {target_cell.traversability_estimate:.2f}")
        print(f"  > Probabilità blocco: {target_cell.stuck_probability_estimate:.2f}")
        print(f"  > Confidenza: {target_cell.confidence:.2f}")
    else:
        print("  > Cella fuori dal raggio visivo del drone (target non raggiunto).")

    # ==========================================
    # 4. COSTRUZIONE DEL GRAFO DI PIANIFICAZIONE
    # ==========================================
    print("\n--- COSTRUZIONE DELLA MAPPA (GRAFO) ---")
    graph = build_weighted_graph(terrain_map)
    print(f"Grafo generato per il Path Planning NetworkX:")
    print(f"  > Nodi esplorati/stimati: {graph.number_of_nodes()}")
    print(f"  > Archi percorribili: {graph.number_of_edges()}")
    
    # Calcolo del percorso ottimale Start → Target
    import networkx as nx
    start_node = (5, 5)
    end_node = target_pos
    if start_node in graph and end_node in graph:
        try:
            path = nx.shortest_path(graph, source=start_node, target=end_node, weight='weight')
            path_cost = nx.shortest_path_length(graph, source=start_node, target=end_node, weight='weight')
            print(f"  > Percorso ottimale Start→{end_node}: {len(path)} passi, costo={path_cost:.2f}")
        except nx.NetworkXNoPath:
            print(f"  > Nessun percorso connesso trovato tra Start e {end_node}.")
    else:
        missing = [n for n in [start_node, end_node] if n not in graph]
        print(f"  > Impossibile calcolare il percorso. Nodi mancanti nel grafo: {missing}")

    # ==========================================
    # 5. VISUALIZZAZIONE DELLA MAPPA (COME map_printing.py)
    # ==========================================
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import LinearSegmentedColormap
    
    print("\n--- GENERAZIONE PLOT DELLA MAPPA ---")
    GRID_SIZE = terrain_map.width
    
    # Helper per estrarre la griglia
    def extract_grid(attr_name, default_val=np.nan):
        grid = np.full((GRID_SIZE, GRID_SIZE), default_val)
        for (x, y), cell in terrain_map.grid.items():
            val = getattr(cell, attr_name)
            if val is not None:
                grid[y, x] = val
        return grid

    traversability = extract_grid("traversability_estimate")
    slope          = extract_grid("slope")
    uphill_angle   = extract_grid("uphill_angle")
    stuck_event    = extract_grid("stuck_probability_estimate")
    texture        = extract_grid("texture")
    color_val      = extract_grid("color")

    # Figure layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Robot Navigation Map (Estimated) — 50 × 50 m", fontsize=16, fontweight="bold", y=0.98)
    fig.patch.set_facecolor("#1a1a2e")

    LABEL_KW = dict(fontsize=10, color="white", pad=8)

    def styled_imshow(ax, data, cmap, title, vmin=None, vmax=None, cbar_label=""):
        ax.set_facecolor("#0d0d1a")
        # Copy and prep cmap to handle NaNs explicitly
        current_cmap = plt.get_cmap(cmap).copy()
        current_cmap.set_bad(color="#0d0d1a") 
        
        im = ax.imshow(
            data, origin="lower", aspect="equal",
            cmap=current_cmap, vmin=vmin, vmax=vmax,
            interpolation="nearest"
        )
        ax.set_title(title, **LABEL_KW)
        ax.tick_params(colors="gray", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.ax.yaxis.set_tick_params(color="gray", labelsize=8)
        cb.set_label(cbar_label, color="gray", fontsize=8)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color="gray")
        return im

    # 1 - Traversability Estimate
    cmap_trav = LinearSegmentedColormap.from_list("trav", ["#d62728", "#ff7f0e", "#2ca02c"])
    styled_imshow(axes[0, 0], traversability, cmap_trav,
                  "① Traversability Estimate", 0, 1, "0 = blocked  →  1 = free")

    # 2 - Slope
    styled_imshow(axes[0, 1], slope, "plasma", "② Slope (°)", cbar_label="degrees")

    # 3 - Uphill angle
    styled_imshow(axes[0, 2], uphill_angle, "twilight", "③ Uphill Heading Angle (rad)", cbar_label="radians")

    # 4 - Stuck events (binary overlay)
    ax_stuck = axes[1, 0]
    ax_stuck.set_facecolor("#0d0d1a")
    
    cmap_stuck_base = plt.get_cmap("Greys").copy()
    cmap_stuck_base.set_bad(color="#0d0d1a")
    
    # Fill base with traversability 
    ax_stuck.imshow(np.where(np.isnan(traversability), 0, traversability), origin="lower", aspect="equal",
                    cmap=cmap_stuck_base, alpha=0.3, interpolation="nearest")
                    
    stuck_overlay = np.where(stuck_event > 0.5, 1.0, np.nan)
    stuck_cmap = LinearSegmentedColormap.from_list("stuck", ["#e63946", "#e63946"])
    stuck_cmap.set_bad(color=(0,0,0,0))
    ax_stuck.imshow(stuck_overlay, origin="lower", aspect="equal",
                    cmap=stuck_cmap, alpha=0.9, interpolation="nearest")
    
    ax_stuck.set_title("④ Stuck Risk Estimate > 50%", **LABEL_KW)
    ax_stuck.tick_params(colors="gray", labelsize=8)
    patch = mpatches.Patch(color="#e63946", label="stuck risk")
    ax_stuck.legend(handles=[patch], loc="lower right", fontsize=8, facecolor="#222", labelcolor="white")
    for spine in ax_stuck.spines.values():
        spine.set_edgecolor("#444")

    # 5 - Texture
    styled_imshow(axes[1, 1], texture, "YlOrBr", "⑤ Texture", cbar_label="roughness")

    # 6 - Colour value
    styled_imshow(axes[1, 2], color_val, "viridis", "⑥ Colour Value", cbar_label="normalised")

    # Grid overlay
    for ax in axes.flat:
        ax.set_xticks(np.arange(-0.5, GRID_SIZE, 5), minor=False)
        ax.set_yticks(np.arange(-0.5, GRID_SIZE, 5), minor=False)
        ax.set_xticklabels(range(0, GRID_SIZE + 1, 5), fontsize=7, color="gray")
        ax.set_yticklabels(range(0, GRID_SIZE + 1, 5), fontsize=7, color="gray")
        ax.grid(which="major", color="#333", linewidth=0.4)
        ax.set_xlabel("x (m)", color="gray", fontsize=8)
        ax.set_ylabel("y (m)", color="gray", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save image
    out_path = PROJECT_ROOT / "run_simulation_map.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  > Grafico salvato come {out_path}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"Non è stato possibile visualizzare il grafico in modo interattivo: {e}")

if __name__ == '__main__':
    run_combined_exploration()
