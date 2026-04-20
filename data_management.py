from dataclasses import dataclass, field

from src.map_api_core import TerrainObservation

@dataclass
class CellData:
    # --- Layer 1: Raw observations from agents ---
    texture: float | None = None
    color: float | None = None
    slope: float | None = None
    uphill_angle: float | None = None
    
    is_stuck: bool | None = None
    real_traversability: float | None = None  # <--- NUOVO: traversabilità reale calcolata passandoci

    # --- Layer 2: Derived estimates (output of predictive model) ---
    traversability_estimate: float | None = None
    stuck_probability_estimate: float = 0.0
    confidence: float = 0.0

    def set_texture(self, texture: float):
        if self.texture is None:
            self.texture = texture
    def set_color(self, color: float):
        if self.color is None:
            self.color = color
    def set_slope(self, slope: float):
        if self.slope is None:
            self.slope = slope
    def set_uphill_angle(self, uphill_angle: float):
        if self.uphill_angle is None:
            self.uphill_angle = uphill_angle


class TerrainMap:
    """
    A sparse dictionary-based mapping system that houses coordinate-mapped `CellData` objects.
    Useful for partial explorations without maintaining large static memory overheads.
    """
    def __init__(self, width: int = 50, height: int = 50):
        self.grid: dict[tuple[int, int], CellData] = {}
        self.width = width
        self.height = height
        self.grid_size = (self.width, self.height)

    def get_cell(self, x: int, y: int) -> CellData:
        """Helper to get a cell, automatically generating it if it doesn't already exist."""
        coords = (int(x), int(y))
        if coords not in self.grid:
            self.grid[coords] = CellData()
        return self.grid[coords]

    def store_observation(self, obs: list[TerrainObservation]):
        if obs is not None:
            for ob in obs:
                cell = self.get_cell(ob.x, ob.y)
                cell.set_texture(ob.features.get("texture"))
                cell.set_color(ob.features.get("color"))
                cell.set_slope(ob.features.get("slope"))
                cell.set_uphill_angle(ob.features.get("uphill_angle"))

    def __store_stuck_information(self, x: int, y: int, get_stuck):
        cell = self.get_cell(x,y)
        if get_stuck is not None:
            cell.is_stuck = get_stuck


    def refresh_estimation(self, movement_information=None):
        """
        Aggiorna le stime di attraversabilità e probabilità di blocco usando
        l'Inverse Distance Weighting (IDW) nello spazio delle features (texture, color).
        """
        known_cells = []
        unvisited_cells = []

        # 1. Separiamo le celle esplorate fisicamente da quelle solo percepite
        for coords, cell in self.grid.items():
            if cell.texture is not None and cell.color is not None:
                # Se abbiamo la traversabilità reale, il robot ci è passato
                if cell.real_traversability is not None:
                    known_cells.append(cell)
                else:
                    unvisited_cells.append(cell)

        # Se lo scout non ha ancora attraversato nessuna cella, non possiamo stimare nulla
        if not known_cells:
            return

        # 2. Stimiamo i valori per le celle non esplorate
        for u_cell in unvisited_cells:
            total_weight = 0.0
            weighted_trav_sum = 0.0
            weighted_stuck_sum = 0.0
            min_dist = float('inf')

            for k_cell in known_cells:
                # Distanza Euclidea nello spazio (texture, color)
                # Più texture e colore sono simili, più la distanza si avvicina a 0
                feature_dist = math.sqrt(
                    (u_cell.texture - k_cell.texture)**2 + 
                    (u_cell.color - k_cell.color)**2
                )
                
                # Calcolo del peso (epsilon per evitare divisioni per zero)
                weight = 1.0 / (feature_dist + 1e-5)
                
                # --- Stima Attraversabilità Continua ---
                weighted_trav_sum += k_cell.real_traversability * weight
                
                # --- Stima Eventi di Blocco (Booleani) ---
                stuck_val = 1.0 if k_cell.is_stuck else 0.0
                weighted_stuck_sum += stuck_val * weight

                total_weight += weight
                if feature_dist < min_dist:
                    min_dist = feature_dist

            # 3. Aggiorniamo le stime del Layer 2 della cella
            if total_weight > 0:
                u_cell.traversability_estimate = weighted_trav_sum / total_weight
                u_cell.stuck_probability_estimate = weighted_stuck_sum / total_weight
                u_cell.confidence = 1.0 / (1.0 + min_dist)