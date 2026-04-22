class CellData:
    # --- Layer 1: Raw observations from agents ---
    texture: float
    color: float
    slope: float
    uphill_angle: float

    is_stuck: bool | None = None
    real_traversability: float | None = None

    # --- Layer 2: Derived estimates (output of predictive model) ---
    traversability_estimate: float | None = None
    stuck_probability_estimate: float = 0.0
    confidence: float = 0.0

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y


    def __hash__(self):
        return hash((self.x, self.y))

    def set_texture(self, texture: float):
        self.texture = texture

    def set_color(self, color: float):
        self.color = color

    def set_slope(self, slope: float):
        self.slope = slope

    def set_uphill_angle(self, uphill_angle: float):
        self.uphill_angle = uphill_angle