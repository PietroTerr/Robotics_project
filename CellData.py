from dataclasses import dataclass


@dataclass
class CellData:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
        # ── Layer 1: Raw observations ─────────────────────────────────────────
        self.texture: float | None = None
        self.color: float | None = None
        self.slope: float | None = None
        self.uphill_angle: float | None = None

        self.is_stuck: bool | None = None
        self.real_traversability: float | None = None

        # ── Layer 2: Derived estimates (output of TerrainPredictor) ───────────
        self.traversability_estimate: float = 5.0
        self.confidence: float = 0.0
        self.stuck_probability_estimate: float = 5.0

        # ── Status flags ──────────────────────────────────────────────────────
        self.is_observed: bool = False
        self.is_visited: bool = False

    # ── Setters (auto-update status flags) ────────────────────────────────────

    def set_texture(self, texture: float | None) -> None:
        self.texture = texture
        self._refresh_observed()

    def set_color(self, color: float | None) -> None:
        self.color = color
        self._refresh_observed()

    def set_slope(self, slope: float | None) -> None:
        self.slope = slope
        self._refresh_observed()

    def set_uphill_angle(self, uphill_angle: float | None) -> None:
        self.uphill_angle = uphill_angle
        self._refresh_observed()

    def set_is_stuck(self, is_stuck: bool) -> None:
        self.is_stuck = is_stuck
        self._refresh_visited()
    def set_real_traversability(self, real_traversability: float | None) -> None:
        self.real_traversability = real_traversability
        self._refresh_visited()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _refresh_observed(self) -> None:
        """Mark cell as observed once all four sensor fields are populated."""
        if all(
                v is not None
                for v in (self.texture, self.color, self.slope, self.uphill_angle)
        ):
            self.is_observed = True
            self._refresh_visited()

    def _refresh_visited(self) -> None:
        """
        Mark cell as visited once the agent has physically traversed it and
        produced ground-truth traversability data.

        Call this from TerrainMap after writing real_traversability and is_stuck.
        """
        if (
                self.is_observed
                and self.is_stuck is not None
                and self.real_traversability is not None
        ):
            self.is_visited = True