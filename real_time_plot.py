import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from itertools import cycle


class MapPlotter:

    _PALETTE = ['cyan', 'magenta', 'yellow', 'lime', 'orange', 'white', 'pink']
    _UNOBSERVED_RGB = np.array([0.5, 0.5, 0.5])
    _STUCK_RGB      = np.array([0.0, 0.0, 0.0])
    _OBSERVED_BLEND = 0.5   # how much the real color shows through on observed-only cells

    def __init__(self, grid_size: int = 50, capture_every: int = 3,live: bool = False):
        self.grid_size     = grid_size
        self.capture_every = capture_every
        self.live          = live
        self._tick         = 0

        if not self.live:
            # Lightweight mode: only track agent positions
            self._path_log: list[tuple] = []   # (step, robot_id, x, y)
            return

        # ── Live mode only below this point ──────────────────────────────────
        self._frames = []

        self._color_pool  = cycle(self._PALETTE)
        self._agent_color = {}   # robot_id -> color

        self._visited: set[tuple[int, int]] = set()
        self._observed: set[tuple[int, int]] = set()

        # Per-agent drawing state (lazy init)
        self._agent_lines   = {}   # robot_id -> Line2D  (full path)
        self._agent_heads   = {}   # robot_id -> Line2D  (current position dot)
        self._agent_history = {}   # robot_id -> {'x': [], 'y': []}

        # ── Figure setup ──────────────────────────────────────────────────────
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-0.5, grid_size - 0.5)
        self.ax.set_ylim(-0.5, grid_size - 0.5)
        self.ax.set_title("Robot Exploration Map")
        self.ax.set_aspect("equal")

        # Colormaps
        self._trav_cmap  = mcolors.LinearSegmentedColormap.from_list("RdGr", ["red", "green"])
        self._stuck_cmap = mcolors.LinearSegmentedColormap.from_list("OrRd", ["orange", "red"])

        # Grid image
        self._map_rgb = np.full((grid_size, grid_size, 3), 0.5)
        self._img = self.ax.imshow(
            self._map_rgb, origin="lower",
            extent=[-0.5, grid_size - 0.5, -0.5, grid_size - 0.5],
            interpolation="nearest",
        )

        # Stuck-probability scatter
        self._scatter = self.ax.scatter(
            [], [], marker="x", s=40, c=[],
            cmap=self._stuck_cmap, vmin=0.5, vmax=1.0, zorder=3,
        )

        # Frontier lines (boundary between visited and observed)
        self._frontier_lc = LineCollection([], colors="red", linewidths=1.5, alpha=0.8, zorder=2)
        self.ax.add_collection(self._frontier_lc)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, cells_dict: dict, agent_states: list):
        """
        cells_dict   : {(x, y): cell_object}
        agent_states : list of AgentState  (uses .agent.robot_id, .agent.x, .agent.y)
        """
        self._tick += 1

        if not self.live:
            for state in agent_states:
                self._path_log.append((
                    self._tick,
                    state.agent.robot_id,
                    state.agent.x,
                    state.agent.y,
                ))
            return

        self._update_grid(cells_dict)
        self._update_frontier()
        self._update_agents(agent_states)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        if self._tick % self.capture_every == 0:
            self.fig.canvas.draw()
            rgba = np.frombuffer(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
            w, h = self.fig.canvas.get_width_height(physical=True)
            self._frames.append(rgba.reshape(h, w, 4)[..., :3].copy())

    def save(self, path: str = "simulation.mp4", fps: int = 15):
        if not self.live:
            csv_path = path.rsplit(".", 1)[0] + "_paths.csv"
            with open(csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "robot_id", "x", "y"])
                writer.writerows(self._path_log)
            print(f"Saved {len(self._path_log)} rows → {csv_path}")
            return

        # ── Live mode: save video ─────────────────────────────────────────────
        if not self._frames:
            print("No frames captured — nothing to save.")
            return

        from matplotlib.animation import FuncAnimation, FFMpegWriter

        save_fig, save_ax = plt.subplots(figsize=self.fig.get_size_inches())
        save_ax.axis("off")
        save_ax.set_position([0, 0, 1, 1])
        im = save_ax.imshow(self._frames[0])

        def _frame(i):
            im.set_data(self._frames[i])
            return (im,)

        anim = FuncAnimation(
            save_fig, _frame,
            frames=len(self._frames),
            interval=1000 / fps,
            blit=True,
        )

        try:
            anim.save(path, writer=FFMpegWriter(fps=fps))
            print(f"Saved {len(self._frames)} frames → {path}")
        except Exception:
            gif_path = path.rsplit(".", 1)[0] + ".gif"
            anim.save(gif_path, writer="pillow", fps=fps)
            print(f"FFmpeg unavailable — saved as GIF → {gif_path}")

        plt.close(save_fig)

    # ── Private helpers (live mode only) ──────────────────────────────────────

    def _trav_color(self, value: float) -> np.ndarray:
        return np.array(self._trav_cmap(value)[:3])

    def _get_color(self, robot_id: str) -> str:
        if robot_id not in self._agent_color:
            self._agent_color[robot_id] = next(self._color_pool)
        return self._agent_color[robot_id]

    def _init_agent(self, robot_id: str):
        color = self._get_color(robot_id)
        line, = self.ax.plot([], [], color=color, linewidth=1.5,
                             label=robot_id, alpha=0.8, zorder=4)
        head, = self.ax.plot([], [], color=color, marker="o",
                             markersize=8, markeredgecolor="white", zorder=5)
        self._agent_lines[robot_id]   = line
        self._agent_heads[robot_id]   = head
        self._agent_history[robot_id] = {"x": [], "y": []}
        self.ax.legend(loc="upper right", fontsize="small")

    def _update_grid(self, cells_dict: dict):
        scatter_x, scatter_y, scatter_c = [], [], []
        new_visited:  set[tuple[int, int]] = set()
        new_observed: set[tuple[int, int]] = set()

        for (x, y), cell in cells_dict.items():
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                continue

            if cell.is_visited:
                new_visited.add((x, y))
                color = (self._STUCK_RGB if cell.is_stuck
                         else self._trav_color(cell.real_traversability or 0.5))
                self._map_rgb[y, x] = color

            elif cell.is_observed:
                new_observed.add((x, y))
                est  = cell.traversability_estimate or 0.5
                full = self._trav_color(est)
                self._map_rgb[y, x] = (self._UNOBSERVED_RGB * (1 - self._OBSERVED_BLEND)
                                       + full * self._OBSERVED_BLEND)
                if cell.stuck_probability_estimate > 0.5:
                    scatter_x.append(x)
                    scatter_y.append(y)
                    scatter_c.append(cell.stuck_probability_estimate)
            else:
                self._map_rgb[y, x] = self._UNOBSERVED_RGB

        self._visited  = new_visited
        self._observed = new_observed
        self._img.set_data(self._map_rgb)

        if scatter_x:
            self._scatter.set_offsets(np.c_[scatter_x, scatter_y])
            self._scatter.set_array(np.array(scatter_c))
        else:
            self._scatter.set_offsets(np.empty((0, 2)))
            self._scatter.set_array(np.array([]))

    def _update_frontier(self):
        """Draw a red edge wherever a visited cell borders an observed-only cell."""
        _BORDERS = [
            ( 1,  0,  0.5, -0.5,  0.5,  0.5),
            (-1,  0, -0.5, -0.5, -0.5,  0.5),
            ( 0,  1, -0.5,  0.5,  0.5,  0.5),
            ( 0, -1, -0.5, -0.5,  0.5, -0.5),
        ]
        segments = []
        for vx, vy in self._visited:
            for dx, dy, x1, y1, x2, y2 in _BORDERS:
                nb = (vx + dx, vy + dy)
                if nb in self._observed and nb not in self._visited:
                    segments.append([(vx + x1, vy + y1), (vx + x2, vy + y2)])
        self._frontier_lc.set_segments(segments)

    def _update_agents(self, agent_states: list):
        for state in agent_states:
            rid = state.agent.robot_id
            x, y = state.agent.x, state.agent.y

            if rid not in self._agent_lines:
                self._init_agent(rid)

            hist = self._agent_history[rid]
            hist["x"].append(x)
            hist["y"].append(y)
            self._agent_lines[rid].set_data(hist["x"], hist["y"])
            self._agent_heads[rid].set_data([x], [y])