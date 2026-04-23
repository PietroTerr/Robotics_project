import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors


class MapPlotter:
    def __init__(self, grid_size: int = 50):
        self.grid_size = grid_size

        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(-0.5, grid_size - 0.5)
        self.ax.set_ylim(-0.5, grid_size - 0.5)
        self.ax.set_title("Robot Exploration Map")
        self.ax.set_aspect('equal')

        # Grid Background (Gray default)
        self.map_rgb = np.full((grid_size, grid_size, 3), 0.5)
        self.img = self.ax.imshow(
            self.map_rgb,
            origin='lower',
            extent=[-0.5, grid_size - 0.5, -0.5, grid_size - 0.5],
            interpolation='nearest'
        )

        # 'X' Marks: Only > 0.5 probability, Orange to Red
        self.stuck_cmap = mcolors.LinearSegmentedColormap.from_list("OrRd", ["orange", "red"])
        self.scatter = self.ax.scatter([], [], marker='x', s=40, c=[], cmap=self.stuck_cmap, vmin=0.5, vmax=1.0)

        self.frontier_lines = LineCollection([], colors='red', linewidths=2)
        self.ax.add_collection(self.frontier_lines)

        self.agent_colors = ['cyan', 'magenta', 'yellow']
        self.agent_names = ['rover', 'scout', 'drone']

        # Dictionaries to track lines and current-position markers
        self.agent_lines = {}
        self.agent_heads = {}  # Fix 14: New dictionary for current positions
        self.agent_history = {}

        # Traversability Colormap: Red (0.0) -> Green (1.0)
        self.trav_cmap = mcolors.LinearSegmentedColormap.from_list("RdGr", ["red", "green"])

    def update(self, cells_dict: dict, agents_list: list):
        scatter_x, scatter_y, scatter_c = [], [], []
        visited_cells, observed_cells = set(), set()

        # 1. Update Grid Cells & Scatter
        for (x, y), cell in cells_dict.items():
            if not (0 <= x < self.grid_size and 0 <= y < self.grid_size):
                continue

            if cell.is_visited:
                visited_cells.add((x, y))
                if cell.is_stuck:
                    self.map_rgb[y, x] = [0.0, 0.0, 0.0]  # Black
                else:
                    trav = cell.real_traversability if cell.real_traversability is not None else 0.5
                    self.map_rgb[y, x] = self.trav_cmap(trav)[:3]

            elif cell.is_observed:
                observed_cells.add((x, y))
                est_trav = cell.traversability_estimate if cell.traversability_estimate is not None else 0.5
                self.map_rgb[y, x] = self.trav_cmap(est_trav)[:3]

                # Plot X ONLY if probability > 50%
                if cell.stuck_probability_estimate > 0.5:
                    scatter_x.append(x)
                    scatter_y.append(y)
                    scatter_c.append(cell.stuck_probability_estimate)
            else:
                self.map_rgb[y, x] = [0.5, 0.5, 0.5]  # Unobserved is Gray

        self.img.set_data(self.map_rgb)

        if scatter_x:
            self.scatter.set_offsets(np.c_[scatter_x, scatter_y])
            self.scatter.set_array(np.array(scatter_c))
        else:
            self.scatter.set_offsets(np.empty((0, 2)))
            # Fix 13: Reset the color array to prevent colormap dimension mismatches
            self.scatter.set_array(np.array([]))

            # 2. Frontier Lines
        segments = []
        directions = [(1, 0, 0.5, -0.5, 0.5, 0.5), (-1, 0, -0.5, -0.5, -0.5, 0.5),
                      (0, 1, -0.5, 0.5, 0.5, 0.5), (0, -1, -0.5, -0.5, 0.5, -0.5)]

        for vx, vy in visited_cells:
            for dx, dy, x1, y1, x2, y2 in directions:
                nx, ny = vx + dx, vy + dy
                if (nx, ny) in observed_cells and (nx, ny) not in visited_cells:
                    segments.append([(vx + x1, vy + y1), (vx + x2, vy + y2)])
        self.frontier_lines.set_segments(segments)

        # 3. Agent Paths and Heads
        for i, (ax_pos, ay_pos) in enumerate(agents_list):
            agent_id = self.agent_names[i]
            color = self.agent_colors[i]

            if agent_id not in self.agent_lines:
                # Initialize the path line
                line, = self.ax.plot([], [], color=color, linewidth=2, label=agent_id)
                self.agent_lines[agent_id] = line

                # Fix 14: Initialize the current position marker (a solid dot with a white edge)
                head, = self.ax.plot([], [], color=color, marker='o', markersize=8, markeredgecolor='white')
                self.agent_heads[agent_id] = head

                self.agent_history[agent_id] = {'x': [], 'y': []}
                self.ax.legend(loc='upper right', fontsize='small')

            # Update history line
            self.agent_history[agent_id]['x'].append(ax_pos)
            self.agent_history[agent_id]['y'].append(ay_pos)
            self.agent_lines[agent_id].set_data(
                self.agent_history[agent_id]['x'], self.agent_history[agent_id]['y']
            )

            # Fix 14: Update current position marker
            self.agent_heads[agent_id].set_data([ax_pos], [ay_pos])

        # Fix 12: Explicitly flag the canvas for a redraw and flush GUI events
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()