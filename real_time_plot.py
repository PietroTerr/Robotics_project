import numpy as np
from matplotlib import pyplot as plt, animation


class RealTimePlot:
    def __init__(self, terrain_map, agents):
        self.terrain_map = terrain_map
        self.agents = agents
        self.agents_path = {agent: [(agent.x, agent.y)] for agent in self.agents}
        self.grid_snapshots = []  # Store grid state at each step
        self.stuck_history = []   # Store stuck cells at each step

    def record(self):
        """Call this inside your while loop instead of plot()."""
        # Record agent positions
        for agent in self.agents:
            self.agents_path[agent].append((agent.x, agent.y))

        # Record grid snapshot
        self.grid_snapshots.append(self.__prepare_grid_view())

        # Record stuck cells
        stuck = [(x, y) for (x, y), cell in self.terrain_map.grid.items()
                 if getattr(cell, 'is_stuck', False)]
        self.stuck_history.append(stuck)

    def __prepare_grid_view(self):
        grid_view = np.full((self.terrain_map.height, self.terrain_map.width), np.nan)
        for (x, y), cell in self.terrain_map.grid.items():
            if cell.traversability_estimate is not None:
                grid_view[y, x] = cell.traversability_estimate
        return grid_view

    def plot_final(self):
        """Call this once after the while loop ends."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Exploration Map")

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgrey")

        # Show the final grid state
        img = ax.imshow(
            self.grid_snapshots[-1], origin="lower",
            cmap=cmap, vmin=0.0, vmax=1.0
        )
        fig.colorbar(img, ax=ax, shrink=0.8, label="Traversability Estimate")

        # Plot full paths
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.agents)))
        for i, agent in enumerate(self.agents):
            path = np.array(self.agents_path[agent])
            ax.plot(path[:, 0], path[:, 1], color=colors[i],
                    linewidth=1.5, label=f"{agent.__class__.__name__} Path")
            # Mark start and end
            ax.scatter(*path[0],  marker='o', color=colors[i], s=100, zorder=5)
            ax.scatter(*path[-1], marker='*', color=colors[i], s=200, zorder=5)

        # Plot all stuck cells (deduplicated)
        all_stuck = list({pt for step in self.stuck_history for pt in step})
        if all_stuck:
            sx, sy = zip(*all_stuck)
            ax.scatter(sx, sy, c='red', marker='X', s=100, label='Stuck Cell', zorder=4)

        ax.set_xlim(-0.5, self.terrain_map.width - 0.5)
        ax.set_ylim(-0.5, self.terrain_map.height - 0.5)
        ax.legend(loc='upper right')
        plt.show()

    def plot_animated(self):
        """Optional: replay the recorded run as an animation after the loop."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title("Exploration Replay")

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgrey")

        img = ax.imshow(self.grid_snapshots[0], origin="lower",
                        cmap=cmap, vmin=0.0, vmax=1.0)
        fig.colorbar(img, ax=ax, shrink=0.8, label="Traversability Estimate")

        colors = plt.cm.tab10(np.linspace(0, 1, len(self.agents)))
        path_lines = {}
        for i, agent in enumerate(self.agents):
            line, = ax.plot([], [], color=colors[i], linewidth=1.5,
                            label=f"{agent.__class__.__name__} Path")
            path_lines[agent] = line

        initial_positions = np.array([self.agents_path[agent][0] for agent in self.agents])
        robot_scatter = ax.scatter(initial_positions[:, 0], initial_positions[:, 1],
                                   c=colors, s=80, edgecolors='black', zorder=5)
        stuck_scatter = ax.scatter([], [], c='red', marker='X', s=100,
                                   label='Stuck Cell', zorder=4)

        ax.set_xlim(-0.5, self.terrain_map.width - 0.5)
        ax.set_ylim(-0.5, self.terrain_map.height - 0.5)
        ax.legend(loc='upper right')

        def update(frame):
            img.set_data(self.grid_snapshots[frame])

            positions = []
            for agent in self.agents:
                path = np.array(self.agents_path[agent][:frame + 1])
                path_lines[agent].set_data(path[:, 0], path[:, 1])
                positions.append(self.agents_path[agent][frame])
            robot_scatter.set_offsets(np.array(positions))

            stuck = self.stuck_history[frame]
            if stuck:
                stuck_scatter.set_offsets(np.array(stuck))
            else:
                stuck_scatter.set_offsets(np.empty((0, 2)))

            return [img, robot_scatter, stuck_scatter] + list(path_lines.values())

        ani = animation.FuncAnimation(fig, update, frames=len(self.grid_snapshots),
                                      interval=100, blit=True, repeat=False)
        plt.show()