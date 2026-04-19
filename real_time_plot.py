import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class RealTimePlot:
    def __init__(self, terrain_map, agents):
        self.terrain_map = terrain_map
        self.agents = agents

        # Store paths
        self.agents_path = {agent: [(agent.x, agent.y)] for agent in self.agents}

        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Prepare grid
        self.grid_view = self.__prepare_grid_view()

        # Colormap
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="lightgrey")  # unknown

        self.img = self.ax.imshow(
            self.grid_view,
            origin="lower",
            cmap=cmap,
            vmin=0,
            vmax=1.0
        )

        # 🎨 Different colors per robot
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(self.agents)))

        # Robots
        self.robot_scatter = self.ax.scatter([], [], c=self.colors)

        # Paths
        self.path_lines = {
            agent: self.ax.plot([], [], color=self.colors[i], linewidth=1)[0]
            for i, agent in enumerate(self.agents)
        }

        # ❗ Stuck events (explicit markers)
        self.stuck_scatter = self.ax.scatter([], [], c='black', marker='x', s=60)

        self.ax.set_title("Exploration")
        plt.ion()
        plt.show()

    # ------------------------
    def __prepare_grid_view(self):
        grid_view = np.full(
            (self.terrain_map.height, self.terrain_map.width),
            np.nan
        )

        for (x, y), cell in self.terrain_map.grid.items():
            if cell.traversability_estimate is not None:
                grid_view[y, x] = cell.traversability_estimate

        return grid_view

    # ------------------------
    def plot(self):
        # Update paths
        for agent in self.agents:
            self.agents_path[agent].append((agent.x, agent.y))

        # Update grid
        self.grid_view = self.__prepare_grid_view()
        self.img.set_data(self.grid_view)

        # Update robot positions
        positions = np.array([(a.x, a.y) for a in self.agents])
        self.robot_scatter.set_offsets(positions)

        # Update paths
        for agent in self.agents:
            path = np.array(self.agents_path[agent])
            self.path_lines[agent].set_data(path[:, 0], path[:, 1])

        # Update stuck markers
        stuck_points = [
            (x, y)
            for (x, y), cell in self.terrain_map.grid.items()
            if cell.is_stuck
        ]

        if stuck_points:
            self.stuck_scatter.set_offsets(np.array(stuck_points))
        else:
            self.stuck_scatter.set_offsets(np.empty((0, 2)))

        # 🔥 THIS is what makes it "real-time"
        plt.pause(0.001) #
    # ------------------------
    def __animate(self, steps=200, interval=100):
        self.ani = FuncAnimation(
            self.fig,
            self.update,
            frames=steps,
            interval=interval,
            blit=True
        )
        plt.show()

