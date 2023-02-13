from typing import Optional, Tuple

import pybullet as p
import numpy as np
import pybullet_data
import math
import functools

p.connect(p.GUI)


class Environment:
    def __init__(
            self,
            grid: np.ndarray,
            agent_init_pos: Optional[Tuple[float, float]],
            goal_position: Optional[Tuple[float, float]],
            resolution=0.01,
            position_stochasticity=0.,
            lidar_stochasticity=0.2,
            lidar_angles=32
    ):
        """
        Simple grid world environment.

        :param grid: numpy array which specifies the grid occupancy
        :param agent_init_pos: initial position in [0; 1] x [0; 1] of the agent
        :param goal_position: goal position in [0; 1] x [0; 1] of the agent which
        :param resolution:
            the grid of possible configurations will be of size (1 / resolution) x (1 / resolution)
        :param position_stochasticity: probability that agent will stuck in previous position after making a move attempt.
        :param lidar_stochasticity: std of scaling factor for lidar distance noise model, see lidar methods
        :param lidar_angles: number of angles that lidar will return
        """
        assert grid.shape[0] == grid.shape[1]

        self.goal_position = goal_position
        self.lidar_angles = lidar_angles
        self.position_stochasticity = position_stochasticity
        self.lidar_stochasticity = lidar_stochasticity
        self.resolution = resolution
        k = int((1 / self.resolution) / grid.shape[0])
        self.gridmap = grid.repeat(k, axis=0).repeat(k, axis=1).T
        self.total_steps = 0
        self.total_lidar_readings = 0

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        # Load the environment: urdf file is not quicker.
        self.cell_length = 1 / grid.shape[0]
        for row, col in np.ndindex(grid.shape):
            if grid[row, col] == 1:
                self.load_cube(col * self.cell_length, row * self.cell_length, self.cell_length / 2, self.cell_length, color=[1, 0.5, 0.7, 1])

        # Init agent position randomly in a not occupied position if initial agent position not set.
        if agent_init_pos is None:
            idxs = list(np.ndindex(self.gridmap.shape))
            agent_init_pos = self.rowcol_to_xy(idxs[np.random.choice(
                range(len(idxs)), p=(1 - self.gridmap).flatten() / np.sum(1 - self.gridmap))])

        # Load the agent.
        self.agent = self.load_cube(*agent_init_pos, self.cell_length / 4, self.cell_length / 2, color=[0.5, 1., 0.5, 1])

    def load_cube(self, x, y, z, length, color=None) -> int:
        if color is None:
            color = [1, 0.5, 0.7, 1]
        cube = p.loadURDF("cube.urdf", [x + length / 2, y + length / 2, z], globalScaling=length)
        p.changeVisualShape(cube, -1, rgbaColor=color)
        return cube

    def xy_to_rowcol(self, pos: Tuple[float, float]) -> Tuple[int, int]:
        return int(round(pos[0] / self.resolution)), int(round(pos[1] / self.resolution))

    def rowcol_to_xy(self, rowcol: Tuple[int, int]) -> Tuple[float, float]:
        return (rowcol[0] * self.resolution, rowcol[1] * self.resolution)

    def step(self, delta: Tuple[int, int]) -> None:
        """
        Performs one step of the agent of the grid
        """
        assert delta in [(0, 1), (1, 0), (0, -1), (-1, 0)]
        position, orn = p.getBasePositionAndOrientation(self.agent)
        pos = self.xy_to_rowcol((position[0], position[1]))
        pos, delta = np.array(pos), np.array(delta)

        # Check if target position is occupied, if so, don't move the agent.
        target = pos + delta
        if self.gridmap[target[0], target[1]]:
            return

        # Move the agent, applying simple stochasticity movement model.
        target = self.rowcol_to_xy(target)
        if np.random.uniform(0, 1) > self.position_stochasticity:
            p.resetBasePositionAndOrientation(self.agent, [*target, position[2]], orn)

        self.total_steps += 1

    def position(self) -> Tuple[float, float]:
        """
        Returns agent position, use only in mapping setting.
        :return: position of the agent in xy coordinates, [0; 1] x [0; 1] range.
        """
        pos, orn = p.getBasePositionAndOrientation(self.agent)
        return pos[0], pos[1]

    def success(self) -> bool:
        return np.linalg.norm(np.array(self.position()) - self.goal_position) < self.resolution * 3

    @functools.cache
    def ideal_lidar(self, pos: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns ideal lidar distances, use directly only in localization setting.
        You can use it in localization setting any time you want.

        :param pos: given position in xy coordinates for which to calculate lidar model.
        :return: ideal lidar distances.
        """
        pos = np.asarray(pos)
        angles = [2 * math.pi * k / self.lidar_angles for k in range(self.lidar_angles)]
        pos = np.append(pos, 0.75 * self.cell_length)
        ray_to_positions = np.array([pos + 2 * np.array([math.cos(angle), math.sin(angle), 0.]) for angle in angles])
        ray_from_positions = np.array([pos] * len(angles))
        results = p.rayTestBatch(rayFromPositions=ray_from_positions, rayToPositions=ray_to_positions)
        hit_positions = np.array([r[3] for r in results])
        return angles, np.linalg.norm(hit_positions - ray_from_positions, axis=1)

    def lidar(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: lidar distances from current agent position with applied noise model.
        """
        self.total_lidar_readings += 1
        angles, distances = self.ideal_lidar(tuple(self.position()))
        return angles, distances * np.random.normal(loc=1, scale=self.lidar_stochasticity, size=len(angles))
