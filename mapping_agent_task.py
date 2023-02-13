import itertools
from typing import Tuple, Optional

import cv2

import numpy as np
from matplotlib import pyplot as plt

from environment import Environment
from localization_agent_task import a_star_search
from utils import generate_maze, bresenham

from math import sin, cos


class OccupancyMap:
    def __init__(self, environment):
        self.env = environment
        self.lods = np.zeros_like(environment.gridmap, dtype=np.float32)

    def point_update(self, pos: Tuple[int, int], distance: Optional[float], total_distance: Optional[float], occupied: bool) -> None:
        """
        Update regarding noisy occupancy information inferred from lidar measurement.
        :param pos: rowcol grid coordinates of position being updated
        :param distance: optional distance from current agent position to the :param pos: (your solution don't have to use it)
        :param total_distance: optional distance from current agent position to the final cell from current laser beam (your solution don't have to use it)
        :param occupied: whether our lidar reading tell us that a cell on :param pos: is occupied or not
        """
        self.lods[pos] += 0.1 if occupied else -0.01

    def map_update(self, pos: Tuple[float, float], angles: np.ndarray, distances: np.ndarray) -> None:
        """
        :param pos: current agent position in xy in [0; 1] x [0; 1]
        :param angles: angles of the beams that lidar has returned
        :param distances: distances from current agent position to the nearest obstacle in directions :param angles:
        """
        for angle, dist in zip(angles, distances):
            start = pos
            end = (start[0] + cos(angle) * dist, start[1] + sin(angle) * dist)
            start = self.env.xy_to_rowcol(start)
            end = self.env.xy_to_rowcol(end)
            points = bresenham(start, end)
            for p in points:
                if 0 <= p[0] < self.lods.shape[0] and 0 <= p[1] < self.lods.shape[1]:
                    self.point_update((p[0], p[1]), None, None, p[0] == end[0] and p[1] == end[1])

        # Clamp to avoid overflow.
        self.lods[self.lods < -50] = -50
        self.lods[self.lods > 50] = 50


class MappingAgent:
    def __init__(self, environment):
        self.env = environment
        self.occmap = OccupancyMap(environment)
        self.goal_pos = self.env.xy_to_rowcol(self.env.goal_position)

    def step(self) -> None:
        """
        Mapping agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the occupancy map beliefs/probabilities about their state
            * choosing and executing the next agent action in the environment
        """
        angles, distances = self.env.lidar()
        pos = self.env.position()
        self.occmap.map_update(pos, angles, distances)

        OCCUPIED_THRESH = 1
        binary_occmap = (self.occmap.lods >= OCCUPIED_THRESH).astype(int)
        pos = self.env.xy_to_rowcol(pos)
        path = a_star_search(binary_occmap, pos, self.goal_pos)
        next_cell = path[pos]
        action = (next_cell[0] - pos[0], next_cell[1] - pos[1])
        self.env.step(action)

    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of given cell occupancy
        """
        return 1 - (1 / (1 + np.exp(self.occmap.lods)))


if __name__ == "__main__":
    maze = generate_maze((11, 11))

    env = Environment(
        maze,
        resolution=1/11/10,
        agent_init_pos=(0.136, 0.136),
        goal_position=(0.87, 0.87),
        lidar_angles=256
    )
    agent = MappingAgent(env)
    
    while not env.success():
        agent.step()

        if env.total_steps % 10 == 0:
            plt.imshow(agent.visualize())
            plt.colorbar()
            plt.savefig('/tmp/map.png')
            plt.close(plt.gcf())

            cv2.imshow('map', cv2.imread('/tmp/map.png'))
            cv2.waitKey(1)

    print(f"Total steps taken: {env.total_steps}, total lidar readings: {env.total_lidar_readings}")
