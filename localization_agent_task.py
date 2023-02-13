from queue import PriorityQueue
from typing import Dict, Tuple

import cv2
import numpy as np
from matplotlib import pyplot as plt
from environment import Environment
from utils import generate_maze

from math import sqrt
from scipy.stats import norm


def a_star_search(occ_map: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> Dict[Tuple[int, int], Tuple[int, int]]:
    """
    Implements the A* search with heuristic function being distance from the goal position.
    :param occ_map: Occupancy map, 1 – field is occupied, 0 – is not occupied.
    :param start: Start position from which to perform search
    :param end: Goal position to which we want to find the shortest path
    :return: The dictionary containing at least the optimal path from start to end in the form:
        {start: intermediate, intermediate: ..., almost: goal}
    """

    penalty = cv2.blur(occ_map, (3, 3))
    penalty = cv2.blur(occ_map, (15, 15))
    penalty[occ_map == 1] = 1

    OCCUPIED = 1
    FREE = 0
    NEIGHBOURS = ((1, 0), (0, 1), (-1, 0), (0, -1))
    INFINITY = float('inf')

    def _distance(u, v):
        return sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2)

    def _neighbours(v):
        def _in_free_space(v):
            return 0 <= v[0] < occ_map.shape[0] and 0 <= v[1] < occ_map.shape[1] and occ_map[v] == FREE
        return list(filter(_in_free_space, [(v[0] + a, v[1] + b) for a, b in NEIGHBOURS]))
        
    # Check if target pos is reachable.
    if occ_map[start] == OCCUPIED or occ_map[end] == OCCUPIED:
        return dict()
    
    # Note that algorithm is run from [end] to [start]
    # because this way it is easier to compute shortest path
    # from [start] to [end] (one does not have to reverse edges).

    distance_score = {end: 0}
    cummulative_score = {end: _distance(end, start)}
    shortest_path = {end: end}

    # holds pairs: (score, vert)
    que = PriorityQueue()
    que.put((0, end)) # set score to 0 to enforce processing initial vertice

    while not que.empty():
        score, vert = que.get()

        if vert == start:
            return shortest_path
        if score > cummulative_score.get(vert, INFINITY):
            continue

        for u in _neighbours(vert):
            new_distance_score = distance_score[vert] + 1
            if new_distance_score < distance_score.get(u, INFINITY):
                shortest_path[u] = vert
                distance_score[u] = new_distance_score
                cummulative_score[u] = new_distance_score + _distance(u, start) + penalty[u] * 500
                que.put((cummulative_score[u], u))

    return dict()


class LocalizationMap:
    def __init__(self, environment):
        self.environment = environment
        self.gridmap = environment.gridmap
        # Initialize localization probabilities to uniform distribution over
        # free spaces in the known gridmap.
        self.probs = (1 - environment.gridmap).astype(np.float32)
        self.probs = self.probs / np.sum(self.probs)
        assert np.isclose(np.sum(self.probs), 1)

        self.perfect_distances = np.zeros((*self.gridmap.shape, self.environment.lidar_angles), np.float32)
        for r in range(self.gridmap.shape[0]):
            for c in range(self.gridmap.shape[1]):
                xy = self.environment.rowcol_to_xy((r, c))
                _, distances = self.environment.ideal_lidar(xy)
                self.perfect_distances[r, c, :] = distances

        def _pdf(x):
            # Probability density function of measurement model.
            return norm.pdf(x, loc=1, scale=self.environment.lidar_stochasticity)
        
        self.pdf_partition_min = 1 - 5
        self.pdf_partition_max = 1 + 5
        self.pdf_num_partitions = 10000
        partition_size = (self.pdf_partition_max - self.pdf_partition_min) / self.pdf_num_partitions
        self.partitioned_prob = np.linspace(self.pdf_partition_min, self.pdf_partition_max - partition_size, self.pdf_num_partitions) + partition_size / 2
        self.partitioned_prob[:] = [_pdf(x) * partition_size for x in self.partitioned_prob]

    def _partition_index(self, err):
        if err < self.pdf_partition_min or err > self.pdf_partition_max:
            return None
        
        return round((self.pdf_num_partitions - 1) * (err - self.pdf_partition_min) / (self.pdf_partition_max - self.pdf_partition_min))

    def _ideal_lidar(self, rowcol):
        if rowcol not in self.ideal_lidar_cache:
            xy = self.environment.rowcol_to_xy(rowcol)
            self.ideal_lidar_cache[rowcol] = self.environment.ideal_lidar(xy)
        return self.ideal_lidar_cache[rowcol]

    def position_update_by_motion_model(self, delta: np.ndarray) -> None:
        """
        :param delta: Movement taken by agent in the previous turn.
        It should be one of [[0, 1], [0, -1], [1, 0], [-1, 0]]
        """
        # I also accept a [0, 0] case for which nothing is changed.
        delta = tuple(delta)
        if delta == (0, 0):
            return
        
        OCCUPIED = 1
        FREE = 0
        prob_moved = 1 - self.environment.position_stochasticity
        prob_stayed = 1 - prob_moved

        new_probs = np.zeros_like(self.probs)

        gridmap = self.environment.gridmap
        for r in range(gridmap.shape[0]):
            for c in range(gridmap.shape[1]):
                if gridmap[r, c] == OCCUPIED:
                    continue

                prev_pos = (r - delta[0], c - delta[1])
                prev_pos_legal = 0 <= prev_pos[0] < gridmap.shape[0] and 0 <= prev_pos[1] < gridmap.shape[1] and gridmap[prev_pos] == FREE

                prob_came_from_prev_pos = self.probs[prev_pos] * prob_moved if prev_pos_legal else 0 # avoids 'index out of range'
                prob_stayed_at_the_same_pos = self.probs[r, c] * prob_stayed
                new_probs[r, c] = prob_came_from_prev_pos + prob_stayed_at_the_same_pos

        self.probs = new_probs / np.sum(new_probs)


    def position_update_by_measurement_model(self, distances: np.ndarray) -> None:
        """
        Updates the probabilities of agent position using the lidar measurement information.
        :param distances: Noisy distances from current agent position to the nearest obstacle.
        """
        OCCUPIED = 1

        gridmap = self.environment.gridmap
        for r in range(gridmap.shape[0]):
            for c in range(gridmap.shape[1]):
                if gridmap[r, c] == OCCUPIED:
                    continue

                measurement_prob = 1
                for dist, gt in zip(distances, self.perfect_distances[r, c]):
                    err = dist / gt if gt != 0 else 0
                    index = self._partition_index(err)
                    prob_z = self.partitioned_prob[index] if index is not None else 0
                    measurement_prob *= prob_z
                self.probs[r, c] *= measurement_prob

        self.probs = self.probs / np.sum(self.probs) # normalization


    def position_update(self, distances: np.ndarray, delta: np.ndarray = None):
        self.position_update_by_motion_model(delta)
        self.position_update_by_measurement_model(distances)


class LocalizationAgent:
    def __init__(self, environment):
        self.environment = environment
        self.goal_position = environment.goal_position
        self.localization_map = LocalizationMap(environment)
        self.last_action = None
        

    def step(self) -> None:
        """
        Localization agent step, which should (but not have to) consist of the following:
            * reading the lidar measurements
            * updating the agent position probabilities
            * choosing and executing the next agent action in the environment
        """
        _, distances = self.environment.lidar()
        last_action = self.last_action if self.last_action is not None else (0, 0)
        self.localization_map.position_update(distances, last_action)

        probs = self.localization_map.probs
        pos_estimate = int(np.argmax(probs))
        pos_estimate = (pos_estimate // probs.shape[0], pos_estimate % probs.shape[1])

        plan = a_star_search(self.environment.gridmap, pos_estimate, self.environment.xy_to_rowcol(self.goal_position))

        next_pos = plan[pos_estimate]
        action = (next_pos[0] - pos_estimate[0], next_pos[1] - pos_estimate[1])

        self.last_action = action
        self.environment.step(action)

    def visualize(self) -> np.ndarray:
        """
        :return: the matrix of probabilities of estimation of current agent position
        """
        return self.localization_map.probs


if __name__ == "__main__":
    maze = generate_maze((11, 11))
    env = Environment(
        maze,
        lidar_angles=3,
        resolution=1/11/10,
        agent_init_pos=None,
        goal_position=(0.87, 0.87),
        position_stochasticity=0.5
    )
    agent = LocalizationAgent(env)

    while not env.success():
        agent.step()

        if env.total_steps % 10 == 0:
            rowcol = env.xy_to_rowcol(env.position())
            viz = agent.visualize()

            plt.imshow(viz)
            plt.colorbar()
            plt.savefig('/dev/shm/map.png')
            plt.close(plt.gcf())

            img = cv2.imread('/dev/shm/map.png')
            cv2.imshow('map', img)
            cv2.waitKey(1)

    print(f"Total steps taken: {env.total_steps}, total lidar readings: {env.total_lidar_readings}")
