import numpy as np


def bresenham(start, end):
    """
    The MIT License (MIT)

    Copyright (c) 2016 - 2022 Atsushi Sakai and other contributors:
    https://github.com/AtsushiSakai/PythonRobotics/contributors

    Implementation of Bresenham's line drawing algorithm
    See en.wikipedia.org/wiki/Bresenham's_line_algorithm
    Bresenham's Line Algorithm
    Produces a np.array from start and end (original from roguebasin.com)
    >>> points1 = bresenham((4, 4), (6, 10))
    >>> print(points1)
    np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
    """
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points


def generate_maze(shape):
    """
    MIT License

    Copyright (c) 2019 郭飞
    https://github.com/guofei9987/python-maze
    """
    class Maze:
        def __init__(self, maze, point):
            self.step_set = np.array([[1, 0],
                                      [-1, 0],
                                      [0, 1],
                                      [0, -1]])
            self.maze = maze
            self.length, self.width = maze.shape
            self.init_maze()
            self.maze = self.find_next_step(self.maze, point)

        def init_maze(self):
            length, width = self.maze.shape
            maze_0 = np.zeros(shape=(length, width))
            maze_0[::2, ::2] = 1
            maze = np.where(self.maze < 0, self.maze, maze_0)
            self.maze = maze

        def find_next_step(self, maze, point):
            step_set = np.random.permutation(self.step_set)
            for next_step in step_set:
                next_point = point + next_step * 2
                x, y = next_point
                if 0 <= x < self.length and 0 <= y < self.width:
                    if maze[x, y] == 1:
                        maze[x, y] = 2
                        maze[(point + next_step)[0], (point + next_step)[1]] = 2
                        maze = self.find_next_step(maze, next_point)
            return maze

    maze = np.zeros(shape=shape)
    start_point = np.array([0, 0])
    maze_generator = Maze(maze[1:-1, 1:-1], start_point)
    maze[:, :] = 1.
    maze[1:-1, 1:-1] = 1 - maze_generator.maze / 2
    return maze
