import tables
import numpy as np
import imageio.v2 as iio
from collections import defaultdict
import sys 
import pygame
import time

# ---------------------------- A* SEARCH ALGORITHM ----------------------------
# 
# Heuristic function uses taxicab geometry: https://en.wikipedia.org/wiki/Taxicab_geometry
# A* algorithm inspired by https://en.wikipedia.org/wiki/A*_search_algorithm

def reconstruct_path(cameFrom, current):
    """
    Reconstructs a path from the current node (goal) and backtracks to the start.
    :param cameFrom: dictionary of previous nodes visited
    :param current:  goal node 
    """
    total_path = [current]
    while current in cameFrom.keys():
        current = cameFrom[current]
        total_path.insert(0, current)
    return total_path

def d(current, neighbor):
    """
    Returns the weight of the edge from current to neighbor. Currently forced to 1.
    """
    return 1

def h(node, goal):
    """
    The heuristic function that calculates the taxicab geometry from the current node to the goal.
    :param node: the current cell in the form of (x, y)
    :param goal: the goal cell in the form of (x, y)
    """
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def get_min_h_cell(openSet, goal):
    min_cell = sys.maxsize
    min_h = sys.maxsize
    for x in openSet:
        x_h = h(x, goal)
        if x_h < min_h:
            min_h = x_h
            min_cell = x
    return min_cell

def a_star(start, goal):
    """
    The A* search algorithm implementation
    :param start: the start cell
    :param goal: the goal cell
    """
    openSet = {start}
    cameFrom = {}

    gScore = defaultdict(lambda: sys.maxsize)
    gScore[start] = 0

    fScore = defaultdict(lambda: sys.maxsize)
    fScore[start] = h(start, goal)

    while openSet:
        current = get_min_h_cell(openSet, goal)

        if current == goal:
            return reconstruct_path(cameFrom, current)

        openSet.remove(current)
        for neighbor in T.get_states_accessible_from(current):
            tentative_gScore = gScore[current] + d(current, neighbor)
            
            if tentative_gScore < gScore[neighbor]:
                cameFrom[neighbor] = current
                gScore[neighbor] = tentative_gScore
                fScore = tentative_gScore + h(neighbor, goal)
                if neighbor not in openSet:
                    openSet.add(neighbor)
    
    return False

# ------------------------------ PYGAME DRAWING -------------------------------

def draw_grid():
    blockSize = 20 # Set the size of the grid block
    for x in range(0, width * blockSize, blockSize):
        for y in range(0, height * blockSize, blockSize):
            rect = pygame.Rect(x, y, blockSize, blockSize)

            _x = int(x / blockSize)
            _y = int(y / blockSize)
            # Color start and goal cells
            if (_x, _y) == start:
                pygame.draw.rect(SCREEN, (255, 255, 0), rect)
            elif (_x, _y) == goal:
                pygame.draw.rect(SCREEN, (0, 255, 0), rect)

            # Draw walls
            if im[_y][_x] == 0:
                pygame.draw.rect(SCREEN, (0, 0, 255), rect)

            pygame.draw.rect(SCREEN, (0, 0, 0), rect, 1)

def pygame_init(path):
    global SCREEN, CLOCK
    blockSize = 20
    pygame.init()
    SCREEN = pygame.display.set_mode((width * 20, height * 20))
    CLOCK = pygame.time.Clock()
    SCREEN.fill((255, 255, 255))

    while True:
        draw_grid()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                for x, y in path:
                    rect = pygame.Rect(x * blockSize, y * blockSize, blockSize, blockSize)
                    pygame.draw.rect(SCREEN, (255, 0, 0), rect)
                    pygame.display.flip()
                    time.sleep(0.1)
        
        pygame.display.update()


if __name__ == '__main__':
    IMAGE_FOLDER = 'images/'
    T = tables.TTable()

    action_to_direction = {
        0:  np.array([1, 0]),  # right
        1:  np.array([0, 1]),  # up
        2:  np.array([-1, 0]),  # left
        3:  np.array([0, -1]),  # down
    }

    # ----------------------------- IMAGE READING -----------------------------

    #im = iio.imread(IMAGE_FOLDER + 'empty_10_10.bmp')
    #im = iio.imread(IMAGE_FOLDER + 'empty.bmp')
    #im = iio.imread(IMAGE_FOLDER + 'mess.bmp')
    im = iio.imread(IMAGE_FOLDER + input("Enter the bitmap filename: "))
    height, width  = im.shape

    # ---------------------- INIT T TABLE (artificially) ----------------------
    for y in range(0, height):
        for x in range(0, width):
            for a in action_to_direction:
                s_prime = tuple(sum(x) for x in zip((x, y), tuple(action_to_direction[a])))

                # force it to stay within the grid and only add non-wall cells
                if (s_prime[0] >= 0 and s_prime[1] >= 0) and (s_prime[0] < width and s_prime[1] < height) and im[s_prime[1]][s_prime[0]] != 0:
                    T[(x, y), a, s_prime] = 0

    # ------------------------- GET START + GOAL NODES ------------------------
    # start = (2, 2)    # (1, 1)
    # goal = (45, 11)   # (8, 5)

    start = tuple(int(x) for x in input("Enter the start cell: ").split(","))
    goal = tuple(int(x) for x in input("Enter the goal cell: ").split(","))

    # ------------------------ CALCULATE PATH AND DRAW ------------------------
    path = a_star(start, goal)
    pygame_init(path)