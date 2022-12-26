import gym
from gym import spaces
import pygame
import numpy as np
import math
import time
import matplotlib.pyplot as plt

from tables import StateActionTable

GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100000}

    def __init__(self, img, render_mode=None, size=(48, 17), block_size=20):
        self.size_x = size[0]
        self.size_y = size[1]
        self.block_size = block_size
        self.window_width = self.size_x * self.block_size
        self.window_height = self.size_y * self.block_size
        self.img = img
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([self.size_x, self.size_y]), dtype=int)

        # 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken. i.e., 0 corresponds to "right", 1 to "up" etc. 
        Note the above predefined "constants" above for 0, 1, 2, 3
        """
        self._action_to_direction = {
            0:  np.array([1, 0]),  # right
            1:  np.array([0, 1]),  # up
            2:  np.array([-1, 0]),  # left
            3:  np.array([0, -1]),  # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

        self.q_ax = None
        self.heatmap = np.full((self.size_y, self.size_x), fill_value=0, dtype=int)

    def _get_obs(self):
        return self._agent_location

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Set agent and target location on reset
        self._agent_location = np.array([2, math.floor(self.size_y / 2)])
        self._target_location = np.array([self.size_x - 3, math.floor(self.size_y / 2)])

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        old_location = self._agent_location
        
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, [1, 1], [self.size_x - 2, self.size_y - 2]
        )

        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        observation = self._get_obs()                     
        info = self._get_info()

        if self.img[self._agent_location[1]][self._agent_location[0]] == 0: # if agent hits a wall
            reward = -0.1
            self._agent_location = old_location
        elif terminated:
            reward = 10
        else:
            reward = -0.01

        if self.render_mode == "human":
            self._render_frame()

        self.heatmap[observation[1], observation[0]] += 1
            
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self, path=None):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255))

        pix_square_size = (
            self.block_size
        )  # The size of a single grid square in pixels

        # Draw target
        pygame.draw.rect(
            canvas,
            GREEN,
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        
        # Draw agent
        pygame.draw.circle(
            canvas,
            BLACK,
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        for x in range(0, self.window_width, self.block_size):
            for y in range(0, self.window_height, self.block_size):
                rect = pygame.Rect(x, y, self.block_size, self.block_size)

                _x = int(x / self.block_size)
                _y = int(y / self.block_size)

                # Draw walls
                if self.img[_y][_x] == 0:
                    pygame.draw.rect(canvas, BLUE, rect)

                pygame.draw.rect(canvas, BLACK, rect, 1)

        # Draw best path if its passed in
        if path:
            for x, y in path:
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(canvas, (255, 0, 0), rect)
                pygame.draw.rect(canvas, BLACK, rect, 1)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def draw_best_path(self, curr_path, prev_path):
        if curr_path != prev_path:
            self._render_frame(curr_path)
            time.sleep(1)

    def show_plots(self, Q_table: StateActionTable):
        if self.q_ax is None:
            _, self.q_ax = plt.subplots(1, 2)
            plt.ion()

        self.q_ax[0].cla()
        self.q_ax[0].set_title('perceived state values (max Q values)')
        value_map = np.zeros((self.size_y, self.size_x))
        for coords in np.ndindex(value_map.shape):
            value_map[coords[0], coords[1]] = max(Q_table.get_action_values((coords[1], coords[0]), list(self._action_to_direction)).values())
        self.q_ax[0].imshow(value_map, cmap='copper')

        self.q_ax[1].cla()
        self.q_ax[1].set_title('agent heat map')
        self.q_ax[1].imshow(self.heatmap, cmap='hot')

        plt.pause(0.00001)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
