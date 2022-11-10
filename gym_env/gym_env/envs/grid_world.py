import gym
from gym import spaces
import pygame
import numpy as np
import math

GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RIGHT = 0
UP = 1
LEFT = 2
DOWN = 3

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100000}

    def __init__(self, render_mode=None):
        self.size_x = 10#  48
        self.size_y = 10#  17
        self.block_size = 20
        self.window_width = self.size_x * self.block_size
        self.window_height = self.size_y * self.block_size

        self.observation_space = spaces.Box(np.array([0, 0]), np.array([self.size_x, self.size_y]), dtype=int)

        # 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken. i.e., 0 corresponds to "right", 1 to "up" etc. 
        Note the above predefined "constants" above for 0, 1, 2, 3
        """
        self._action_to_direction = {
            RIGHT:  np.array([1, 0]),
            UP:     np.array([0, 1]),
            LEFT:   np.array([-1, 0]),
            DOWN:   np.array([0, -1]),
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

        if (old_location == self._agent_location).all(): # if agent hits a wall
            reward = -0.1
        elif terminated:
            reward = 10
        else:
            reward = -0.01

        if self.render_mode == "human":
            self._render_frame()
            
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
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

        # Draw gridlines
        for x in range (0, self.window_width, self.block_size):
            for y in range(0, self.window_height, self.block_size):
                if y == 0 or x == 0 or y == (self.window_height - self.block_size) or x == (self.window_width - self.block_size):
                    rect = pygame.Rect(x, y, self.block_size, self.block_size)
                    pygame.draw.rect(canvas, BLUE, rect)
                
                rect = pygame.Rect(x, y, self.block_size, self.block_size)
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

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
