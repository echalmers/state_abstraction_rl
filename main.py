import gym
import gym_env
import numpy as np
import random
from matplotlib import pyplot as plt

# This is just here for testing - allows you to print the full numpy array.
# import sys
# np.set_printoptions(threshold=sys.maxsize)

MAX_EPISODES = 25
MAX_TRY = 5000

# figure window for perceived values
q_fig, q_ax = plt.subplots()
plt.ion()


def model_based_reinforcement_learner(env, Ss, As, discount_factor, epsilon):

    # Create and initialize arrays
    Q = np.full(Ss + (As,), fill_value=100, dtype=int)
    R = np.full(Ss + (As,), fill_value=0, dtype=float)
    T = np.full((Ss[0], Ss[1], Ss[0], Ss[1], As), fill_value=0, dtype=int)
    C = np.full(Ss + (As,), fill_value=0, dtype=int)

    for episode in range(MAX_EPISODES):
        total_episode_reward = 0

        curr_state, _ = env.reset()

        for t in range(MAX_TRY):

            # display perceived values
                q_ax.cla()
                value_map = Q.max(axis=2).transpose()
                q_ax.imshow(value_map)
                plt.title('perceived state values (max Q values)')
                plt.pause(0.0001)

            # select action
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[curr_state[0], curr_state[1]])

            # execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            total_episode_reward += reward

            # update T, C, and R tables
            T[curr_state[0], curr_state[1], next_state[0], next_state[1], action] += 1
            C[curr_state[0], curr_state[1], action] += 1
            R[curr_state[0], curr_state[1], action] += (reward - R[curr_state[0], curr_state[1], action]) / C[curr_state[0], curr_state[1], action]

            # update Q tables
            # naive implementation - super inefficient and slow
            # updates every entry of the Q table after every action
            # (a literal implementation of fig 12.6 from https://artint.info/2e/html/ArtInt2e.Ch12.S8.html)
            for x1 in range(Q.shape[0]):
                for y1 in range(Q.shape[1]):
                    for act in range(Q.shape[2]):
                        if C[x1, y1, act] == 0:
                            continue

                        Q[x1, y1, act] = R[x1, y1, act]

                        for x2 in range(Q.shape[0]):
                            for y2 in range(Q.shape[1]):

                                Q[x1, y1, act] += discount_factor * (T[x1, y1, x2, y2, act] / C[x1, y1, act]) * \
                                                    Q[x2, y2, :].max()

            curr_state = next_state

            if terminated or t >= MAX_TRY:
                break

        print(
            f"Episode #{episode} complete with a total reward of {round(total_episode_reward, 2)}. Target found? {terminated}"
        )

    env.close()


"""
MAIN DRIVER
"""
env = gym.make("gym_env/GridWorld-v0", render_mode="human")
env.action_space.seed(42)

# Set of states
Ss = tuple(
    (env.observation_space.high + np.ones(env.observation_space.shape)).astype(int)
)

# Set of actions
As = env.action_space.n

discount_factor = 0.9

# Value to determine whether or not the agent explores more or not. 
# i.e., a higher epsilon == more exploring
epsilon = 0.1

model_based_reinforcement_learner(env, Ss, As, discount_factor, epsilon)
