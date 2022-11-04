import gym
import gym_env
import numpy as np
import random

# This is just here for testing - allows you to print the full numpy array.
# import sys
# np.set_printoptions(threshold=sys.maxsize)

MAX_EPISODES = 8
MAX_TRY = 5000


def model_based_reinforcement_learner(env, Ss, As, discount_factor, epsilon):
    # Create and initialize arrays
    Q = np.full(Ss + (As,), fill_value=10, dtype=int)
    R = np.full(Ss + (As,), fill_value=0, dtype=float)
    T = np.full((Ss[0], Ss[1], Ss[0], Ss[1], As), fill_value=0, dtype=int)
    C = np.full(Ss + (As,), fill_value=0, dtype=int)

    for episode in range(MAX_EPISODES):
        total_episode_reward = 0

        curr_state, _ = env.reset()
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)

        s1 = curr_state
        a1 = action
        s2 = next_state

        for t in range(MAX_TRY):
            T[curr_state[0], curr_state[1], next_state[0], next_state[1], action] += 1
            C[curr_state[0], curr_state[1], action] += 1

            R[curr_state[0], curr_state[1], action] += (reward - R[curr_state[0], curr_state[1], action]) / C[curr_state[0], curr_state[1], action]

            total_episode_reward += reward
            curr_state = next_state

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[curr_state[0], curr_state[1]])

            a2 = action

            Q[s1[0], s1[1], a1] = R[s1[0], s1[1], a1] + discount_factor * (T[s1[0], s1[1], s2[0], s2[1], a1] / C[s1[0], s1[1], a1]) * np.argmax(Q[s2[0], s2[1], a2])

            curr_state = s2
            s1 = s2
            a1 = a2

            next_state, reward, terminated, truncated, info = env.step(action)

            s2 = next_state

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
epsilon = 0.9

model_based_reinforcement_learner(env, Ss, As, discount_factor, epsilon)
