import gym
import gym_env
import numpy as np
import random
import tables
import pqueue
from matplotlib import pyplot as plt


MAX_EPISODES = 25
MAX_TRY = 5000

# figure window for perceived values
q_fig, q_ax = plt.subplots(1, 2)
plt.ion()


def model_based_reinforcement_learner(env, Ss, As, discount_factor, epsilon):

    # Create and initialize StateActionTables, TTable, and PQueue.
    Q = tables.StateActionTable(default_value=100)
    R = tables.StateActionTable(default_value=0)
    C = tables.StateActionTable(default_value=0)
    T = tables.TTable()
    PQueue = pqueue.StateActionPQueue()
    q_updates_count = 0

    heatmap = np.full((Ss[1], Ss[0]), fill_value=0, dtype=int)

    for episode in range(MAX_EPISODES):
        total_episode_reward = 0
        
        curr_state, _ = env.reset()

        for t in range(MAX_TRY):

            # select action
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = Q.get_best_action(state=tuple(curr_state), actions=[0, 1, 2, 3])

            # execute action
            next_state, reward, terminated, truncated, info = env.step(action)
            heatmap[next_state[1], next_state[0]] += 1
            total_episode_reward += reward

            # update T, C, and R tables
            T[tuple(curr_state), action, tuple(next_state)] += 1
            C[tuple(curr_state), action] += 1
            R[tuple(curr_state), action] += (reward - R[tuple(curr_state), action]) / C[tuple(curr_state), action]

            # Set priority for current state
            priority = abs(reward + discount_factor * \
                max(Q.get_action_values(state=tuple(next_state), actions=[0, 1, 2, 3]).values()) - Q[tuple(curr_state), action])

            if priority > 0:
                PQueue.insert(curr_state, action, priority)

            while not PQueue.is_empty():
                (x1, y1), act, _ = PQueue.pop()

                Q[(x1, y1), act] = R[(x1, y1), act]

                # Loop for all (state, action) pairs predicted to lead to S:
                for (x2, y2) in T.get_states_accessible_from((x1, y1)):

                    Q[(x1, y1), act] += discount_factor * (T[(x1, y1), act, (x2, y2)] / C[(x1, y1), act]) * \
                                max(Q.get_action_values(state=(x2, y2), actions=[0, 1, 2, 3]).values())
                    q_updates_count += 1

                for (xbar, ybar), act_from_sbar_to_s in T.get_state_actions_with_access_to((x1, y1)):
                    predicted_reward = R[(xbar, ybar), act_from_sbar_to_s]

                    # Set priority for (sbar, act) pair
                    priority = abs(predicted_reward + discount_factor * \
                        max(Q.get_action_values(state=(x1, y1), actions=[0, 1, 2, 3]).values()) - Q[(xbar, ybar), act_from_sbar_to_s])

                    if priority > 0:
                        PQueue.insert((xbar, ybar), act_from_sbar_to_s, priority)
                    
            curr_state = next_state

            if terminated or t >= MAX_TRY:
                break

            # display perceived values
            q_ax[0].cla()
            q_ax[1].cla()
            value_map = (Q.convert_to_np_arr(Ss + (As,), 100)).max(axis=2).transpose()
            q_ax[0].set_title('perceived state values (max Q values)')
            q_ax[1].set_title('agent heat map')
            q_ax[0].imshow(value_map)
            q_ax[1].imshow(heatmap, cmap='hot')
            plt.pause(0.00001)

        print(
            f"Episode #{episode} complete with a total reward of {round(total_episode_reward, 2)}. Target found? {terminated}. Q table accesses is at {q_updates_count}"
        )

    plt.pause(10)
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
