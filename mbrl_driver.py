import gym
import mbrl
import pygame_envs
import numpy as np
import time
import astar

MAX_EPISODES = 25
MAX_TRY = 5000
discount_factor = 0.9

# for priorities
theta_threshold = 0.01

# Value to determine whether or not the agent explores more or not. 
# i.e., a higher epsilon == more exploring
epsilon = 0.1

# create environment
env = gym.make("GridWorld-v0", render_mode='human', size=(48, 17))

# Initialize the model-based reinforcement learner
mbrl = mbrl.MBRL(
    actions=list(range(env.action_space.n)),
    epsilon=epsilon,
    discount_factor=discount_factor,
    theta_threshold=theta_threshold,
    max_value_iterations=100
)

start_time = time.time()
prev_path = None

for episode in range(MAX_EPISODES):
    s, _ = env.reset()
    start = tuple(s)
    total_episode_reward = 0

    for t in range(MAX_TRY):
        # get action
        a = mbrl.choose_action(s)

        # execute action
        s_prime, reward, terminated, truncated, info = env.step(a)
        total_episode_reward += reward

        # plot
        if t % 10 == 0:
            env.show_plots(mbrl.Q)

        # update StateActionTables and graphs
        mbrl.update(s, a, s_prime, reward)

        # updates the new current state
        s = s_prime

        if terminated or t >= MAX_TRY:
            if terminated:
                env.draw_best_path(astar.a_star(start, tuple(s), mbrl.T), prev_path)
                prev_path = astar.a_star(start, tuple(s), mbrl.T)
            break
    
    print(
        f"Episode #{episode} complete with a total reward of {round(total_episode_reward, 2)}. Target found? {terminated}. Q table accesses is at {mbrl.q_table_updates}"
    )

print('total time:', time.time() - start_time)
time.sleep(10)
env.close()
