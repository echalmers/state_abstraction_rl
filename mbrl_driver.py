import gym
import mbrl
import gym_env
import numpy as np

MAX_EPISODES = 25
MAX_TRY = 5000

env = gym.make("gym_env/GridWorld-v0", render_mode="human")
env.action_space.seed(42)

# Set of states
states = tuple(
    (env.observation_space.high + np.ones(env.observation_space.shape)).astype(int)
)

# Set of actions
action_space = env.action_space.n
actions = [i for i in range(action_space)]

discount_factor = 0.9

# for priorities
theta_threshold = 0

# Value to determine whether or not the agent explores more or not. 
# i.e., a higher epsilon == more exploring
epsilon = 0.1

# Initialize the model-based reinforcement learner
mbrl = mbrl.MBRL(
    states=states,
    action_space=action_space,
    actions=actions,
    epsilon=epsilon,
    discount_factor=discount_factor,
    theta_threshold=theta_threshold,
    display_graphs=False
)

for episode in range(MAX_EPISODES):
    s, _ = env.reset()

    for t in range(MAX_TRY):
        # get action
        a = mbrl.choose_action(s, env.action_space.sample())

        # execute action
        s_prime, reward, terminated, truncated, info = env.step(a)

        # update StateActionTables and graphs
        mbrl.update(s, a, s_prime, reward)

        # updates the new current state
        s = s_prime

        if terminated or t >= MAX_TRY:
            break
    
    print(
        f"Episode #{episode} complete with a total reward of {round(mbrl.total_episode_reward, 2)}. Target found? {terminated}. Q table accesses is at {mbrl.q_table_updates}"
    )
    mbrl.reset_total_episode_reward()

mbrl.plt.pause(10)
env.close()