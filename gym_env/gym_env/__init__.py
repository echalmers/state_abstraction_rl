from gym.envs.registration import register

register(
    id='gym_env/GridWorld-v0',
    entry_point='gym_env.envs:GridWorldEnv',
)