from gym.envs.registration import register

register(
    id='MiniGrid-GridWorld-v0',
    entry_point='minigrid_envs.envs:GridWorld',
)