from gym.envs.registration import register

register(
    id="picking-v0",
    entry_point="gym_picking.envs:Block_Pick_Env",
    max_episode_steps=500,
)
