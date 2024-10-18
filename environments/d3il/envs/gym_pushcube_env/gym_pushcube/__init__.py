from gym.envs.registration import register

register(
    id="pushcube-v0",
    entry_point="gym_pushcube.envs:Push_Cube_Env",
    max_episode_steps=400,
    kwargs={'render':False, 'if_vision':False}
)
