import os
import numpy as np
import random
import pickle
import gym
import sys
import gym_stacking

# env = gym.make('stacking-v0', max_steps_per_episode=1000, render=False)
sys.path.append(os.path.abspath('/home/xueyinli/project/d3il'))
from environments.d3il.envs.gym_stacking_env.gym_stacking.envs.stacking import CubeStacking_Env
env = CubeStacking_Env
random.seed(0)
np.random.seed(0)

test_contexts = []
for i in range(100):

    context = env.manager.sample()
    test_contexts.append(context)

with open("test_contexts.pkl", "wb") as f:
    pickle.dump(test_contexts, f)

