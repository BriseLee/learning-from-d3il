import os
import numpy as np
import random
import pickle
import gym
from envs.gym_picking_env.gym_picking.envs.picking import Block_Pick_Env
# import gym_picking


# print(gym.envs.registry.items())



env = Block_Pick_Env()
env.start()

random.seed(2)
np.random.seed(2)

test_contexts = []
for i in range(60):

    context = env.manager.sample()
    test_contexts.append(context)

with open("test_contexts.pkl", "wb") as f:
    pickle.dump(test_contexts, f)

# file_lists = os.listdir("train")
file_lists = np.load("train_files.pkl", allow_pickle=True)

train_contexts = []

# for file in file_lists[:60]:

#     arr = np.load("all_data/" + file, allow_pickle=True,)

#     train_contexts.append(arr["context"])
for file in file_lists[:60]:
    file_path = "all_data/" + file
    if os.path.exists(file_path):
        arr = np.load(file_path, allow_pickle=True)
        train_contexts.append(arr["context"])
    else:
        print(f"File not found: {file_path}")

with open("train_contexts.pkl", "wb") as f:
    pickle.dump(train_contexts, f)