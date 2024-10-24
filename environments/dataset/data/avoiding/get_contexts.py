import os
import numpy as np
import random
import pickle
import gym
import sys
import os

sys.path.append(os.path.abspath('/home/xueyinli/project/d3il'))
from environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv
# import gym_picking
# print(gym.envs.registry.items())


# /home/xueyinli/project/d3il/environments/dataset/data/avoiding/data
# all_data = os.listdir("environments/dataset/data/avoiding/data")
env = ObstacleAvoidanceEnv()
env.start()

random.seed(2)
np.random.seed(2)


test_contexts = []
for i in range(60):

    context = env.manager.sample()
    # print(f"context:{context}")
    test_contexts.append(context)

with open("test_contexts.pkl", "wb") as f:
    pickle.dump(test_contexts, f)


# file_lists = os.listdir("all_data")
file_lists = np.load("train_files.pkl", allow_pickle=True)

train_contexts = []

for file in file_lists[:10]:

    arr = np.load("/home/xueyinli/project/d3il/environments/dataset/data/avoiding/data/" + file, allow_pickle=True,)
    print(f"arr:{arr}")
    if "context" in arr:
        train_contexts.append(arr["context"])
    else:
        print(f"Missing 'context' key in: {arr}")

    # train_contexts.append(arr["context"])

with open("train_contexts.pkl", "wb") as f:
    pickle.dump(train_contexts, f)

# test_contexts = []
# for i in range(60):

#     context = env.manager.sample()
#     test_contexts.append(context)

# with open("test_contexts.pkl", "wb") as f:
#     pickle.dump(test_contexts, f)

# # file_lists = os.listdir("train")
# file_lists = np.load("train_files.pkl", allow_pickle=True)

# train_contexts = []


# for file in file_lists[:60]:
#     print(f"okok")
#     file_path = "/home/xueyinli/project/d3il/environments/dataset/data/pushcube/push_data/feedback_withoutforce/state/" + file
#     if os.path.exists(file_path):
#         arr = np.load(file_path, allow_pickle=True)
#         train_contexts.append(arr["context"])
#     else:
#         print(f"File not found: {file_path}")

# with open("train_contexts.pkl", "wb") as f:
#     pickle.dump(train_contexts, f)