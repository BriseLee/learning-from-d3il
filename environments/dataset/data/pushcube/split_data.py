

import os
import numpy as np
import random
import pickle
import matplotlib.pyplot as plt

random.seed(42)

train_files = []
eval_files = []

all_data = os.listdir("/home/xueyinli/project/d3il/environments/dataset/data/pushcube/push_data/feedback_withoutforce/state/")

random.shuffle(all_data)

# num_split = int(len(all_data) * 0.1)
num_split = 80

train_files += all_data[num_split:]
eval_files += all_data[:num_split]

with open("train_files.pkl", "wb") as f:
    pickle.dump(train_files, f)

with open("eval_files.pkl", "wb") as f:
    pickle.dump(eval_files, f)

print("finish")