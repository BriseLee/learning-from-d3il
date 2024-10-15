import os
import numpy as np
import random
import pickle
all_data = os.listdir("/home/xueyinli/project/d3il/environments/dataset/data/picking/pick_data/feedback_withoutforce/state")
# file_lists = os.listdir("all_data")
random.shuffle(all_data)


def rotation_distance(p: np.array, q: np.array):
    """
    Calculates the rotation angular between two quaternions
    param p: quaternion
    param q: quaternion
    theta: rotation angle between p and q (rad)
    """
    assert p.shape == q.shape, "p and q should be quaternion"
    theta = 2 * np.arccos(abs(p @ q))
    return theta

z_pos = []

pos_diff = []
quat_diff = []

robot_box_dists = []

lengths = []

# file_lists = np.load("train_files.pkl", allow_pickle=True)

for file in all_data:

    # arr = np.load("/home/xueyinli/project/d3il/environments/dataset/data/picking/pick_data/feedback_withoutforce/state/" + file, allow_pickle=True,)
    with open("/home/xueyinli/project/d3il/environments/dataset/data/picking/pick_data/feedback_withoutforce/state/PickandPlaceBox_096.pkl", "rb") as f:
        arr = np.load(f)
    lengths.append(len(arr['panda_robot']['des_j__pos']))

    red_box_pos = arr['picked_box']['pos'][-1, :2]
    red_box_quat = arr['picked_box']['quat']

    green_target_pos = arr['target_box']['pos'][-1, :2]
#
    pos_diff.append(min(np.linalg.norm(red_box_pos-green_target_pos), np.linalg.norm(red_box_pos-green_target_pos)))
    
lengths = np.array(lengths)

print("data points: ", np.sum(lengths))

print('mean: ', np.mean(lengths))
print('std: ', np.std(lengths))
print('max: ', np.max(lengths))
print('min: ', np.min(lengths))

pos_diff = np.array(pos_diff)
a= 0
print('pos_diff: ', np.mean(pos_diff))