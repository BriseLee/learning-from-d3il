from typing import Optional, Callable, Any
import logging

import os
import glob
import torch
import pickle
import numpy as np

from torch.utils.data import TensorDataset
from environments.dataset.base_dataset import TrajectoryDataset
from agents.utils.sim_path import sim_framework_path
from .geo_transform import quat2euler


class Picking_Dataset(TrajectoryDataset):
    def __init__(
            self,
            data_directory: os.PathLike,
            device="cpu",
            obs_dim: int = 20,
            action_dim: int = 2,
            max_len_data: int = 256,
            window_size: int = 1,
    ):

        super().__init__(
            data_directory=data_directory,
            device=device,
            obs_dim=obs_dim,
            action_dim=action_dim,
            max_len_data=max_len_data,
            window_size=window_size
        )

        logging.info("Loading Block Pick Dataset")

        inputs = []
        actions = []
        masks = []

        # for root, dirs, files in os.walk(self.data_directory):
        #
        #     for mode_dir in dirs:

        # state_files = glob.glob(os.path.join(root, mode_dir) + "/env*")
        # data_dir = os.path.join(sim_framework_path(data_directory), "local")
        # data_dir = sim_framework_path(data_directory)
        # state_files = glob.glob(data_dir + "/env*")

        bp_data_dir = sim_framework_path("environments/dataset/data/picking/pick_data/feedback_withoutforce/state/")

        state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

        for file in state_files:
            with open(os.path.join(bp_data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

            # lengths.append(len(env_state['robot']['des_c_pos']))

            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            # print(f"aaaaaaa{env_state}")  # 输出 env_state 的内容


            # robot and box positions
            robot = "panda_robot"
            state_data = env_state["state"]
            robot_des_j_pos = state_data[robot]['des_j_pos']
            # print(f"des_pos:{robot_des_j_pos}")
            
            robot_gripper = np.expand_dims(state_data[robot]['gripper_width'], -1)
            robot_gripper = robot_gripper[:, 0, :]
            # print(f"gripper:{robot_gripper}")

            red_box_pos = env_state['state']['picked_box']['pos']
            red_box_quat = np.tan(quat2euler(env_state['state']['picked_box']['quat'])[:, -1:])

            green_target_pos = env_state['state']['target_box']['pos']
            green_target_quat= np.tan(quat2euler(env_state['state']['target_box']['quat'])[:, -1:])
            
            input_state = np.concatenate((robot_des_j_pos, robot_gripper, red_box_pos, red_box_quat,green_target_pos,green_target_quat), axis=-1)
            print(f"input_state shape: {input_state.shape}") 
            print(f"zero_obs shape: {zero_obs.shape}")       
            robot_state = np.concatenate((robot_des_j_pos,robot_gripper),axis=1)

            vel_state = robot_state[1:] - robot_state[:-1]
            # vel_state = (robot_des_j_pos[2:] - robot_des_j_pos[:-2]) / 2
            valid_len = len(input_state)-1
            # print(f"valid :{valid_len}")
            # print("vel_state shape:",  zero_obs[0, :valid_len, :12].shape)          # vel_state 的形状
            # print("INPUT shape:", input_state[:-1].shape)  # robot_gripper[1:] 的形状


            zero_obs[0, :valid_len, :] = input_state[:-1]
            zero_action[0, :valid_len, :] = vel_state
            zero_mask[0, :valid_len] = 1

            inputs.append(zero_obs)
            actions.append(zero_action)
            masks.append(zero_mask)

        # shape: B, T, n
        self.observations = torch.from_numpy(np.concatenate(inputs)).to(device).float()
        self.actions = torch.from_numpy(np.concatenate(actions)).to(device).float()
        self.masks = torch.from_numpy(np.concatenate(masks)).to(device).float()

        self.num_data = len(self.observations)

        self.slices = self.get_slices()

    def get_slices(self):
        slices = []

        min_seq_length = np.inf
        for i in range(self.num_data):
            T = self.get_seq_length(i)
            min_seq_length = min(T, min_seq_length)

            if T - self.window_size < 0:
                print(f"Ignored short sequence #{i}: len={T}, window={self.window_size}")
            else:
                slices += [
                    (i, start, start + self.window_size) for start in range(T - self.window_size + 1)
                ]  # slice indices follow convention [start, end)

        return slices

    def get_seq_length(self, idx):
        return int(self.masks[idx].sum().item())

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_all_observations(self):
        result = []
        # mask out invalid observations
        for i in range(len(self.masks)):
            T = int(self.masks[i].sum().item())
            result.append(self.observations[i, :T, :])
        return torch.cat(result, dim=0)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):

        i, start, end = self.slices[idx]

        obs = self.observations[i, start:end]
        act = self.actions[i, start:end]
        mask = self.masks[i, start:end]

        return obs, act, mask