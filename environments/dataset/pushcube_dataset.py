import random
from typing import Optional, Callable, Any
import logging

import os

import torch
import pickle
import numpy as np
from tqdm import tqdm
from .geo_transform import quat2euler
from environments.dataset.base_dataset import TrajectoryDataset
from agents.utils.sim_path import sim_framework_path


class PushCube_Dataset(TrajectoryDataset):
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

        logging.info("Loading Push Cube Dataset")

        inputs = []
        actions = []
        masks = []

        # data_dir = sim_framework_path(data_directory)
        # state_files = glob.glob(data_dir + "/env*")

        rp_data_dir = sim_framework_path("environments/dataset/data/pushcube/push_data/feedback_withoutforce/state/")
        state_files = np.load(sim_framework_path(data_directory), allow_pickle=True)

        for file in state_files:

            with open(os.path.join(rp_data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

            # lengths.append(len(env_state['robot']['des_c_pos']))

            zero_obs = np.zeros((1, self.max_len_data, self.obs_dim), dtype=np.float32)
            zero_action = np.zeros((1, self.max_len_data, self.action_dim), dtype=np.float32)
            zero_mask = np.zeros((1, self.max_len_data), dtype=np.float32)

            # robot and box positions
            robot = "panda_robot"
            state_data = env_state["state"]
            
            robot_des_j_pos = state_data[robot]['des_j_pos']
            robot_des_c_pos = state_data[robot]['des_c_pos'][:,:2]

            robot_c_pos = state_data[robot]['c_pos'][:,:2]
            robot_c_quat = state_data[robot]['c_quat']
            euler_c_angles = quat2euler(robot_c_quat)

            push_box_pos = state_data['pushed_box']['pos'][:,:2]
            push_box_quat = quat2euler(state_data['pushed_box']['quat'])[:, -1:]

            target_box_pos = state_data['target_box']['pos'][:,:2]
            target_box_quat = quat2euler(state_data['target_box']['quat'])[:, -1:]

            # target_box_pos = np.zeros(push_box_pos.shape)
            # target_box_quat = np.zeros(push_box_quat.shape)
            # target_box_pos[:] = push_box_pos[-1:]
            # target_box_quat[:] = push_box_quat[-1:]

            # input_state = np.concatenate((robot_des_j_pos, push_box_pos, push_box_quat, target_box_pos, target_box_quat), axis=-1)
            # input_state = np.concatenate((robot_c_pos, robot_c_quat, push_box_pos, push_box_quat, target_box_pos, target_box_quat), axis=-1)
            input_state = np.concatenate((robot_c_pos, push_box_pos, push_box_quat), axis=-1)
            # robot_des_j_pos = np.array(robot_des_j_pos)
            # input_state = np.concatenate((robot_c_pos, euler_c_angles, push_box_pos, push_box_quat, target_box_pos, target_box_quat), axis=-1)

            # robot_des_c_quat = np.array(robot_c_quat)
            # vel_state = robot_des_c_pos[1:]-robot_des_c_pos[:-1]
            # vel_state = robot_des_j_pos[1:]-robot_des_j_pos[:-1]
            vel_state = robot_c_pos[1:] - robot_c_pos[:-1]  
            # euler_c_delta = euler_c_angles[1:] - euler_c_angles[:-1]  
            # vel_state = np.concatenate((robot_c_pos, robot_c_quat), axis=1)
            # vel_state = np.concatenate((pos_state, euler_c_delta), axis=1)
            # vel_state = robot_des_c_pos[1:]-robot_des_c_pos[:-1]
          
            valid_len = len(input_state)-1
            # print(f"valid {valid_len}")
            # print(f"input_state shape: {input_state.shape}") 
            # print(f"zero_obs shape: {zero_obs.shape}") 
            print(f"vel_state shape: {vel_state.shape}") 

            # zero_obs[0, :valid_len, :] = input_state[:]
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