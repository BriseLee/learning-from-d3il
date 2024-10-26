import logging
import os

import multiprocessing as mp
import random
import sys
sys.path.append(os.path.abspath('/home/xueyinli/project/d3il'))
from environments.d3il.envs.gym_pushcube_env.gym_pushcube.envs.pushcube import Push_Cube_Env
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat, quat2euler
import numpy as np
import torch
import wandb

from simulation.base_sim import BaseSim
from agents.utils.sim_path import sim_framework_path


log = logging.getLogger(__name__)


train_contexts = np.load(sim_framework_path("environments/dataset/data/pushcube/train_contexts.pkl"),
                         allow_pickle=True)

test_contexts = np.load(sim_framework_path("environments/dataset/data/pushcube/test_contexts.pkl"),
                        allow_pickle=True)


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class PushCube_Sim(BaseSim):
    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_contexts: int = 30,
            n_trajectories_per_context: int = 1,
            if_vision: bool = False
    ):
        super().__init__(seed, device, render, n_cores, if_vision)

        self.n_contexts = n_contexts
        self.n_trajectories_per_context = n_trajectories_per_context

    def eval_agent(self, agent, contexts, n_trajectories, mode_encoding, successes, mean_distance, pid, cpu_set):

        print(os.getpid(), cpu_set)
        assign_process_to_cpu(os.getpid(), cpu_set)

        env = Push_Cube_Env(render=self.render, if_vision=self.if_vision)
        env.start()

        random.seed(pid)
        torch.manual_seed(pid)
        np.random.seed(pid)

        for context in contexts:
            for i in range(n_trajectories):

                agent.reset()

                print(f'Context {context} Rollout {i}')
                # training contexts
                # env.manager.set_index(context)
                # obs = env.reset(random=False, context=test_contexts[context])

                # obs = env.reset()
                print(f"test_contexts[context]: {test_contexts[context]}")
                obs = env.reset(random=False, context=test_contexts[context])

                # test contexts
                # test_context = env.manager.sample()
                # obs = env.reset(random=False, context=test_context)

                if self.if_vision:
                    env_state, bp_image, inhand_image = obs
                    bp_image = bp_image.transpose((2, 0, 1)) / 255.
                    inhand_image = inhand_image.transpose((2, 0, 1)) / 255.

                    des_robot_pos = env_state[:3]
                    done = False

                    while not done:
                        pred_action = agent.predict((bp_image, inhand_image, des_robot_pos), if_vision=self.if_vision)
                        pred_action = pred_action[0] + des_robot_pos

                        pred_action = np.concatenate((pred_action, [0, 1, 0, 0]), axis=0)
                        obs, reward, done, info = env.step(pred_action)

                        des_robot_pos = pred_action[:7]

                        robot_pos, bp_image, inhand_image = obs

                        # cv2.imshow('0', bp_image)
                        # cv2.waitKey(1)
                        #
                        # cv2.imshow('1', inhand_image)
                        # cv2.waitKey(1)

                        bp_image = bp_image.transpose((2, 0, 1)) / 255.
                        inhand_image = inhand_image.transpose((2, 0, 1)) / 255.

                else:


                    pred_action = env.robot_state()
                    pred_action_aim=[]
                    pred_action_final=[]
                    obs_new = []
                    done = False
                    z=0.28
                    while not done:
#TO DO change something
                        # pred_action=pred_action[:6].flatten()
                        pred_action=pred_action[:2].flatten()
                        # obs_new = np.concatenate((pred_action[:3],[0,1,0,0], obs))
                        obs_new = np.concatenate((pred_action, obs))
                        # print(f"pred_action first: {pred_action}")
                        print(f"obs: {obs_new}")
                        # print(f"predict obs first: {agent.predict(obs_new)}")
                        pred_action = agent.predict(obs_new)
                        print(f"pred_action: {pred_action}")
                        # print(f"obs shape: {obs.shape}")
                        
                        # TODO: Ask david
                        pred_action = pred_action[:2]+obs_new[:2]
                        # pred_action = pred_action[:2]
                        pred_action = pred_action.flatten()
                        pred_action_aim = np.concatenate((pred_action, np.array([z])))
                        # pred_action_aim = pred_action.flatten()
                        # print(f"pred_action aim: {pred_action_aim}")
                        # pred_quat = euler2quat(pred_action_aim[3:6]).flatten()
                        pred_action_final = np.concatenate((pred_action_aim[:3],[0,1,0,0]), axis=0)
                        # pred_action_final = np.concatenate((pred_action_aim[:3],pred_quat), axis=0)

                        # pred_action_final = pred_action_aim
                        # print(f"pred_action final: {pred_action_final}")


                        # pred_action = np.concatenate((pred_action), axis=0)

                        obs, reward, done, info = env.step(pred_action_final)

                mode_encoding[context, i] = torch.tensor(info['mode'])
                successes[context, i] = torch.tensor(info['success'])
                mean_distance[context, i] = torch.tensor(info['mean_distance'])
               

    ################################
    # we use multi-process for the simulation
    # n_contexts: the number of different contexts of environment
    # n_trajectories_per_context: test each context for n times, this is mostly used for multi-modal data
    # n_cores: the number of cores used for simulation
    ###############################
    def test_agent(self, agent):

        log.info('Starting trained model evaluation')

        mode_encoding = torch.zeros([self.n_contexts, self.n_trajectories_per_context]).share_memory_()
        successes = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()
        mean_distance = torch.zeros((self.n_contexts, self.n_trajectories_per_context)).share_memory_()

        contexts = np.arange(self.n_contexts)

        workload = self.n_contexts // self.n_cores

        num_cpu = mp.cpu_count()
        cpu_set = list(range(num_cpu))

        # start = self.seed * 20
        # end = start + 20
        #
        # cpu_set = cpu_set[start:end]
        print("there are cpus: ", num_cpu)

        ctx = mp.get_context('spawn')

        p_list = []
        if self.n_cores > 1:
            for i in range(self.n_cores):
                p = ctx.Process(
                    target=self.eval_agent,
                    kwargs={
                        "agent": agent,
                        "contexts": contexts[i * workload:(i + 1) * workload],
                        "n_trajectories": self.n_trajectories_per_context,
                        "mode_encoding": mode_encoding,
                        "successes": successes,
                        "mean_distance": mean_distance,
                        "pid": i,
                        "cpu_set": set(cpu_set[i:i + 1]),
                    },
                )
                print("Start {}".format(i))
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]

        else:
            self.eval_agent(agent, contexts, self.n_trajectories_per_context, mode_encoding, successes, mean_distance, 0, cpu_set=set([0]))

        n_modes = 2

        success_rate = torch.mean(successes).item()
        mode_probs = torch.zeros([self.n_contexts, n_modes])
        if n_modes == 1:
            for c in range(self.n_contexts):
                mode_probs[c, :] = torch.tensor(
                    [sum(mode_encoding[c, successes[c, :] == 1] == 0) / self.n_trajectories_per_context])

        elif n_modes == 2:
            for c in range(self.n_contexts):
                mode_probs[c, :] = torch.tensor(
                    [sum(mode_encoding[c, successes[c, :] == 1] == 0) / self.n_trajectories_per_context,
                     sum(mode_encoding[c, successes[c, :] == 1] == 1) / self.n_trajectories_per_context])

        mode_probs /= (mode_probs.sum(1).reshape(-1, 1) + 1e-12)
        print(f'p(m|c) {mode_probs}')

        entropy = - (mode_probs * torch.log(mode_probs + 1e-12) / torch.log(
            torch.tensor(n_modes))).sum(1).mean()

        wandb.log({'score': 0.5 * (success_rate + entropy)})
        wandb.log({'Metrics/successes': success_rate})
        wandb.log({'Metrics/entropy': entropy})
        wandb.log({'Metrics/distance': mean_distance.mean().item()})

        print(f'Mean Distance {mean_distance.mean().item()}')
        print(f'Successrate {success_rate}')
        print(f'entropy {entropy}')

        return success_rate, mode_encoding#, mean_distance