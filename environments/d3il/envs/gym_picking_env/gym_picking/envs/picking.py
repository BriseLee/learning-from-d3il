import numpy as np
import copy
import time

import sys
import sys
import os

sys.path.append(os.path.abspath('/home/xueyinli/project/d3il'))


from gym.spaces import Box

from environments.d3il.d3il_sim.utils.sim_path import d3il_path
from environments.d3il.d3il_sim.core import Scene
from environments.d3il.d3il_sim.core.logger import ObjectLogger, CamLogger
from environments.d3il.d3il_sim.gyms.gym_env_wrapper import GymEnvWrapper
from environments.d3il.d3il_sim.gyms.gym_utils.helpers import obj_distance
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat, quat2euler

from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from environments.d3il.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from environments.d3il.d3il_sim.sims import MjCamera

from .objects.picking_objects import get_obj_list, init_end_eff_pos
obj_list = get_obj_list()
# picked_box, target_box = get_obj_list()


class BPCageCam(MjCamera):
    """
    Cage camera. Extends the camera base class.
    """

    def __init__(self, width: int = 96, height: int = 96, *args, **kwargs):
        super().__init__(
            "bp_cam",
            width,
            height,
            init_pos=[1.05, 0, 1.2],
            init_quat=[
                0.6830127,
                0.1830127,
                0.1830127,
                0.683012,
            ],  # Looking with 30 deg to the robot
            *args,
            **kwargs,
        )


class BlockContextManager:
    def __init__(self, scene, index=0, seed=42) -> None:
        self.scene = scene

        np.random.seed(seed)

        self.red_box_space = Box(
            low=np.array([0.4, -0.5, -90]), high=np.array([0.5, 0.5, 90])#, seed=seed
        )
        self.target_space = Box(
            low=np.array([0.4, -0.25, -90]), high=np.array([0.6, 0.25, 90])  # , seed=seed
        )
       
        self.index = index

    def start(self, random=True, context=None):

        if random:
            self.context = self.sample()
        else:
            self.context = context
        self.context = self.sample()
        self.set_context(self.context)

    def sample(self):

        picked_box_pos = self.red_box_space.sample()
        pick_angle = [0, 0, picked_box_pos[-1] * np.pi / 180]
        picked_box_quat = euler2quat(pick_angle)
        print(f"picked pos:{picked_box_pos} qua is:{picked_box_quat}")

        target_box_pos = self.target_space.sample()
        target_angle = [0, 0, target_box_pos[-1] * np.pi / 180]
        target_box_quat = euler2quat(target_angle)
        print(f"target pos:{target_box_pos} qua is:{target_box_quat}")

        context=(
            [picked_box_pos, picked_box_quat],
            [target_box_pos,target_box_quat],
        )

        

        return context

    def set_context(self, context):
        picked_pos = context[0][0]
        picked_quat = context[0][1]
        print(f"context{context},picked{picked_pos} p_q is:{picked_quat}")
        target_pos = context[1][0]
        target_quat = context[1][1]
        print(f"target{target_pos } t_q is:{target_quat}")
       
        

        self.scene.set_obj_pos_and_quat(
            [picked_pos[0], picked_pos[1], 0.0],
            picked_quat,
            obj_name="picked_box",
        )

        self.scene.set_obj_pos_and_quat(
            [target_pos[0], target_pos[1], 0.0],
            target_quat,
            obj_name="target_box",
        )

    
    # def random_context(self):

    #     red_pos = self.red_box_space.sample()
    #     goal_angle = [0, 0, red_pos[-1] * np.pi / 180]
    #     quat = euler2quat(goal_angle)

    #     self.scene.set_obj_pos_and_quat(
    #         [red_pos[0], red_pos[1], 0.00],
    #         quat,
    #         obj_name="picked_box",
    #     )


    #     return red_pos, quat

    # def olb_set_context(self, index):
    #     goal_angle = [0, 0, self.deg_list[index] * np.pi / 180]
    #     quat = euler2quat(goal_angle)

    #     self.scene.set_obj_pos_and_quat(
    #         [self.x1_list[index], self.y_list[index], 0.00],
    #         quat,
    #         obj_name="push_box",
    #     )

        # goal_angle2 = [0, 0, self.deg_list[len(self.x1_list) + index] * np.pi / 180]
        # quat2 = euler2quat(goal_angle2)
        # self.scene.set_obj_pos_and_quat(
        #     [self.x2_list[index], self.y_list[len(self.x1_list) + index], 0.00],
        #     quat2,
        #     obj_name="push_box2",
        # )
        # print("Set Context {}".format(index))

    # def next_context(self):
    #     self.index = (self.index + 1) % len(self.x1_list)
    #     self.olb_set_context(self.index)

    def set_index(self, index):
        self.index = index


class Block_Pick_Env(GymEnvWrapper):
    def __init__(
        self,
        n_substeps: int = 35,
        max_steps_per_episode: int = 5000,
        debug: bool = False,
        random_env: bool = False,
        interactive: bool = False,
        render: bool = True
    ):

        sim_factory = MjFactory()
        render_mode = Scene.RenderMode.HUMAN if render else Scene.RenderMode.BLIND
        scene = sim_factory.create_scene(
            object_list=obj_list, render=render_mode, dt=0.001
        )
        #why use invisible robot here??
        robot = MjRobot(
            scene,
            xml_path=d3il_path("./models/mj/robot/panda_invisible.xml"),
        )
        # controller = robot.cartesianPosQuatTrackingController
        controller = robot.jointTrackingController
        # controller = GymCartesianVelController(robot, fixed_orientation=[0,1,0,0])

        super().__init__(
            scene=scene,
            controller=controller,
            max_steps_per_episode=max_steps_per_episode,
            n_substeps=n_substeps,
            debug=debug,
        )

        self.action_space = Box(
            low=np.array([-0.01, -0.01]), high=np.array([0.01, 0.01])
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(8, )
        )

        self.interactive = interactive

        self.random_env = random_env
        self.manager = BlockContextManager(scene, index=2)

        self.bp_cam = BPCageCam()
        self.inhand_cam = robot.inhand_cam
        self.scene.add_object(self.bp_cam)

        self.picked_box = obj_list[0]
        
        self.target_box= obj_list[1]
        

        # for obj in [
        #     self.picked_box,
            
        #     self.target_box,
          
        # ]:
        #     self.scene.add_object(obj)

        

        self.log_dict = {
            "picked_box": ObjectLogger(scene, self.picked_box),
            "target_box": ObjectLogger(scene, self.target_box),
        }

        self.cam_dict = {
            "bp-cam": CamLogger(scene, self.bp_cam),
            "inhand-cam": CamLogger(scene, self.inhand_cam)
        }

        # for _, v in self.log_dict.items():
        #     scene.add_logger(v)

        # # for _, v in self.cam_dict.items():
        # #     scene.add_logger(v)

        # self.target_min_dist = 0.02
        # self.bp_mode = None
        # self.first_visit = -1
        # self.mode_encoding = []
        for _, v in self.log_dict.items():
            scene.add_logger(v)

        for _, v in self.cam_dict.items():
            scene.add_logger(v)

        self.target_min_dist = 0.01

        self.min_inds = []
        self.mode_encoding = []
    
    def robot_state(self):
        # Update Robot State
        self.robot.receiveState()

        # joint state
        joint_pos = self.robot.des_joint_pos
        joint_vel = self.robot.current_j_vel
        gripper_width = np.array([self.robot.gripper_width])

        tcp_pos = self.robot.current_c_pos
        tcp_quad = self.robot.current_c_quat

        return np.concatenate((joint_pos, gripper_width))
        # return np.concatenate((tcp_pos, tcp_quad, gripper_width))


    def get_observation(self) -> np.ndarray:

        robot_pos = self.robot_state()[:7]
        gripper_width = np.array([self.robot.gripper_width])

        picked_box_pos = self.scene.get_obj_pos(self.picked_box)  # - robot_pos
        picked_box_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.picked_box))[-1:])

        target_pos = self.scene.get_obj_pos(self.target_box) #- robot_c_pos
        target_quat = np.tan(quat2euler(self.scene.get_obj_quat(self.target_box))[-1:])

        env_state = np.concatenate(
            [
                robot_pos,
                gripper_width,
                picked_box_pos,
                picked_box_quat,
                target_pos,
                target_quat,

                # goal_1_pos,
                # goal_2_pos,
            ]
        )

        return env_state.astype(np.float32)
        # return np.concatenate([robot_state, env_state])

    def start(self):
        self.scene.start()

        # reset view of the camera
        if self.scene.viewer is not None:
            # self.scene.viewer.cam.elevation = -55
            # self.scene.viewer.cam.distance = 1.7
            # self.scene.viewer.cam.lookat[0] += -0.1
            # self.scene.viewer.cam.lookat[2] -= 0.2

            self.scene.viewer.cam.elevation = -55
            self.scene.viewer.cam.distance = 2.0
            self.scene.viewer.cam.lookat[0] += 0
            self.scene.viewer.cam.lookat[2] -= 0.2

            # self.scene.viewer.cam.elevation = -60
            # self.scene.viewer.cam.distance = 1.6
            # self.scene.viewer.cam.lookat[0] += 0.05
            # self.scene.viewer.cam.lookat[2] -= 0.1

        # reset the initial state of the robot
        initial_cart_position = copy.deepcopy(init_end_eff_pos)
        # initial_cart_position[2] = 0.12
        self.robot.gotoCartPosQuatController.setDesiredPos(
            [
                initial_cart_position[0],
                initial_cart_position[1],
                initial_cart_position[2],
                0,
                1,
                0,
                0,
            ]
        )
        self.robot.gotoCartPosQuatController.initController(self.robot, 1)

        self.robot.init_qpos = self.robot.gotoCartPosQuatController.trajectory[
            -1
        ].copy()
        self.robot.init_tcp_pos = initial_cart_position
        self.robot.init_tcp_quat = [0, 1, 0, 0]

        self.robot.beam_to_joint_pos(
            self.robot.gotoCartPosQuatController.trajectory[-1]
        )
        # self.robot.gotoJointPosition(self.robot.init_qpos, duration=0.05)
        # self.robot.wait(duration=2.0)

        self.robot.gotoCartPositionAndQuat(
            desiredPos=initial_cart_position, desiredQuat=[0, 1, 0, 0], duration=0.5, log=False
        )

    def step(self, action, gripper_width=None, desired_vel=None, desired_acc=None):
        # observation, reward, done, _ = super().step(action, gripper_width, desired_vel=desired_vel, desired_acc=desired_acc)
        # self.success = self._check_early_termination()
        # mode, mean_distance = self.check_mode()
        j_pos = action[:7]
        # j_vel = action[7:14]
        gripper_width = action[-1]

        if gripper_width > 0.075:

            self.robot.open_fingers()

            # if self.gripper_flag == 0:
            #     print(0)
            #     self.robot.open_fingers()
            #     self.gripper_flag = 1
        else:
            self.robot.close_fingers(duration=0.0)
            # if self.gripper_flag == 1:
            #
            #     print(1)
            #     self.robot.close_fingers(duration=0.5)
            #     print(self.robot.set_gripper_width)
            #
            #     self.gripper_flag = 0

        # self.robot.set_gripper_width = gripper_width

        # c_pos, c_quat = self.robot.getForwardKinematics(action)
        # c_action = np.concatenate((c_pos, c_quat))

        # c_pos = action[:3]
        # c_quat = euler2quat(action[3:6])
        # c_action = np.concatenate((c_pos, c_quat))

        self.controller.setSetPoint(action[:-1])#, desired_vel=desired_vel, desired_acc=desired_acc)
        # self.controller.setSetPoint(action)#, desired_vel=j_vel, desired_acc=desired_acc)
        self.controller.executeControllerTimeSteps(
            self.robot, self.n_substeps, block=False
        )

        observation = self.get_observation()
        reward = self.get_reward()
        done = self.is_finished()

        for i in range(self.n_substeps):
            self.scene.next_step()

        debug_info = {}
        if self.debug:
            debug_info = self.debug_msg()

        self.env_step_counter += 1

        self.success = self._check_early_termination()
        mode_encoding, mean_distance = self.check_mode()

        mode = ''
        # print(f"Type of mode_encoding: {type(mode_encoding)}")
        mode = mode.join(mode_encoding)
        

        return observation, reward, done, {'mode': mode, 'success':  self.success, 'mean_distance': mean_distance}

    # def check_mode(self):
    #     box_1_pos = self.scene.get_obj_pos(self.picked_box)
    
    #     goal_1_pos = self.scene.get_obj_pos(self.target_box)
 

    #     dis_rr, _ = obj_distance(box_1_pos, goal_1_pos)
    #     visit = -1
    #     mode = -1

    #     if dis_rr <= self.target_min_dist and self.first_visit != 0:
    #         visit = 0
    #     # elif dis_rg <= self.target_min_dist and self.first_visit != 1:
    #     #     visit = 1
    #     # elif dis_gr <= self.target_min_dist and self.first_visit != 2:
    #     #     visit = 2
    #     # elif dis_gg <= self.target_min_dist and self.first_visit != 3:
    #     #     visit = 3

    #     if self.first_visit == -1:
    #         self.first_visit = visit
    #     # else:
    #     #     if self.first_visit == 0 and visit == 3:
    #     #         mode = 0  # rr -> gg
    #     #     elif self.first_visit == 3 and visit == 0:
    #     #         mode = 1  # gg -> rr
    #     #     elif self.first_visit == 1 and visit == 2:
    #     #         mode = 2  # rg -> gr
    #     #     elif self.first_visit == 2 and visit == 1:
    #     #         mode = 3  # gr -> rg

    #     mean_distance = dis_rr

    #     return mode, mean_distance
    def check_mode(self):
        # 获取盒子和目标的位置
        red_box_pos = self.scene.get_obj_pos(self.picked_box)[:2]  # 假设你只有一个 box
        target_pos = self.scene.get_obj_pos(self.target_box)[:2]

        # 计算盒子与目标之间的距离
        dist = np.linalg.norm(red_box_pos - target_pos)
        
        # 判断距离是否小于阈值
        if dist <= self.target_min_dist:
            self.mode_encoding.append('red-box')  
            self.min_inds.append(0)  

        return self.mode_encoding, dist


    def get_reward(self, if_sparse=False):
        return 0
        # if if_sparse:
        #     return 0

        # robot_pos = self.robot_state()[:2]

        # box_1_pos = self.scene.get_obj_pos(self.push_box1)
        # box_2_pos = self.scene.get_obj_pos(self.push_box2)
        # goal_1_pos = self.scene.get_obj_pos(self.target_box_1)
        # goal_2_pos = self.scene.get_obj_pos(self.target_box_2)

        # dis_robot_box_r, _ = obj_distance(robot_pos, box_1_pos[:2])
        # dis_robot_box_g, _ = obj_distance(robot_pos, box_2_pos[:2])

        # dis_rr, _ = obj_distance(box_1_pos, goal_1_pos)
        # dis_rg, _ = obj_distance(box_1_pos, goal_2_pos)
        # dis_gr, _ = obj_distance(box_2_pos, goal_1_pos)
        # dis_gg, _ = obj_distance(box_2_pos, goal_2_pos)

        # # reward for pushing red box to red target area.

        # # reward_red = 5 - (dis_robot_box_r + dis_rr)
        # # reward_red = reward_red if reward_red > 0 else 0

        # # reward_green = 100 - (dis_robot_box_g + dis_gg) * 3
        # # reward_green = reward_green if reward_green > 50 else 50

        # return (-1) * (dis_robot_box_r + dis_rr)

        # if dis_rr > self.target_min_dist:
        #     robot_factor = 1
        # else:
        #     robot_factor = 0
        #
        # return (-1) * (dis_robot_box_r * robot_factor + dis_rr - 2) + (-1) * (dis_robot_box_g * (1 - robot_factor) + dis_gg)

        # # reward for pushing two boxes
        # if dis_rr > self.target_min_dist:
        #
        #     self.reward_offset = dis_robot_box_g + dis_gg
        #
        #     return (-1) * (dis_robot_box_r + dis_rr)
        #
        # else:
        #     return (-1) * (dis_robot_box_g + dis_gg) + self.reward_offset

        dis_modes = np.array([dis_rr, dis_rg, dis_gr, dis_gg])

        min_ind = np.argmin(dis_modes)

        # four modes: [rr, gg], [rg, gr], [gr, rg], [gg, rr]
        min_dis = dis_modes[min_ind]
        if self.bp_mode is None and min_dis <= self.target_min_dist:
            self.bp_mode = min_ind

        if min_ind == 0 or min_ind == 3:
            return (-1) * (dis_rr + dis_gg)
        else:
            return (-1) * (dis_rg + dis_gr)

    def _check_early_termination(self) -> bool:
        # calculate the distance from end effector to object
        box_1_pos = self.scene.get_obj_pos(self.picked_box)[:2]
        # box_2_pos = self.scene.get_obj_pos(self.push_box2)
        goal_1_pos = self.scene.get_obj_pos(self.target_box)[:2]
        # goal_2_pos = self.scene.get_obj_pos(self.target_box_2)

        dis, _ = obj_distance(box_1_pos, goal_1_pos)
        # dis_rg, _ = obj_distance(box_1_pos, goal_2_pos)
        # dis_gr, _ = obj_distance(box_2_pos, goal_1_pos)
        # dis_gg, _ = obj_distance(box_2_pos, goal_2_pos)

        if dis <= self.target_min_dist:
            # terminate if end effector is close enough
            self.terminated = True
            return True

        return False

    def reset(self, random=True, context=None):
        self.terminated = False
        self.env_step_counter = 0
        self.episode += 1
        # self.first_visit = -1
        self.min_inds = []
        self.mode_encoding = []

        self.bp_mode = None
        obs = self._reset_env(random=random, context=context)

        return obs

    def _reset_env(self, random=True, context=None):

        if self.interactive:
            for log_name, s in self.cam_dict.items():
                s.reset()

            for log_name, s in self.log_dict.items():
                s.reset()

        self.scene.reset()
        self.robot.beam_to_joint_pos(self.robot.init_qpos)
        self.robot.open_fingers()

        self.manager.start(random=True, context=context)
        self.scene.next_step(log=False)

        observation = self.get_observation()

        return observation

        # if self.random_env:
        #     new_box1 = [self.push_box1, self.push_box1_space.sample()]
        #     new_box2 = [self.push_box2, self.push_box2_space.sample()]
        #
        #     self.scene.reset([new_box1, new_box2])
        # else:
        #     self.scene.reset()
