from typing import Dict
from scipy.spatial.transform import Rotation as R
import numpy as np
import os
from gym.spaces import Box as SamplingSpace

from ...sf_simulator import SFSimulator
from alr_sim.utils.sim_path import sim_framework_path

from alr_sim.sims.mj_beta import MjRobot
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import MujocoObject
from alr_sim.sims.mj_beta.mj_utils.mj_scene_object import CustomMujocoObject

from ..Collision_finger import get_collisions , aim_resultant_force
import abc


class PushCube(SFSimulator):

    def __init__(self, host_address=None):
        self.box_space = SamplingSpace(
            low=np.array([0.3, -0.3, 0]),
            high=np.array([0.6, 0.3, 0]),
            seed=np.random.randint(0, 1000),
        )
        super().__init__('PushCube',host_address)
        self.haptic_on: bool = False
        self.haptic_aim = False

    def create_robots(self) -> Dict[str, MjRobot]:
        self.pick_robot = self.sim_factory.create_robot(
            self.mj_scene,
            xml_path=sim_framework_path("./models/mj/robot/panda.xml"),
        )
        return {"panda_robot": self.pick_robot}

    def create_objects(self) -> Dict[str, MujocoObject]:
        self.pushed_box = CustomMujocoObject(
            object_name="pushed_box",
            object_dir_path=os.path.dirname(os.path.abspath(__file__)),
            pos=[0.4, 0, 0.3],
            quat=[0, 0, 0, 1],
        )
        self.target_box = CustomMujocoObject(
            object_name="target_box",
            object_dir_path=os.path.dirname(os.path.abspath(__file__)),
            pos=[0.4, 0.3, 0.0],
            quat=[0, 0, 0, 1],
        )
        return {
            "pushed_box": self.pushed_box,
            "target_box": self.target_box,
        }

                   

    def reset(self):
        # return
        pushed_box_pos = self.box_space.sample()
        pushed_box_euler = np.array([np.random.uniform(-180, 180), 0, 0])
        pushed_box_quat = R.from_euler("xyz", pushed_box_euler).as_quat()
        self.mj_scene.set_obj_pos_and_quat(
            new_pos=pushed_box_pos,
            new_quat=pushed_box_quat,
            obj_name="pushed_box",
        )
        while True:
            target_box_pos = self.box_space.sample()
            if np.linalg.norm(
                np.array(target_box_pos) - np.array(pushed_box_pos)
            ) > 0.2:
                break
        target_box_euler = np.array([np.random.uniform(-180, 180), 0, 0])
        target_box_quat = R.from_euler("xyz", target_box_euler).as_quat()
        self.mj_scene.set_obj_pos_and_quat(
            new_pos=target_box_pos,
            new_quat=target_box_quat,
            obj_name="target_box",
        )
        self.pick_robot.gotoCartPositionAndQuat(
            desiredPos=[pushed_box_pos[0]-0.1, pushed_box_pos[1], 0.6],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        self.pick_robot.gotoCartPositionAndQuat(
            desiredPos=[pushed_box_pos[0]-0.1, pushed_box_pos[1], 0.2],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        self.pick_robot.activeController = self.controller_dict["panda_robot"]
        self.pick_robot.activeController.setSetPoint(
            np.hstack((self.pick_robot.current_c_pos, [0, 1, 0, 0]))
        )

    def before_step(self):
        pass
    def after_step(self):
        pass


class PushCubeVibration(PushCube):

    def __init__(self, host_address=None):
        super().__init__(host_address)
        # self.haptic_on: bool = False

    def after_step(self):
               
        replace_rb0_l=0
        replace_rb0_r=0
        replace_rb0_l, replace_rb0_r  = get_collisions(
        self.mj_scene,
        target_pairs1={
            ('pushed_box_a', 'finger1_rb0_tip_collision'),
            ('pushed_box_b', 'finger1_rb0_tip_collision'),
                       },
        target_pairs2={
            ('pushed_box_a', 'finger2_rb0_tip_collision'),
            ('pushed_box_b', 'finger2_rb0_tip_collision'),
            }
        )
        if (replace_rb0_l != 0 or replace_rb0_r !=0) :
            self.haptic_on = True
        else:
            self.haptic_on = False

