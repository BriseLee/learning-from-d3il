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


class BoxPushing(SFSimulator):

    def __init__(self, host_address=None):
        self.box_space = SamplingSpace(
            low=np.array([0.3, -0.3, 0]),
            high=np.array([0.6, 0.3, 0]),
            seed=np.random.randint(0, 1000),
        )
        super().__init__('BoxPushing', host_address)
        self.haptic_on: bool = False
        self.haptic_aim: bool = False



    def create_robots(self) -> Dict[str, MjRobot]:
        self.push_robot = self.sim_factory.create_robot(
            self.mj_scene,
            xml_path=sim_framework_path("./models/mj/robot/panda_rod.xml"),
        )
        return {"panda_robot": self.push_robot}

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
        current_pos = self.push_robot.current_c_pos
        self.push_robot.gotoCartPositionAndQuat(
            desiredPos=[current_pos[0], current_pos[1], 0.25],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        self.push_robot.gotoCartPositionAndQuat(
            desiredPos=[pushed_box_pos[0], pushed_box_pos[1], 0.25],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        self.push_robot.gotoCartPositionAndQuat(
            desiredPos=[pushed_box_pos[0], pushed_box_pos[1], 0.13],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        inital_pos = np.array([pushed_box_pos[0], pushed_box_pos[1], 0.13])
        self.push_robot.activeController = self.controller_dict["panda_robot"]
        self.push_robot.activeController.setSetPoint(
            np.hstack((inital_pos, np.array([0, 1, 0, 0])))
        )

    def before_step(self):
        pass

    def after_step(self):
        pass
