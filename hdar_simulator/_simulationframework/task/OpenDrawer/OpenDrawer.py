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

class OpenDrawer(SFSimulator):

    def __init__(self,host_address=None):
        self.box_space = SamplingSpace(
            low=np.array([0.3, -0.3, 0]),
            high=np.array([0.6, 0.3, 0]),
            seed=np.random.randint(0, 1000),
        )
        super().__init__('OpenDrawer', host_address)
        self.haptic_on: bool = False
        self.haptic_aim: bool = False

    def create_robots(self) -> Dict[str, MjRobot]:
        self.pick_robot = self.sim_factory.create_robot(
            self.mj_scene,
            xml_path=sim_framework_path("./models/mj/robot/panda.xml"),
        )
        return {"panda_robot": self.pick_robot}

    def create_objects(self) -> Dict[str, MujocoObject]:
        self.picked_box = CustomMujocoObject(
            object_name="picked_box",
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
            "pick_box": self.picked_box,
            "target_box": self.target_box,
        }


    def reset(self):
        # return
        # pushed_box_pos = self.box_space.sample()
        picked_box_pos = [np.random.uniform(0.3, 0.5),np.random.uniform(-0.4, 0.4),0]
        picked_box_euler = np.array([np.random.uniform(-90, -45), 0, 0])
        # pushed_box_euler = np.array([-20, 0, 0])
        picked_box_quat = R.from_euler("xyz", picked_box_euler).as_quat()
        self.mj_scene.set_obj_pos_and_quat(
            new_pos=picked_box_pos,
            new_quat=picked_box_quat,
            obj_name="picked_box",
        )
        
        target_box_quat = R.from_euler("xyz", picked_box_euler).as_quat()
        target_box_pos= picked_box_pos+np.array([0,0,0.06])
        self.mj_scene.set_obj_pos_and_quat(
            new_pos=target_box_pos,
            new_quat=target_box_quat,
            obj_name="target_box",
        )
        self.pick_robot.gotoCartPositionAndQuat(
            desiredPos=[self.picked_box.pos[0]-0.05, self.picked_box.pos[1], 0.4],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        self.pick_robot.gotoCartPositionAndQuat(
            desiredPos=[self.picked_box.pos[0]-0.05, self.picked_box.pos[1], 0.3],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        self.pick_robot.activeController = self.controller_dict["panda_robot"]
        self.pick_robot.activeController.setSetPoint(
            np.hstack((self.pick_robot.current_c_pos_global, [0, 1, 0, 0]))
        )
    def before_step(self):
        pass

    def after_step(self):
        pass


class OpenDrawerVibration(OpenDrawer):

    def __init__(self, host_address=None):
        super().__init__(host_address)
        

    def after_step(self):
               
        replace_rb0_l=0
        replace_rb0_r=0
        replace_rb0_l, replace_rb0_r  = get_collisions(
        self.mj_scene,
        target_pairs1={
            ('picked_handle1', 'finger1_rb0_tip_collision'),
            ('picked_handle2', 'finger1_rb0_tip_collision'),
            ('picked_handle', 'finger1_rb0_tip_collision'),
        },
        target_pairs2={
            ('picked_handle1', 'finger2_rb0_tip_collision'),
            ('picked_handle2', 'finger1_rb0_tip_collision'),
            ('picked_handle', 'finger2_rb0_tip_collision'),
        }
        )
        if (replace_rb0_l != 0 or replace_rb0_r !=0) :
            self.haptic_on = True
        else:
            self.haptic_on = False


                  