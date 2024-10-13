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


class PutIteminDrawer(SFSimulator):

    def __init__(self, host_address=None):
        self.box_space = SamplingSpace(
            low=np.array([0.3, -0.3, 0]),
            high=np.array([0.6, 0.3, 0]),
            seed=np.random.randint(0, 1000),
        )
        super().__init__('PutIteminDrawer',host_address)
        self.haptic_on_right: bool = False
        self.haptic_on_left: bool = False
        self.haptic_aim_right = False
        self.haptic_aim_left = False

    def create_robots(self) -> Dict[str, MjRobot]:
        self.pick_robot1 = self.sim_factory.create_robot(
            self.mj_scene,
            xml_path=sim_framework_path("./models/mj/robot/panda.xml"),
            base_position = [0.0, 0.35, 0.0]
        )
        self.pick_robot2 = self.sim_factory.create_robot(
            self.mj_scene,
            xml_path=sim_framework_path("./models/mj/robot/panda.xml"),
            base_position = [0.0, -0.35, 0.0]
        )
        return {"panda_robot1": self.pick_robot1, 
                "panda_robot2": self.pick_robot2}

    def create_objects(self) -> Dict[str, MujocoObject]:
        self.picked_box = CustomMujocoObject(
            object_name="picked_box",
            object_dir_path=os.path.dirname(os.path.abspath(__file__)),
            pos=[0.4, 0.3, 0],
            quat=[0, 0, 0, 1],
        )
        self.target_box = CustomMujocoObject(
            object_name="target_box",
            object_dir_path=os.path.dirname(os.path.abspath(__file__)),
            pos=[0.4, -0.3, 0.0],
            quat=[0, 0, 0, 1],
        )
        return {
            "picked_box": self.picked_box,
            "target_box": self.target_box,
        }

                   

    def reset(self):
        # return
        picked_box_pos = self.box_space.sample()
        picked_box_pos = (picked_box_pos[0],picked_box_pos[1],0)
        picked_box_euler = np.array([np.random.uniform(-180, 180), 0, 0])
        picked_box_quat = R.from_euler("xyz", picked_box_euler).as_quat()
        self.mj_scene.set_obj_pos_and_quat(
            new_pos=picked_box_pos,
            new_quat=picked_box_quat,
            obj_name="picked_box",
        )
        target_box_pos= picked_box_pos+np.array([0,0,0.06])
        target_box_quat = R.from_euler("xyz", picked_box_euler).as_quat()
        self.mj_scene.set_obj_pos_and_quat(
            new_pos=target_box_pos,
            new_quat=target_box_quat,
            obj_name="target_box",
        )
       
        self.pick_robot1.gotoCartPositionAndQuat(
            desiredPos=[picked_box_pos[0]+0.1, picked_box_pos[1]+0.15, 0.35],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        
        self.pick_robot2.gotoCartPositionAndQuat(
            desiredPos=[target_box_pos[0]+0.1, target_box_pos[1]-0.15, 0.35],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        self.pick_robot1.gotoCartPositionAndQuat(
            desiredPos=[picked_box_pos[0]+0.1, picked_box_pos[1]+0.15, 0.3],
            desiredQuat=[0, 1, 0, 0],
            duration=1.0,
        )
        self.pick_robot2.gotoCartPositionAndQuat(
            desiredPos=[target_box_pos[0]+0.1, target_box_pos[1]-0.15, 0.3],
            desiredQuat=[0, 1, 0, 0],
            duration=1.0,
        )
        
        
        self.pick_robot1.activeController = self.controller_dict["panda_robot1"]
        self.pick_robot2.activeController = self.controller_dict["panda_robot2"]


        self.pick_robot1.activeController.setSetPoint(
            np.hstack((self.pick_robot1.current_c_pos, [0, 1, 0, 0]))
        )

        self.pick_robot2.activeController.setSetPoint(
            np.hstack((self.pick_robot2.current_c_pos, [0, 1, 0, 0]))
        )


    def before_step(self):
        pass
    def after_step(self):
        pass


class PutIteminDrawerVibration(PutIteminDrawer):

    def __init__(self, host_address=None):
        super().__init__(host_address)
        # self.haptic_on: bool = False

    def after_step(self):
               
        replace_rb0_l=0
        replace_rb0_r=0
        replace_rb0_l, replace_rb0_r  = get_collisions(
        self.mj_scene,
        target_pairs1={
            ('picked_handle', 'finger1_rb0_tip_collision'),
            ('picked_handle_short', 'finger1_rb0_tip_collision'),
            },
        target_pairs2={
            ('picked_handle', 'finger2_rb0_tip_collision'),
            ('picked_handle_short', 'finger2_rb0_tip_collision'),
            }
        )
        replace_rb1_l=0
        replace_rb1_r=0
        replace_rb1_l, replace_rb1_r = get_collisions(
        self.mj_scene,
        target_pairs1={
            ('picked_item', 'finger1_rb1_tip_collision'),
            },
        target_pairs2={
            ('picked_item', 'finger2_rb1_tip_collision'),
            }
        )
        replace_aim=0 
        replace_aim= aim_resultant_force(
        self.mj_scene,
        target_pairs={
                ('picked_item', 'target_box0'),
                ('picked_item','target_box1'),
                ('picked_item', 'target_box2'),
                ('picked_item', 'target_box4'),
        },
        )
        if replace_rb0_l != 0 or replace_rb0_r !=0:
            self.haptic_on_right = True
            if replace_aim != 0:
                self.haptic_aim_right = True
            else:
                self.haptic_aim_right = False
        else:
            self.haptic_on_right = False
            self.haptic_aim_right = False

        if replace_rb1_l != 0 or replace_rb1_r !=0:
            self.haptic_on_left = True
            if replace_aim != 0:
                self.haptic_aim_left = True
            else:
                self.haptic_aim_left = False
        else:
            self.haptic_on_left = False
            self.haptic_aim_left = False
 
