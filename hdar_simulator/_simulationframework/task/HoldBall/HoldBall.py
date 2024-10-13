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


class HoldBall(SFSimulator):

    def __init__(self, host_address=None):
        self.box_space = SamplingSpace(
            low=np.array([0.3, -0.3, 0]),
            high=np.array([0.6, 0.3, 0]),
            seed=np.random.randint(0, 1000),
        )
        super().__init__('HoldBall', host_address)
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
        self.picked_Sphere = CustomMujocoObject(
            object_name="picked_Sphere",
            object_dir_path=os.path.dirname(os.path.abspath(__file__)),
            pos=[0.4, 0.0, 0.3],
            quat=[0, 0, 0, 1],
        )
        self.target_Sphere = CustomMujocoObject(
            object_name="target_Sphere",
            object_dir_path=os.path.dirname(os.path.abspath(__file__)),
            pos=[0.4, -0.0, 0.0],
            quat=[0, 0, 0, 1],
        )
        return {
            "pick_Sphere": self.picked_Sphere,
            "target_Sphere": self.target_Sphere,
        }

    def reset(self):
        # return
        while True:
            pushed_box_pos = self.box_space.sample()
            if 0.2 < np.linalg.norm(
                [0,0,0] - np.array(pushed_box_pos)
            ) < 0.4:
                break
        pushed_box_euler = np.array([np.random.uniform(-180, 180), 0, 0])
        pushed_box_quat = R.from_euler("xyz", pushed_box_euler).as_quat()
        self.mj_scene.set_obj_pos_and_quat(
            new_pos=pushed_box_pos,
            new_quat=pushed_box_quat,
            obj_name="picked_Sphere",
        )
        target_box_pos= pushed_box_pos+np.array([0,0,0.2])
        target_box_euler = np.array([np.random.uniform(-180, 180), 0, 0])
        target_box_quat = R.from_euler("xyz", target_box_euler).as_quat()
        self.mj_scene.set_obj_pos_and_quat(
            new_pos=target_box_pos,
            new_quat=target_box_quat,
            obj_name="target_Sphere",
        )
        self.pick_robot1.gotoCartPositionAndQuat(
            desiredPos=[0.4, 0.2, 0.3],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        
        self.pick_robot2.gotoCartPositionAndQuat(
            desiredPos=[0.4, -0.2, 0.3],
            desiredQuat=[0, 1, 0, 0],
            duration=2.0,
        )
        self.pick_robot1.gotoCartPositionAndQuat(
            desiredPos=[0.4, 0.2, 0.2],
            desiredQuat=[0, 1, 0, 0],
            duration=1.0,
        )
        self.pick_robot2.gotoCartPositionAndQuat(
            desiredPos=[0.4, -0.2, 0.2],
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


class HoldBallVibration(HoldBall):
    def __init__(self, host_address=None):
        super().__init__(host_address)

    def after_step(self):
        
        replace_rb0_l=0
        replace_rb0_r=0
        replace_rb0_l, replace_rb0_r  = get_collisions(
        self.mj_scene,
        target_pairs1={
            ('picked_Sphere', 'finger1_rb0_tip_collision'),
            
                       },
        target_pairs2={
            ('picked_Sphere', 'finger2_rb0_tip_collision'),
          
            }
        )
        replace_rb1_l=0
        replace_rb1_r=0
        replace_rb1_l, replace_rb1_r  = get_collisions(
        self.mj_scene,
        target_pairs1={
            ('picked_Sphere', 'finger1_rb1_tip_collision'),
            
                       },
        target_pairs2={
            ('picked_Sphere', 'finger2_rb1_tip_collision'),
          
            }
        )

        if replace_rb0_l != 0 or replace_rb0_r !=0:
            self.haptic_on_right = True
            
        else:
            self.haptic_on_right = False
            
        if replace_rb1_l != 0 or replace_rb1_r !=0:
            self.haptic_on_left = True
            
        else:
            self.haptic_on_left = False

