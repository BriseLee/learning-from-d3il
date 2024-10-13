import numpy as np
import pickle
import time
import sys
import os

# Add project root directory to sys.path
sys.path.append(os.path.abspath('/home/xueyinli/project/d3il'))

from environments.d3il.d3il_sim.sims.mj_beta.MjRobot import MjRobot
from environments.d3il.d3il_sim.sims.mj_beta.MjFactory import MjFactory
from environments.d3il.d3il_sim.core import Scene
from environments.d3il.envs.gym_picking_env.gym_picking.envs.objects.picking_objects import get_obj_list

obj_list, picked_box, target_box = get_obj_list()


class BlockPickReplay:
    def __init__(self, scene, robot, recorded_data_path, n_substeps=35):
        self.scene = scene
        self.robot = robot
        self.n_substeps = n_substeps
        
        # Load recorded data
        with open(recorded_data_path, "rb") as f:
            self.recorded_data = pickle.load(f)

        # Debug: print the keys of the recorded data to understand its structure
        print("Loaded recorded data keys:", self.recorded_data.keys())

    def init(self):
        self.picked_box = picked_box
        self.target_box = target_box
        # for obj in [
        #     self.picked_box,
            
        #     self.target_box,
          
        # ]:
        #     self.scene.add_object(obj)
        
        init_state = self.recorded_data["init_state"]
        robot_init_state = init_state.get("panda_robot")
        red_box_init_state = init_state.get("picked_box")["pos"]
        print(init_state)
        red_box_init_state_quat = init_state.get("picked_box")["quat"]
        green_target_init_state = init_state.get("target_box")["pos"]
        green_target_init_state_quat = init_state.get("target_box")["quat"]
        # Extract robot data from recorded state
        if "state" not in self.recorded_data:
            raise KeyError("'state' key not found in recorded data. Available keys: {}".format(self.recorded_data.keys()))
        self.scene.add_object(picked_box)
        self.scene.add_object(target_box)
        self.scene.set_obj_pos_and_quat(red_box_init_state, red_box_init_state_quat, sim_obj='picked_box')
        self.scene.set_obj_pos_and_quat(green_target_init_state, green_target_init_state_quat, "target_box")
        # picked_box.set_position(red_box_init_state['pos'], red_box_init_state.get('quat', [1, 0, 0, 0]))
        # target_box.set_position(green_target_init_state['pos'], green_target_init_state.get('quat', [1, 0, 0, 0]))
        
        
        


    def replay(self):

        
        state_data = self.recorded_data["state"]
        print("Loaded state data keys:", state_data.keys())
        self.scene.start()
        robot_data = state_data.get("panda_robot")
        if robot_data is None:
            raise KeyError("'panda_robot' key not found in 'state' data. Available keys: {}".format(state_data.keys()))

        # Extract relevant recorded data arrays
        j_pos_data = robot_data["des_j_pos"]
        gripper_width_data = robot_data["gripper_width"]
        
        # Make sure the number of data points is consistent
        assert len(j_pos_data) == len(gripper_width_data), "Mismatch in recorded data lengths"
        
        # Replay loop
        for i in range(len(j_pos_data)):
            # Set robot joint positions and gripper width from recorded data
            self.robot.beam_to_joint_pos(j_pos_data[i])
            if gripper_width_data[i] > 0.075:
                self.robot.open_fingers()
            else:
                self.robot.close_fingers(duration=0.0)

            # Execute one step in the simulation
            self.scene.next_step(log=False)
            time.sleep(0.01)  # Slow down replay for visualization


if __name__ == "__main__":
    # Set up the simulation factory and create a scene
    sim_factory = MjFactory()
    render_mode = Scene.RenderMode.HUMAN
    scene = sim_factory.create_scene(render=render_mode, dt=0.001)
    
    # Set up the robot in the scene
    robot = MjRobot(scene, xml_path="environments/d3il/models/mj/robot/panda_invisible.xml")
    
    # Create a replay instance with the recorded data file path
    replay_instance = BlockPickReplay(scene, robot, recorded_data_path="environments/dataset/data/picking/pick_data/feedback_withoutforce/state/PickandPlaceBox_002.pkl")
    
    # Start the replay
    replay_instance.init()

    replay_instance.replay()