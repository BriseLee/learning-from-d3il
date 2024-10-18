
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
from environments.d3il.envs.gym_pushcube_env.gym_pushcube.envs.objects.pushcube_objects import get_obj_list

obj_list = get_obj_list()
pushed_box = obj_list[0]
target_box = obj_list[1]
platform = obj_list[2]


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
        

    def set_init(self):
        obj_list = get_obj_list()
        pushed_box = obj_list[0]
        target_box = obj_list[1]
        platform = obj_list[2]
        self.pushed_box = pushed_box
        self.target_box = target_box
        self.platform = platform
        for obj in [
            self.pushed_box,
            self.target_box,
            self.platform
          
        ]:
            self.scene.add_object(obj) 
           
           
        init_state = self.recorded_data["init_state"]
        robot_init_state = init_state.get("panda_robot")["joint_pos"]
        print(robot_init_state)
        red_box_init_state = init_state.get("pushed_box")["pos"]
        print(init_state)
        red_box_init_state_quat = init_state.get("pushed_box")["quat"]
        green_target_init_state = init_state.get("target_box")["pos"]
        green_target_init_state_quat = init_state.get("target_box")["quat"]

        # Extract robot data from recorded state
        if "state" not in self.recorded_data:
            raise KeyError("'state' key not found in recorded data. Available keys: {}".format(self.recorded_data.keys()))
        self.scene.start()
        # self.scene.set_obj_pos_and_quat(red_box_init_state, red_box_init_state_quat, obj_name="pushed_box")
        # self.scene.set_obj_pos_and_quat(green_target_init_state, green_target_init_state_quat, "target_box")
        self.robot.beam_to_joint_pos(robot_init_state)
        self.scene._set_obj_pos_and_quat( red_box_init_state, red_box_init_state_quat,self.pushed_box)
        self.scene._set_obj_pos_and_quat(green_target_init_state, green_target_init_state_quat,self.target_box)
        self.scene.set_obj_pos_and_quat(
            [0, 0, 0],
            [0, 0, 0, 1],
            self.platform,
        )
        # picked_box.set_position(red_box_init_state['pos'], red_box_init_state.get('quat', [1, 0, 0, 0]))
        # target_box.set_position(green_target_init_state['pos'], green_target_init_state.get('quat', [1, 0, 0, 0]))
        
        
        


    def replay(self):

        
        state_data = self.recorded_data["state"]
        print("Loaded state data keys:", state_data.keys())
        # self.scene.start()
        robot_data = state_data.get("panda_robot")
        pushed_box_data = state_data.get("pushed_box")
        target_box_data = state_data.get("target_box")
        if robot_data is None:
            raise KeyError("'panda_robot' key not found in 'state' data. Available keys: {}".format(state_data.keys()))

        # Extract relevant recorded data arrays
        j_pos_data = robot_data["des_j_pos"]
        # eef_pos_data = robot_data["gripper_width"]
        robot = "panda_robot"
        robot_c_pos = state_data[robot]['c_pos']
        robot_c_quat = state_data[robot]['c_quat']
        # gripper_width_data = robot_data["gripper_width"]
        pushed_box_pos = pushed_box_data["pos"]
        pushed_box_quat = pushed_box_data["quat"]

        target_box_pos = target_box_data["pos"]
        target_box_quat = target_box_data["quat"]
        # pushed_box_real_pos = self.scene.get_obj_pos(obj_name="pushed_box")
        # Make sure the number of data points is consistent
        # assert len(j_pos_data) == len(gripper_width_data), "Mismatch in recorded data lengths"
        pos_diff = []
        pos_real_diff = []
        # Replay loop
        for i in range(len(j_pos_data)):
            # Set robot joint positions and gripper width from recorded data
            # self.robot.gotoCartPositionAndQuat(robot_c_pos[i],robot_c_quat[i])
            self.robot.beam_to_joint_pos(j_pos_data[i])
            # Execute one step in the simulation
            # self.scene._set_obj_pos_and_quat(pushed_box_pos[i], pushed_box_quat[i],self.pushed_box)
            pushed_box_real_pos = self.scene.get_obj_pos(obj_name="pushed_box")
            # self.scene._set_obj_pos_and_quat( target_box_pos[i], target_box_quat[i],self.target_box)
            self.scene.next_step(log=False)
            time.sleep(0.01)  # Slow down replay for visualization
            pos_diff.append(np.linalg.norm(pushed_box_pos[i][:2] - target_box_pos[i][:2]))
            pos_real_diff.append(np.linalg.norm(pushed_box_real_pos[:2] - target_box_pos[i][:2]))
        pos_diff = np.array(pos_diff)
        pos_real_diff = np.array(pos_real_diff)
        print('minimum diff (x, y): ', np.min(pos_diff))
        print('minimum real diff (x, y): ', np.min(pos_real_diff))
        print('average diff: ', np.mean(pos_diff))

if __name__ == "__main__":
    # Set up the simulation factory and create a scene
    sim_factory = MjFactory()
    render_mode = Scene.RenderMode.HUMAN
    scene = sim_factory.create_scene(render=render_mode, dt=0.001)
    
    # Set up the robot in the scene
    robot = MjRobot(scene, xml_path="environments/d3il/models/mj/robot/panda_rod.xml")
    
    # Create a replay instance with the recorded data file path
    replay_instance = BlockPickReplay(scene, robot, recorded_data_path="environments/dataset/data/pushcube/push_data/feedback_withoutforce/state/PushCube_002.pkl")
    
    # Start the replay
    replay_instance.set_init()

    replay_instance.replay()
    