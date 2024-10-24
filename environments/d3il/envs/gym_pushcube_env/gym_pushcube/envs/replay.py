
import numpy as np
import pickle
import time
import sys
import os
import matplotlib.pyplot as plt


# Add project root directory to sys.path
sys.path.append(os.path.abspath('/home/xueyinli/project/d3il'))
from environments.dataset.geo_transform import quat2euler
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
        self.skip_step = 10
        self.index = 0
        self.eef_xy_trajectory = []
        self.eef_real_xy_trajectory = []
        self.pushed_box_xy_trajectory = []
        self.push_real_box_xy_trajectory = []
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
        self.red_box_init_state = red_box_init_state
        self.green_target_init_state = green_target_init_state
        state_data = self.recorded_data["state"]
        pushed_box_data = state_data.get("pushed_box")
        
        print(f"push box init pos:{pushed_box_data}")

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
        return self.red_box_init_state,self.green_target_init_state
        
    def replay(self):
        # Step the simulation
        self.scene.next_step()
        if self.index % self.skip_step == 0:
            
            state_data = self.recorded_data["state"]
            
            robot_data = state_data.get("panda_robot")
            pushed_box_data = state_data.get("pushed_box")
            pushed_box_pos = pushed_box_data["pos"]
            pushed_box_quat = pushed_box_data["quat"]
            target_box_data = state_data.get("target_box")
            target_box_pos = target_box_data["pos"]
            target_box_quat = target_box_data["quat"]


            # Update robot and object positions
            if robot_data is not None:
                step_index = self.index // self.skip_step
                j_pos_data = robot_data["des_j_pos"]
                des_c_pos = robot_data["des_c_pos"]
                # print(f"des c pos: {des_c_pos}")
                pushed_box_pos = pushed_box_data["pos"][self.index // self.skip_step]
                pushed_box_quat = pushed_box_data["quat"][self.index // self.skip_step]
                print(f"pushed box pos {pushed_box_pos}")
                target_box_pos = target_box_data["pos"][self.index // self.skip_step]

                self.robot.beam_to_joint_pos(j_pos_data[self.index // self.skip_step]) 
                robot_c_pos = robot_data["c_pos"]
                robot_real_c_pos = robot.current_c_pos
                robot_c_real_pos = robot.des_c_pos


                self.eef_xy_trajectory.append(robot_c_pos[step_index][:2])
                self.eef_real_xy_trajectory.append(robot_real_c_pos[:2])
                self.pushed_box_xy_trajectory.append(pushed_box_data["pos"][step_index][:2])
                pushed_box_real_pos = self.scene.get_obj_pos(obj_name="pushed_box")
                pushed_box_real_quat = self.scene.get_obj_quat(obj_name="pushed_box")
                target_box_real_quat = self.scene.get_obj_quat(obj_name="target_box")
                self.push_real_box_xy_trajectory.append(pushed_box_real_pos[:2])   
                box_goal_pos_dist = np.linalg.norm(pushed_box_pos[:2] - target_box_pos[:2])
                # if box_goal_pos_dist < 0.01:
                #     print(f"perfect fit {box_goal_pos_dist}")
                # else : 
                #     print(f"nonono:{{box_goal_pos_dist}}")
                box_quat = quat2euler(self.scene.get_obj_quat(self.pushed_box))
                print(f"box quat: {box_quat}")
                print(f"obj recorded quat{pushed_box_quat},obj real quat{pushed_box_real_quat},target quat{target_box_real_quat}")
        self.index += 1    
        
    def trajectory(self):
        eef_xy_trajectory = np.array(self.eef_xy_trajectory)
        eef_real_xy_trajectory = np.array(self.eef_real_xy_trajectory)
        pushed_box_xy_trajectory = np.array(self.pushed_box_xy_trajectory)
        push_real_box_xy_trajectory = np.array(self.push_real_box_xy_trajectory)

        plt.figure()
        plt.plot(eef_xy_trajectory[:, 0], eef_xy_trajectory[:, 1], label='EEF trajectory', linestyle='--', linewidth=1.5)
        plt.plot(eef_real_xy_trajectory[:, 0], eef_real_xy_trajectory[:, 1], label='EEF real trajectory', linestyle=':', linewidth=1.5)
        plt.plot(pushed_box_xy_trajectory[:, 0], pushed_box_xy_trajectory[:, 1], label='pushed box', linewidth=2)
        plt.plot(push_real_box_xy_trajectory[:, 0], push_real_box_xy_trajectory[:, 1], label='pushed box (real)', linestyle='-.', linewidth=2)
        plt.scatter(self.red_box_init_state[0], self.red_box_init_state[1], color='orange', label='init push box pos', marker='x', s=100)
        plt.scatter(self.green_target_init_state[0], self.green_target_init_state[1], color='green', label='target pos', marker='x', s=100)
        plt.xlabel('X pos (m)')
        plt.ylabel('Y pos (m)')
        plt.title('XY pos')
        plt.legend()
        plt.grid(True)
        plt.show()


   


if __name__ == "__main__":
    # Set up the simulation factory and create a scene
    sim_factory = MjFactory()
    render_mode = Scene.RenderMode.HUMAN
    scene = sim_factory.create_scene(render=render_mode, dt=0.001)
    
    # Set up the robot in the scene
    robot = MjRobot(scene, xml_path="environments/d3il/models/mj/robot/panda_rod.xml")
    
    # Create a replay instance with the recorded data file path
    replay_instance = BlockPickReplay(scene, robot, recorded_data_path="environments/dataset/data/pushcube/push_data/feedback_withoutforce/state/PushCube_005.pkl")
    
    # Start the replay
    replay_instance.set_init()

    while replay_instance.index < len(replay_instance.recorded_data["state"]["panda_robot"]["des_j_pos"]) * replay_instance.skip_step:
        replay_instance.replay()
    replay_instance.trajectory()
    