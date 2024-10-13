import os
import numpy as np
from typing import Dict, Callable

from ..recorder import Recorder, RecordData, RecordDataHeader
from ..recorder import ObjectData
from .sf_simulator import SFSimulator
# from task.contact_force import Collision_finger


class SFRecorder(Recorder):

    def __init__(
        self,
        sf_simulator: SFSimulator,
        save_root_path="./SFDemoData/",
        # record_mode: bool = False,
        record_type: str = "demonstration",
        skip_step: int = 10,
    ) -> None:
        self.mj_scene = sf_simulator.mj_scene
        self.task_name = sf_simulator.task_name
        super().__init__(self.task_name, save_root_path, skip_step)
        self.object_dict = sf_simulator.object_dict
        self.robot_dict = sf_simulator.robot_dict
        self.record_data: RecordData = None
        self.record_type = record_type
        self.skip_step = skip_step
        self.dt = self.mj_scene.dt
        self.record_func: Dict[str, Callable[[], ObjectData]] = {}

    def add_record_func(self, name: str, func: Callable[[], ObjectData]):
        if self.on_recording:
            raise ValueError("Cannot add record function while recording")
        self.record_func[name] = func

    def _record(self):
        current_frame = {}
        for name, robot in self.robot_dict.items():
            current_frame[name] = {
                "joint_pos": robot.current_j_pos,
                "joint_vel": robot.current_j_vel,
                "des_joint_pos": robot.des_joint_pos,
                "des_joint_vel": robot.des_joint_vel,
                "des_joint_acc": robot.des_joint_acc,
                "cart_pos": robot.current_c_pos,
                "cart_quat": robot.current_c_quat,
                "cart_vel": robot.current_c_vel,
                "cart_quat_vel": robot.current_c_quat_vel,
                "des_c_pos": robot.des_c_pos,
                "des_c_vel": robot.des_c_vel,
                "des_quat": robot.des_quat,
                "des_quat_vel": robot.des_quat_vel,
                "gripper_width": np.array([robot.gripper_width]),
            }
        for name, obj in self.object_dict.items():
            current_frame[name] = {
                "pos": self.mj_scene.get_obj_pos(obj_name=obj.name),
                "quat": self.mj_scene.get_obj_quat(obj_name=obj.name),
            }
        # for name1, name2, force, position in self.contact_force_dict.items():
        #     current_frame[name1] = {
        #             "force": self.mj_scene.get_obj_pos(obj_name=obj.name),
        #             "avg_pos": self.mj_scene.get_obj_quat(obj_name=obj.name),
        #         }
        
        for name, func in self.record_func.items():
            current_frame[name] = func()
        self.record_data.append_frame(current_frame)

    def _start_record(self):
        record_header = RecordDataHeader(
            self.record_type,
            "SimulationFramework/MjBeta",
            self.task_name,
            self.skip_step,
            self.dt,
            0,
        )
        record_data = RecordData(record_header)
        init_state = record_data.init_state
        for obj_name, obj in self.object_dict.items():
            init_state[obj_name] = {
                "pos": self.mj_scene.get_obj_pos(obj_name=obj.name),
                "quat": self.mj_scene.get_obj_quat(obj_name=obj.name),
            }
        for robot_name, robot in self.robot_dict.items():
            init_state[robot_name] = {
                "joint_pos": robot.current_j_pos,
            }
        self.record_data = record_data

    def _save_record(self, file_name):
        # in case of data overwriting
        record_data = self.record_data
        record_data.save_to_file(os.path.join(self.save_path, file_name))
        super()._save_record(file_name)
