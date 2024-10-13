from typing import Dict

from alr_sim.controllers.Controller import JointPDController
from alr_sim.sims.mj_beta import MjRobot
from .sf_simulator import SFSimulator
from .sf_recorder import RecordData
from ..data_replayer import DataReplayer
from .task import sf_task_factory


class ReplayerJointController(JointPDController):
    def __init__(
        self,
        record_data: RecordData,
        robot: MjRobot,
        robot_name: str,
    ):
        super().__init__()
        self.joint_state = record_data.state[robot_name]["joint_pos"]
        self.gripper_state = record_data.state[robot_name]["gripper_width"]
        self.init_state = record_data.init_state[robot_name]
        self.sequence_length = record_data.header.sequence_length
        self.robot = robot
        self.index = 0

    def update_state(self):
        self.desired_joint_pos = self.joint_state[self.index]
        if self.gripper_state[self.index] <= 0.075:
            self.robot.close_fingers(duration=0)
        else:
            self.robot.open_fingers()
        self.index += 1

    def reset_robot(self):
        self.robot.beam_to_joint_pos(self.init_state)


class SFReplayer(DataReplayer):

    def __init__(self) -> None:
        self.controller_dict: Dict[str, ReplayerJointController] = {}
        super().__init__()

    def load_simulator(self, simulator: SFSimulator):
        self.simulator = simulator
        self.mj_scene = self.simulator.mj_scene
        init_state = self.record_data.init_state
        self.skip_step = self.record_data.header.skip_step
        init_object_state = {
            obj_name: init_state[obj_name]
            for obj_name in self.simulator.object_dict.keys()
        }
        self.simulator.reset_objects(init_object_state)
        init_robot_state = {
            robot_name: init_state[robot_name]
            for robot_name in self.simulator.robot_dict.keys()
        }
        print(f"state robot{init_robot_state}")
        self.simulator.reset_robots(init_robot_state)
        for robot_name, robot in self.simulator.robot_dict.items():
            self.controller_dict[robot_name] = ReplayerJointController(
                self.record_data, robot, robot_name,
            )
        self.simulator.assign_controller(self.controller_dict)
        return self.simulator

    def create_simulator(self, record_data_path: str) -> SFSimulator:
        self.load_recod_data(record_data_path)
        task_name = self.record_data.header.task
        simulator: SFSimulator = sf_task_factory(task_name)
        self.load_simulator(simulator)
        return simulator

    def replay(self):
        self.index = 0
        return super().replay()

    def replay_step(self):
        self.simulator.next_step()
        if self.index % self.skip_step == 0:
            for controller in self.controller_dict.values():
                controller.update_state()
        self.index += 1
