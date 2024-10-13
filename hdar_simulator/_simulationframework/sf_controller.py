import numpy as np
from alr_sim.sims.mj_beta import MjRobot
from scipy.spatial.transform import Rotation as R

from simpub.xr_device.meta_quest3 import MetaQuest3
from alr_sim.controllers.IKControllers import CartPosQuatImpedenceController


class MetaQuest3Controller(CartPosQuatImpedenceController):

    def __init__(
        self,
        device: MetaQuest3,
        hand: str = "right",
        fix_rotation: bool = False,
        fix_z: bool = False,
        with_gripper: bool = True
    ):
        super().__init__()
        self.device: MetaQuest3 = device
        self.hand = hand
        self.fix_rotation = fix_rotation
        self.fix_z = fix_z
        self.with_gripper = with_gripper
        self.on_control = False
        self.start_pos_offset = None

    def getControl(self, robot: MjRobot):
        input_data = self.device.get_input_data()
        if input_data is None:
            return super().getControl(robot)
        hand = input_data[self.hand]
        if hand["hand_trigger"] is False:
            self.on_control = False
            return super().getControl(robot)
        # initial position
        if self.on_control is False:
            self.start_pos_offset = (
                robot.current_c_pos_global - np.array(hand["pos"])
            )
            self.on_control = True
        # pos and quat offsets
        desired_pos = np.array(hand["pos"]) + self.start_pos_offset
        if self.fix_rotation:
            desired_quat_local = np.array([0, 1, 0, 0])
        else:
            desired_quat = hand["rot"]
            rot = R.from_quat(desired_quat) * R.from_euler(
                "xyz", [-180, 0, 180], True
            )
            desired_quat = rot.as_quat(scalar_first=True)
            desired_quat_local = robot._localize_cart_quat(desired_quat)
        if self.with_gripper:
            if hand["index_trigger"]:
                robot.close_fingers(duration=0.0)
            else:
                robot.open_fingers()
        desired_pos_local = robot._localize_cart_pos(desired_pos)
        if self.fix_z:
            desired_pos_local[2] = robot.des_c_pos[2]
        self.setSetPoint(np.hstack((desired_pos_local, desired_quat_local)))
        return super().getControl(robot)
