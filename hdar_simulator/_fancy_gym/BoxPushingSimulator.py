import numpy as np
from simpub.xr_device.meta_quest3 import MetaQuest3
import time
import pinocchio as pin
import os
import mujoco

from fancy_gym.envs.mujoco.box_pushing.box_pushing_env import BoxPushingEnvBase
from .._simulationframework.sf_simulator import FancyGemSimulator


class PandaPDController:
    # Currentlly only for Panda robot
    def __init__(self):
        self.pgain = np.array([120.0, 120.0, 120.0, 120.0, 50.0, 30.0, 10.0]) * 0.1
        self.dgain = np.array([1.0, 10.0, 10.0, 10.0, 6.0, 5.0, 3.0]) * 0.1
        # self.pgain = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
        # self.dgain = np.array([0.1, 1.0, 1.0, 1.0, 0.6, 0.1, 0.1])

    def get_control(self, target_j_pos, current_j_pos, current_j_vel):
        qd_d = target_j_pos - current_j_pos
        vd_d = - current_j_vel
        return self.pgain * qd_d + self.dgain * vd_d

    def calculateOfflineIK(self, desired_cart_pos, desired_cart_quat):
        """
        calculate offline inverse kinematics for franka pandas
        :param desired_cart_pos: desired cartesian position of tool center point
        :param desired_cart_quat: desired cartesian quaternion of tool center point
        :return: joint angles
        """
        J_reg = 1e-6
        w = np.diag([1, 1, 1, 1, 1, 1, 1])
        target_theta_null = np.array([
            3.57795216e-09,
            1.74532920e-01,
            3.30500960e-08,
            -8.72664630e-01,
            -1.14096181e-07,
            1.22173047e00,
            7.85398126e-01])
        eps = 1e-5          # threshold for convergence
        IT_MAX = 1000
        dt = 1e-3
        i = 0
        pgain = [
            33.9403713446798,
            30.9403713446798,
            33.9403713446798,
            27.69370238555632,
            33.98706171459314,
            30.9185531893281,
        ]
        pgain_null = 5 * np.array([
            7.675519770796831,
            2.676935478437176,
            8.539040163444975,
            1.270446361314313,
            8.87752182480855,
            2.186782233762969,
            4.414432577659688,
        ])
        pgain_limit = 20
        q = self.data.qpos[:7].copy()
        qd_d = np.zeros(q.shape)
        old_err_norm = np.inf

        while True:
            q_old = q
            q = q + dt * qd_d
            q = np.clip(q, q_min, q_max)
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)
            current_cart_pos = self.data.body("tcp").xpos.copy()
            current_cart_quat = self.data.body("tcp").xquat.copy()

            cart_pos_error = np.clip(desired_cart_pos - current_cart_pos, -0.1, 0.1)

            if np.linalg.norm(current_cart_quat - desired_cart_quat) > np.linalg.norm(current_cart_quat + desired_cart_quat):
                current_cart_quat = -current_cart_quat
            cart_quat_error = np.clip(get_quaternion_error(current_cart_quat, desired_cart_quat), -0.5, 0.5)

            err = np.hstack((cart_pos_error, cart_quat_error))
            err_norm = np.sum(cart_pos_error**2) + np.sum((current_cart_quat - desired_cart_quat)**2)
            if err_norm > old_err_norm:
                q = q_old
                dt = 0.7 * dt
                continue
            else:
                dt = 1.025 * dt

            if err_norm < eps:
                break
            if i > IT_MAX:
                break

            old_err_norm = err_norm

            ### get Jacobian by mujoco
            self.data.qpos[:7] = q
            mujoco.mj_forward(self.model, self.data)

            jacp = self.get_body_jacp("tcp")[:, :7].copy()
            jacr = self.get_body_jacr("tcp")[:, :7].copy()

            J = np.concatenate((jacp, jacr), axis=0)

            Jw = J.dot(w)

            # J * W * J.T + J_reg * I
            JwJ_reg = Jw.dot(J.T) + J_reg * np.eye(J.shape[0])

            # Null space velocity, points to home position
            qd_null = pgain_null * (target_theta_null - q)

            margin_to_limit = 0.1
            qd_null_limit = np.zeros(qd_null.shape)
            qd_null_limit_max = pgain_limit * (q_max - margin_to_limit - q)
            qd_null_limit_min = pgain_limit * (q_min + margin_to_limit - q)
            qd_null_limit[q > q_max - margin_to_limit] += qd_null_limit_max[q > q_max - margin_to_limit]
            qd_null_limit[q < q_min + margin_to_limit] += qd_null_limit_min[q < q_min + margin_to_limit]
            qd_null += qd_null_limit

            # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null
            qd_d = np.linalg.solve(JwJ_reg, pgain * err - J.dot(qd_null))

            qd_d = w.dot(J.transpose()).dot(qd_d) + qd_null

            i += 1

        return q


class PandaIKController(PandaPDController):
    def __init__(self):
        super().__init__()

        urdf_path = os.path.join(
            __file__, "/home/xueyinLi/project/HDAR-Simulator/model/panda_arm_hand_pinocchio.urdf"
        )
        self.model = pin.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.end_effector_id = self.model.getFrameId("panda_grasptarget")
        self.max_iter = 10
        self.eps = 1e-4
        self.alpha = 1e-1
        self.control_counter = 0

    def compute_ik(self, target_position, target_quaternion, current_j_pos):

        target_rotation = pin.Quaternion(target_quaternion).toRotationMatrix()

        q = current_j_pos.copy()
        q = np.append(q, [0, 0])

        for i in range(self.max_iter):

            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            current_transform = self.data.oMf[self.end_effector_id]
            current_position = current_transform.translation
            current_rotation = current_transform.rotation


            position_error = target_position - current_position
            rotation_error = 0.5 * pin.log3(current_rotation.T @ target_rotation)
            error = np.concatenate([position_error, rotation_error])


            if np.linalg.norm(error) < self.eps:
                break


            J = pin.computeFrameJacobian(self.model, self.data, q, self.end_effector_id, pin.ReferenceFrame.LOCAL)


            dq = self.alpha * np.linalg.pinv(J) @ error


            q = pin.integrate(self.model, q, dq)

        return q[:7]

    def get_control(self, target_position, target_quaternion, current_j_pos, current_j_vel):
        target_j_pos = self.compute_ik(target_position, target_quaternion, current_j_pos)
        
        control_signal = super().get_control(target_j_pos, current_j_pos, current_j_vel)
        
        return control_signal


# class PandaIKController(PandaPDController):

#     def __init__(self):
#         super().__init__()
#         obj_urdf = os.path.join(
#             __file__, "../../../model/panda_arm_hand_pinocchio.urdf"
#         )
#         self.pin_model = pinocchio.buildModelFromUrdf(obj_urdf)
#         self.pin_data = self.pin_model.createData()

#         self.pin_end_effector_frame_id = self.pin_model.getFrameId(
#             "panda_grasptarget"
#         )

#         self.pin_q = np.zeros(self.pin_model.nv)
#         self.pin_qd = np.zeros(self.pin_model.nv)
#         self.J_reg = 1e-6  # Jacobian regularization constant
#         self.W = np.diag([1, 1, 1, 1, 1, 1, 1])

#         # Null-space theta configuration
#         self.target_th_null = np.array(
#             [
#                 3.57795216e-09,
#                 1.74532920e-01,
#                 3.30500960e-08,
#                 -8.72664630e-01,
#                 -1.14096181e-07,
#                 1.22173047e00,
#                 7.85398126e-01,
#             ]
#         )
#         self.pgain_null = [1.2, 1.2, 1.2, 1.2, 0.5, 0.3, 0.1]

#     def get_control(
#         self,
#         target_c_pos,
#         target_c_rot,
#         current_c_pos,
#         current_c_rot,
#         current_j_pos,
#         current_j_vel
#     ):

#         xd_d = target_c_pos - current_c_pos
#         target_c_acc = self.pgain * xd_d

#         self.pin_q[:7] = current_j_pos
#         pinocchio.computeJointJacobians(self.pin_model, self.pin_data, self.pin_q)
#         pinocchio.framesForwardKinematics(self.pin_model, self.pin_data, self.pin_q)
#         J = pinocchio.getFrameJacobian(
#             self.pin_model,
#             self.pin_data,
#             self.pin_end_effector_frame_id,
#             pinocchio.LOCAL_WORLD_ALIGNED,
#         )[:, :7]

#         J = J[:3, :]
#         Jw = J.dot(self.W)

#         # J *  W * J' + reg * I
#         JwJ_reg = Jw.dot(J.T) + self.J_reg * np.eye(3)

#         # Null space movement
#         qd_null = self.pgain_null * (self.target_th_null - current_j_pos)
#         # W J.T (J W J' + reg I)^-1 xd_d + (I - W J.T (J W J' + reg I)^-1 J qd_null

#         qd_d = np.linalg.solve(JwJ_reg, target_c_acc - J.dot(qd_null))
#         qd_d = self.W.dot(J.transpose()).dot(qd_d) + qd_null

#         robot.des_c_pos = self.desired_c_pos
#         robot.des_c_vel = self.desired_c_vel

#         robot.jointTrackingController.setSetPoint(robot.current_j_pos, qd_d)

#         # return robot.jointTrackingController.getControl(robot)
#         return super().get_control(target_j_pos, current_j_pos, current_j_vel)


class BoxPushingSimulator(FancyGemSimulator):

    def __init__(self):
        self.env: BoxPushingEnvBase
        super().__init__('BoxPushingDense-v0')
        self.device = MetaQuest3("ALRMetaQuest3")
        self.controller = PandaPDController()
        # self.controller = PandaIKController()
        self.target_j_pos = self.env.data.qpos[:7].copy()
        self.control_counter = 0

    def get_action(self):
        input_data = self.device.get_input_data()
        self.control_counter += 1
        if input_data is None:
            return self.controller.get_control(
                self.target_j_pos, self.env.data.qpos[:7], self.env.data.qvel[:7]
            )
        if self.control_counter % 1 == 0:
            right_hand = input_data['right']
            ee_pos = np.array(right_hand["pos"])
            ee_rot = np.array([0, 1, 0, 0])
            # # ee_rot = np.array(right_hand["rot"])
            start = time.time()
            self.target_j_pos = self.env.calculateOfflineIK(
                ee_pos, ee_rot
            )
            print(self.target_j_pos)
            print("IK time:", time.time() - start)
            self.control_counter = 0
        self.control_counter += 1
        current_j_pos = self.env.data.qpos[:7].copy()
        current_j_vel = self.env.data.qvel[:7].copy()
        return self.controller.get_control(
            self.target_j_pos, current_j_pos, current_j_vel
        )


if __name__ == '__main__':
    simulator = BoxPushingSimulator()
    simulator.run()