import os

import rospy
import numpy as np
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import JointState, Image
from ur_kinematics import UrKinematics
import torch
from scipy.spatial.transform import Rotation as R
from cv_bridge import CvBridge

from typing import Tuple


class CartesianPosToVelXYZController(object):
    """
    Cartesian Position Controller only for X,Y,Z with output Cartesian Velocities
    """

    def __init__(self,
                 path_to_nn_model: str,
                 start_dist_rif: np.array,
                 max_twist_cmd: list,
                 kp: list,
                 kd: list,
                 max_time_between_data: float,
                 min_twist_cmd: list = None,
                 spawn_started: bool = True,
                 T_start: np.array = None,
                 T_tcp: np.array = None,
                 min_pos_error=None
                 ):
        """

        :param path_to_nn_model: Path to the Pytorch model
        :param max_twist_cmd: Max velocity command [x,y,z]
        :param min_twist_cmd: Min velocity command [x,y,z], if = None is set to -max_cmd_twist
        :param T_start: Transformation between your desired fixed frame and the base_link, if = None is set to eye(4)
        :param T_tcp: Transformation between tool0 and your tcp, if = None is set to eye(4)
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model1 = torch.load(
            '/mnt/data/drapebot_dataset/2022_02_17/training_2022_03_14_16_55_23/epoch_17/model.pth')
        self.model1.to(device)
        self.model1.eval()

        self.model2 = torch.load('/mnt/data/drapebot_dataset/2022_02_17/training_2022_03_14_15_22_13/epoch_6/model.pth')
        self.model2.to(device)
        self.model2.eval()

        #### Model 3

        self.model3 = torch.load(
            '/mnt/data/drapebot_dataset/2022_02_17/training_2022_03_14_17_26_23/epoch_28/model.pth')
        self.model3.eval()

        self.kp = np.array(kp)
        self.kd = np.array(kd)

        if start_dist_rif:
            self.dist_rif = np.array(start_dist_rif)
        else:
            self.dist_rif = np.array([1, 0, 0])

        self.vel = np.array([0, 0, 0])

        self.image = np.zeros(shape=(128, 128))

        # Joints
        self.joints_name: tuple = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
                                   'wrist_2_joint', 'wrist_3_joint')

        self.j_pos = np.array([0.0 for _ in range(6)])
        self.j_vel = np.array([0.0 for _ in range(6)])

        self.max_twist_cmd = np.array(max_twist_cmd)
        if min_twist_cmd:
            self.min_twist_cmd = np.array(min_twist_cmd)
        else:
            self.min_twist_cmd = -np.array(max_twist_cmd)

        if min_pos_error:
            self.min_pos_error = min_pos_error
        else:
            self.min_pos_error = np.array([0, 0, 0])

        if spawn_started:
            self.start = True

        self.kine = UrKinematics(T_start=T_start, T_tcp=T_tcp)

        # Check status controller variables
        self.heard_at_least_once_jointstate = False
        self.got_at_least_one_img = False
        self.last_time_heard_jointstate = -1
        self.last_time_heard_image = -1

        self.current_jointstate_time = -1
        self.current_image_time = -1

        self.old_pos_error = 0
        self.pos_error_buffer = []
        self.max_len_pos_error_buffer = 5

        self.max_time_between_data = max_time_between_data

        self.cmd_pub = rospy.Publisher('/twist_controller/command', Twist, queue_size=1)
        rospy.Subscriber('/joint_states', JointState, self.get_joint_states)
        rospy.Subscriber('/update_position_reference', Vector3, self.update_distance_reference)
        rospy.Subscriber('/preprocessed_image', Image, self.update_image)

        self.prop_err_pub = rospy.Publisher('/proportional_error', Vector3, queue_size=1)
        self.der_err_pub = rospy.Publisher('/derivative_error', Vector3, queue_size=1)
        self.pos_pub = rospy.Publisher('/robot_pos', Vector3, queue_size=1)


        self.br = CvBridge()

    def update_image(self, img_msg: Image):
        img = self.br.imgmsg_to_cv2(img_msg=img_msg)
        self.image = torch.from_numpy(img.reshape((1, 1, 128, 128))).float().to(self.device)
        self.current_image_time = img_msg.header.stamp.to_sec()
        if not self.got_at_least_one_img:
            self.got_at_least_one_img = True

    def update_distance_reference(self, new_dist_rif: Vector3):
        self.dist_rif = np.array([new_dist_rif.x, new_dist_rif.y, new_dist_rif.z])

    def get_joint_states(self, msg: JointState):
        joints_position_in_msg = [msg.name.index(joint_name) for joint_name in self.joints_name]
        for i, jp in enumerate(joints_position_in_msg):
            self.j_pos[i] = msg.position[jp]
            if not len(msg.velocity) == 0:
                self.j_vel[i] = msg.velocity[jp]
            else:
                self.j_vel[i] = 0
        self.current_jointstate_time = msg.header.stamp.to_sec()
        if not self.heard_at_least_once_jointstate:
            self.heard_at_least_once_jointstate = True

    def saturate_cmd(self, cmd: np.array) -> np.array:
        """
        Saturate the command to max and min twist command
        :param cmd: twist command
        :return:
        """
        cmd = np.minimum(cmd, self.max_twist_cmd)
        cmd = np.maximum(cmd, self.min_twist_cmd)
        return cmd

    def send_cmd(self, cmd: np.array):
        msg = Twist()
        msg.linear.x = cmd[0]
        msg.linear.y = cmd[1]
        msg.linear.z = cmd[2]

        self.cmd_pub.publish(msg)

    def get_last_tcp_pos_vel(self) -> Tuple[np.array, np.array]:
        pos = self.kine.P_tcp_l(self.j_pos)
        jacob = self.kine.J_tcp_l(self.j_pos)
        vel_base_frame = jacob @ self.j_vel.reshape(6, 1)

        R_base_ee = self.kine.T_tcp_l(self.j_pos)[:3, :3]
        vel = np.linalg.inv(R_base_ee) @ vel_base_frame

        return pos, vel

    @torch.no_grad()
    def compute_img_to_distance(self) -> np.array:
        distance: torch.Tensor = self.model1(self.image) / 2 + self.model3(self.image) / 2

        return distance.cpu().detach().numpy()

    def loop_controller(self):
        robot_pos, robot_vel = self.get_last_tcp_pos_vel()

        pos_msg :Vector3 = Vector3(x=robot_pos[0], y=robot_pos[1], z=robot_pos[2])
        self.pos_pub.publish(pos_msg)

        distance = self.compute_img_to_distance()
        pos_error = distance[0] - self.dist_rif

        prop_error_msg: Vector3 = Vector3(x=pos_error[0], y=pos_error[1], z=pos_error[2])

        pos_error *= np.abs(pos_error) > self.min_pos_error

        cmd = pos_error * self.kp
        cmd = self.saturate_cmd(cmd)

        R_base_ee = self.kine.T_base_tcp_l(self.j_pos)[:3, :3]
        cmd = R_base_ee @ cmd


        self.prop_err_pub.publish(prop_error_msg)

        self.send_cmd(cmd)

    def stop(self):
        self.send_cmd([0, 0, 0])

    def check_start_condition(self) -> bool:
        ok = True
        if not self.heard_at_least_once_jointstate:
            rospy.logwarn('Controller is started but I have not received yet any joint state, no command send')
            ok = False
        if not self.got_at_least_one_img:
            rospy.logwarn('Controller is started but I have not received yet any image, no command send')
            ok = False
        return ok

    def check_possible_errors(self) -> bool:
        if rospy.Time.now().to_sec() - self.current_image_time > self.max_time_between_data:
            msg = f'Current image is too old!!! delta between messages ' \
                  f'{rospy.Time.now().to_sec() - self.current_image_time} , STOPPING'
            rospy.logerr(msg)
            return False
        if rospy.Time.now().to_sec() - self.current_jointstate_time > self.max_time_between_data:
            msg = f'Current joint state is too old!!! delta between messages ' \
                  f'{rospy.Time.now().to_sec() - self.current_jointstate_time}, stooping'
            rospy.logerr(msg)
            return False

        return True


if __name__ == '__main__':
    rospy.init_node('ur_pos_to_vel_controller')

    loop_rate = 7

    start_dist_rif = [0, 0.6, 0]
    kp = [-0.5, 0.5, -0.5]
    kd = [0.01, 0.01, 0.01]
    # kp = [0,0,0]
    # kd = [0,0,0]
    max_time_between_data = 1.5 / loop_rate
    max_twist_cmd = [0.05, 0.05, 0.05]
    min_twist_cmd = [-0.05, -0.05, -0.05]
    min_pos_error = [0.0, 0.0, 0.0]
    spawn_started = True

    T_start = np.eye(4)
    # T_start[:3, :3] = R.from_euler(seq='xyz', angles=[0, 0, 1.25 * np.pi]).as_matrix()
    # T_start[:3, 3] = [0, 0, 0.01]

    T_tool0_robotiq = np.eye(4)
    T_tool0_robotiq[:3, :3] = R.from_euler(seq='xyz', angles=[0, 0, np.pi / 2]).as_matrix()
    T_tool0_robotiq[:3, 3] = [0, 0, 0.037]

    T_robotiq_upperclamp = np.eye(4)
    T_robotiq_upperclamp[:3, :3] = R.from_euler(seq='xyz', angles=[0, 0, np.pi / 4]).as_matrix()
    T_robotiq_upperclamp[:3, 3] = [0, 0, 0.021]

    T_upperclamp_ee = np.eye(4)
    T_upperclamp_ee[:3, :3] = R.from_euler(seq='xyz', angles=[0, 0, 0]).as_matrix()
    T_upperclamp_ee[:3, 3] = [0, 0, 0.015]

    # T_tcp = T_tool0_robotiq @ T_robotiq_upperclamp @ T_upperclamp_ee

    # UR5

    T_tcp = np.eye(4)
    T_tcp[:3, :3] = R.from_euler(seq='xyz', angles=[-0.000, -0.000, 0.785]).as_matrix()
    T_tcp[:3, 3] = [-0.000, -0.000, 0.073]

    controller = CartesianPosToVelXYZController(start_dist_rif=start_dist_rif,
                                                kp=kp,
                                                kd=kd,
                                                max_time_between_data=max_time_between_data,
                                                max_twist_cmd=max_twist_cmd,
                                                min_twist_cmd=min_twist_cmd,
                                                min_pos_error=min_pos_error,
                                                spawn_started=spawn_started,
                                                T_start=T_start,
                                                T_tcp=T_tcp,
                                                )

    rate = rospy.Rate(loop_rate)
    while not rospy.is_shutdown():
        if controller.start:
            if controller.check_start_condition():
                if controller.check_possible_errors():
                    controller.loop_controller()
                else:
                    controller.stop()
                    break
        rate.sleep()
