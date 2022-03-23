import numpy as np
import symengine
from symengine import Matrix, pi, cos, sin, Lambdify


class UrKinematics:
    def __init__(self,
                 T_start: np.array = None,
                 T_tcp: np.array = None):

        # UR10
        # self.d1 = 0.1273
        # self.a2 = -0.612
        # self.a3 = -0.5723
        # self.d4 = 0.163941
        # self.d5 = 0.1157
        # self.d6 = 0.0922

        # self.shoulder_offset = 0.220941
        # self.elbow_offset = -0.1719

        # UR5
        self.d1 = 0.089159
        self.a2 = -0.42500
        self.a3 = -0.39225
        self.d4 = 0.10915
        self.d5 = 0.09465
        self.d6 = 0.0823

        self.shoulder_offset = 0.13585
        self.elbow_offset = -0.1197

        self.T_start = T_start
        self.T_tcp = T_tcp

        self.symbolic_kinematics()

    def symbolic_kinematics(self):
        shoulder_height = self.d1
        upper_arm_length = -self.a2
        forearm_length = -self.a3
        wrist_1_length = self.d4 - self.elbow_offset - self.shoulder_offset
        wrist_2_length = self.d5
        wrist_3_length = self.d6

        q0, q1, q2, q3, q4, q5 = symengine.var('q0 q1 q2 q3 q4 q5')

        if self.T_start is None:
            T_start = Matrix(np.eye(4))
        else:
            T_start = Matrix([[self.T_start[row, col] for col in range(self.T_start.shape[1])] for row in
                              range(self.T_start.shape[0])])

        T_base_baselink = Matrix([[cos(pi), -sin(pi), 0.0, 0.0],
                                  [sin(pi), cos(pi), 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])

        shoulder_pan_j = Matrix([[cos(q0), -sin(q0), 0.0, 0.0],
                                 [sin(q0), cos(q0), 0.0, 0.0],
                                 [0.0, 0.0, 1.0, shoulder_height],
                                 [0.0, 0.0, 0.0, 1.0]])

        shoulder_lift_j = Matrix([[cos(q1 + pi / 2), 0.0, sin(q1 + pi / 2), 0.0],
                                  [0.0, 1.0, 0.0, self.shoulder_offset],
                                  [-sin(q1 + pi / 2), 0.0, cos(q1 + pi / 2), 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])

        elbow_j = Matrix([[cos(q2), 0.0, sin(q2), 0.0],
                          [0.0, 1.0, 0.0, self.elbow_offset],
                          [-sin(q2), 0.0, cos(q2), upper_arm_length],
                          [0.0, 0.0, 0.0, 1.0]])

        wrist1_j = Matrix([[cos(q3 + pi / 2), 0.0, sin(q3 + pi / 2), 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [-sin(q3 + pi / 2), 0.0, cos(q3 + pi / 2), forearm_length],
                           [0.0, 0.0, 0.0, 1.0]])

        wrist2_j = Matrix([[cos(q4), -sin(q4), 0.0, 0.0],
                           [sin(q4), cos(q4), 0.0, wrist_1_length],
                           [0.0, 0.0, 1.0, 0.0],
                           [0.0, 0.0, 0.0, 1.0]])

        wrist3_j = Matrix([[cos(q5), 0.0, sin(q5), 0.0],
                           [0.0, 1.0, 0.0, 0.0],
                           [-sin(q5), 0.0, cos(q5), wrist_2_length],
                           [0.0, 0.0, 0.0, 1.0]])

        tool0 = Matrix([[1, 0, 0, 0.0],
                        [0, cos(-pi / 2), -sin(-pi / 2), wrist_3_length],
                        [0, sin(-pi / 2), cos(-pi / 2), 0],
                        [0.0, 0.0, 0.0, 1.0]])

        if self.T_tcp is None:
            tcp = Matrix(np.eye(4))
        else:
            tcp = Matrix([[self.T_tcp[row, col] for col in range(self.T_tcp.shape[1])] for row in
                          range(self.T_tcp.shape[0])])

        self.T01_l = Lambdify([q0], T_start * shoulder_pan_j, backend='llvm')
        self.T12_l = Lambdify([q1], shoulder_lift_j, backend='llvm')
        self.T23_l = Lambdify([q2], elbow_j, backend='llvm')
        self.T34_l = Lambdify([q3], wrist1_j, backend='llvm')
        self.T45_l = Lambdify([q4], wrist2_j, backend='llvm')
        self.T56_l = Lambdify([q5], wrist3_j, backend='llvm')

        T_elbow = T_start * shoulder_pan_j * shoulder_lift_j * elbow_j
        T_tool0 = T_start * shoulder_pan_j * shoulder_lift_j * elbow_j * wrist1_j * wrist2_j * wrist3_j * tool0
        T_tcp = T_tool0 * tcp

        T_base_tcp = T_base_baselink * T_tcp

        P_elbow = T_elbow[0:3, 3]
        P_tool0 = T_tool0[0:3, 3]
        P_tcp = T_tcp[0:3, 3]

        x_elbow = Matrix([P_elbow[0], P_elbow[1], P_elbow[2]])
        x_tool0 = Matrix([P_tool0[0], P_tool0[1], P_tool0[2]])
        x_tcp = Matrix([P_tcp[0], P_tcp[1], P_tcp[2]])

        J_elbow = Matrix([[x_elbow[0].diff(q0), x_elbow[0].diff(q1), x_elbow[0].diff(q2)],
                          [x_elbow[1].diff(q0), x_elbow[1].diff(q1), x_elbow[1].diff(q2)],
                          [x_elbow[2].diff(q0), x_elbow[2].diff(q1), x_elbow[2].diff(q2)]])

        J_tool0 = Matrix([[x_tool0[0].diff(q0), x_tool0[0].diff(q1), x_tool0[0].diff(q2), x_tool0[0].diff(q3),
                           x_tool0[0].diff(q4), x_tool0[0].diff(q5)],
                          [x_tool0[1].diff(q0), x_tool0[1].diff(q1), x_tool0[1].diff(q2), x_tool0[1].diff(q3),
                           x_tool0[1].diff(q4), x_tool0[1].diff(q5)],
                          [x_tool0[2].diff(q0), x_tool0[2].diff(q1), x_tool0[2].diff(q2), x_tool0[2].diff(q3),
                           x_tool0[2].diff(q4), x_tool0[2].diff(q5)]])

        J_tcp = Matrix([[x_tcp[0].diff(q0), x_tcp[0].diff(q1), x_tcp[0].diff(q2), x_tcp[0].diff(q3), x_tcp[0].diff(q4),
                         x_tcp[0].diff(q5)],
                        [x_tcp[1].diff(q0), x_tcp[1].diff(q1), x_tcp[1].diff(q2), x_tcp[1].diff(q3), x_tcp[1].diff(q4),
                         x_tcp[1].diff(q5)],
                        [x_tcp[2].diff(q0), x_tcp[2].diff(q1), x_tcp[2].diff(q2), x_tcp[2].diff(q3), x_tcp[2].diff(q4),
                         x_tcp[2].diff(q5)]])

        self.T_elbow_l = Lambdify([q0, q1, q2], T_elbow, backend='llvm')
        self.T_tool0_l = Lambdify([q0, q1, q2, q3, q4, q5], T_tool0, backend='llvm')
        self.T_tcp_l = Lambdify([q0, q1, q2, q3, q4, q5], T_tcp, backend='llvm')
        self.P_elbow_l = Lambdify([q0, q1, q2], P_elbow, backend='llvm')
        self.P_tool0_l = Lambdify([q0, q1, q2, q3, q4, q5], P_tool0, backend='llvm')
        self.P_tcp_l = Lambdify([q0, q1, q2, q3, q4, q5], P_tcp, backend='llvm')
        self.J_elbow_l = Lambdify([[q0, q1, q2]], J_elbow, backend='llvm')
        self.J_tool0_l = Lambdify([[q0, q1, q2, q3, q4, q5]], J_tool0, backend='llvm')
        self.J_tcp_l = Lambdify([[q0, q1, q2, q3, q4, q5]], J_tcp, backend='llvm')

        self.T_base_tcp_l = Lambdify([q0, q1, q2, q3, q4, q5], T_base_tcp, backend='llvm')
