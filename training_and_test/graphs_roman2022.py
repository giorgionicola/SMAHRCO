import csv
import matplotlib.pyplot as plt
import os
import numpy as np

plt.rc('font', size=15)


# define moving average function
def moving_avg(x, n):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def read_csv(path_to_csv):
    time1 = []
    time2 = []
    time3 = []
    pos_error_x = []
    pos_error_y = []
    pos_error_z = []
    robot_pos_x = []
    robot_pos_y = []
    robot_pos_z = []
    twist_x = []
    twist_y = []
    twist_z = []

    with open(path_to_csv, mode='r') as file:
        csvFile = csv.reader(file)
        line_counter = 0
        for lines in csvFile:
            if line_counter > 0:
                if lines[1] != '':
                    time1.append(float(lines[0]))
                    pos_error_x.append(float(lines[1]))
                    pos_error_y.append(float(lines[2]))
                    pos_error_z.append(float(lines[3]))
                if lines[4] != '':
                    time2.append(float(lines[0]))
                    robot_pos_x.append(float(lines[4]))
                    robot_pos_y.append(float(lines[5]))
                    robot_pos_z.append(float(lines[6]))
                if lines[10] != '':
                    time3.append(float(lines[0]))
                    twist_x.append(float(lines[10]))
                    twist_y.append(float(lines[11]))
                    twist_z.append(float(lines[12]))

            line_counter += 1

    return time1, time2, time3, pos_error_x, pos_error_y, pos_error_z, robot_pos_x, robot_pos_y, robot_pos_z, twist_x, twist_y, twist_z


def plot(path_to_csv):
    time1, time2, time3, pos_error_x, pos_error_y, pos_error_z, robot_pos_x, robot_pos_y, robot_pos_z, twist_x, twist_y, twist_z = read_csv(
        path_to_csv)

    while robot_pos_x[1] - robot_pos_x[0] < 1e-4:
        robot_pos_x.pop(0)
        robot_pos_y.pop(0)
        robot_pos_z.pop(0)
        pos_error_x.pop(0)
        pos_error_y.pop(0)
        pos_error_z.pop(0)
        twist_x.pop(0)
        twist_y.pop(0)
        twist_z.pop(0)
        time1.pop(0)
        time2.pop(0)
        time3.pop(0)

    t0 = time1[0]
    for i in range(len(time1)):
        time1[i] -= t0

    t0 = time2[0]
    for i in range(len(time2)):
        time2[i] -= t0

    t0 = time3[0]
    for i in range(len(time3)):
        time3[i] -= t0

    p0 = robot_pos_x[-1]
    for i in range(len(robot_pos_x)):
        robot_pos_x[i] -= p0

    p0 = robot_pos_y[-1]
    for i in range(len(robot_pos_y)):
        robot_pos_y[i] -= p0

    p0 = robot_pos_z[-1]
    for i in range(len(robot_pos_z)):
        robot_pos_z[i] -= p0
        robot_pos_z[i] = -robot_pos_z[i]

    while abs(twist_x[-1] - twist_x[-2]) < 0.0001 and \
            abs(twist_y[-1] - twist_y[-2]) < 0.0001 and \
            abs(twist_z[-1] - twist_z[-2]) < 0.0001:
        robot_pos_x.pop(-1)
        robot_pos_y.pop(-1)
        robot_pos_z.pop(-1)
        pos_error_x.pop(-1)
        pos_error_y.pop(-1)
        pos_error_z.pop(-1)
        twist_x.pop(-1)
        twist_y.pop(-1)
        twist_z.pop(-1)
        time1.pop(-1)
        time2.pop(-1)
        time3.pop(-1)

    for i in range(len(pos_error_y)):
        pos_error_y[i] += 0.6
        robot_pos_y[i] += 0.6

    _, (ax0, ax1, ax2, ax3) = plt.subplots(nrows=4)
    ax0.plot(time1, pos_error_x, label='Estimated ply deformation $d_x$')
    ax0.plot(time2, robot_pos_x, label='Real ply deformation $d_x^{real}$')
    ax0.plot(time1, [robot_pos_x[-1] for _ in range(len(time1))], 'r--', label='$d_x^{rif}$')
    ax0.set_ylabel('$x~[m]$')
    ax0.legend()
    ax0.grid()

    ax1.plot(time1, pos_error_y, label='Estimated ply deformation $d_y$')
    ax1.plot(time2, robot_pos_y, label='Real ply deformation $d_y^{real}$')
    ax1.plot(time1, [robot_pos_y[-1] for _ in range(len(time1))], 'r--', label='$d_y^{rif}$')
    ax1.set_ylabel('$y~[m]$')
    ax1.legend(loc='upper right')
    ax1.grid()

    ax2.plot(time1, pos_error_z, label='Estimated ply deformation $d_z$')
    ax2.plot(time2, robot_pos_z, label='Real ply deformation $d_z^{real}$')
    ax2.plot(time1, [robot_pos_z[-1] for _ in range(len(time1))], 'r--', label='$d_z^{rif}$')
    ax2.set_ylabel('$z~[m]$')
    ax2.legend()
    ax2.grid()

    ln1 = ax3.plot(time1, [np.sqrt((px - rx) ** 2 + (py - ry) ** 2 + (pz - rz) ** 2) for px, rx, py, ry, pz, rz in
                           zip(pos_error_x, robot_pos_x, pos_error_y, robot_pos_y, pos_error_z, robot_pos_z)],
                   label='Estimation error')
    ax3.set_ylabel('Error $[m]$')
    ax3.set_xlabel('$Time~[s]$')
    ax3.grid()

    ax4 = ax3.twinx()
    ln2 = ax4.plot(time1, [np.sqrt(vx ** 2 + vy ** 2 + vz ** 2) for vx, vy, vz in zip(twist_x, twist_y, twist_z)],
                   label='Tool speed', color='C1')
    ax4.set_ylabel('$Tool~speed~[m/s]$')

    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax4.legend(lns, labs)


plot('/mnt/data/drapebot_dataset/exp_1.csv')
plot('/mnt/data/drapebot_dataset/exp_2.csv')

time1, time2, time3, pos_error_x, pos_error_y, pos_error_z, robot_pos_x, robot_pos_y, robot_pos_z, twist_x, twist_y, twist_z = read_csv(
    '/mnt/data/drapebot_dataset/human_robot2.csv')

while robot_pos_x[3] - robot_pos_x[2] < 1e-3:
    robot_pos_x.pop(0)
    robot_pos_y.pop(0)
    robot_pos_z.pop(0)
    pos_error_x.pop(0)
    pos_error_y.pop(0)
    pos_error_z.pop(0)
    twist_x.pop(0)
    twist_y.pop(0)
    twist_z.pop(0)
    time1.pop(0)
    time2.pop(0)
    time3.pop(0)

t0 = time1[0]
for i in range(len(time1)):
    time1[i] -= t0

t0 = time2[0]
for i in range(len(time2)):
    time2[i] -= t0

t0 = time3[0]
for i in range(len(time3)):
    time3[i] -= t0

ref_pos_x = [rx - err_x for rx, err_x in zip(robot_pos_x, pos_error_x)]
ref_pos_y = [ry - err_y for ry, err_y in zip(robot_pos_y, pos_error_y)]
ref_pos_z = [rz + err_z for rz, err_z in zip(robot_pos_z, pos_error_z)]

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3)
ax0.plot(time1, robot_pos_x, label='Robot position x')
ax0.plot(time1, ref_pos_x, label='Reference position x')
ax0.set_ylabel('$x~[m]$')
ax0.grid()
ax0.axvline(0, color='g', linestyle='--', linewidth=2)
ax0.axvline(10, color='r', linestyle='--', linewidth=2)
ax0.axvline(28, color='g', linestyle='--', linewidth=2, label='Start Movement')
ax0.axvline(35.5, color='r', linestyle='--', linewidth=2, label='End Movement')
# ax0.legend()

ax1.plot(time1, robot_pos_y, label='Robot position y')
ax1.plot(time1, ref_pos_y, label='Reference position y')
ax1.set_ylabel('$y~[m]$')
ax1.grid()
ax1.axvline(20.3, color='g', linestyle='--', linewidth=2, label='Start Movement')
ax1.axvline(27.5, color='r', linestyle='--', linewidth=2, label='End Movement')
# ax1.legend()

ax2.plot(time1, robot_pos_z, label='Robot position')
ax2.plot(time1, ref_pos_z, label='Reference position')
ax2.set_ylabel('$z~[m]$')
ax2.set_xlabel('$Time~[s]$')
ax2.axvline(11.5, color='g', linestyle='--', linewidth=2, label='Start Movement')
ax2.axvline(20, color='r', linestyle='--', linewidth=2, label='End Movement')
ax2.legend()
ax2.grid()

plt.show()
