import numpy as np
import mpl_toolkits
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_quarter_donut(position_x, position_y, position_z, direction_z, angle):
    positions_x = []
    positions_y = []
    positions_z = []

    direction_x = -1
    distance = np.sqrt(position_x ** 2 + position_y ** 2 + position_z ** 2)
    while abs(position_z) <= max_height * np.cos(angle) and max_distance * 1.001 >= distance >= min_distance / 1.001:
        positions_x.append(position_x)
        positions_y.append(position_y)
        positions_z.append(position_z)

        while max_distance * 1.001 >= distance >= min_distance / 1.001 and position_x > 0.099:
            position_x += direction_x * step_x
            distance = np.sqrt(position_x ** 2 + position_y ** 2 + position_z ** 2)
            if max_distance * 1.001 >= distance >= min_distance / 1.001:
                positions_x.append(position_x)
                positions_y.append(position_y)
                positions_z.append(position_z)

        direction_x *= -1
        position_z += direction_z * step_z * np.cos(angle)
        position_y += direction_z * step_z * np.sin(angle)

        distance = np.sqrt(position_x ** 2 + position_y ** 2 + position_z ** 2)
        if not direction_z * max_height * np.cos(angle) / 1.01 < position_z < direction_z * max_height * np.cos(
                angle) * 1.01:
            while distance > max_distance * 1.001 or distance < min_distance / 1.001 or position_x < 0.099:
                position_x += direction_x * step_x
                distance = np.sqrt(position_x ** 2 + position_y ** 2 + position_z ** 2)

    return positions_x, positions_y, positions_z


def move_along_one_tool_orientation(position_x, position_y, position_z, angle):
    up_positions_x, up_positions_y, up_positions_z = make_quarter_donut(position_x=position_x,
                                                                        position_y=position_y,
                                                                        position_z=position_z,
                                                                        angle=angle,
                                                                        direction_z=1)
    position_z -= step_z * np.cos(angle)
    position_y += position_z * np.tan(angle)
    position_x = np.sqrt(max_distance ** 2 - position_y ** 2 - position_z ** 2)
    low_positions_x, low_positions_y, low_positions_z = make_quarter_donut(position_x=position_x,
                                                                           position_y=position_y,
                                                                           position_z=position_z,
                                                                           angle=angle,
                                                                           direction_z=-1)

    positions_x = up_positions_x + low_positions_x
    positions_y = up_positions_y + low_positions_y
    positions_z = up_positions_z + low_positions_z

    return positions_x, positions_y, positions_z


max_distance = 0.85
min_distance = 0.3
max_height = 0.5
min_height = -0.5
max_rotation = 30
min_rotation = -30
max_displacement = 0.3

step_x = 0.025
step_z = 0.025
step_y = 0.1
step_rotation = 10

start_x = max_distance
start_y = 0
start_z = 0
start_rotation = 0

direction_rot = 1
direction_y = 1

rotation = start_rotation
pos_x = start_x
pos_y = start_y
pos_z = start_z

while True:
    print(pos_y)
    pos_x = np.sqrt(max_distance ** 2 - pos_y ** 2 - pos_z ** 2)
    while True:
        positions_x, positions_y, positions_z = move_along_one_tool_orientation(position_x=pos_x,
                                                                                position_y=pos_y,
                                                                                position_z=pos_z,
                                                                                angle=np.deg2rad(rotation))

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(positions_x, positions_y, positions_z, '*')
        ax.set_title(f'Tool rotation: {rotation}, Y: {pos_y}')

        rotation += step_rotation * direction_rot
        if rotation > max_rotation + 1:
            rotation = - step_rotation
            direction_rot = -1
        elif rotation < min_rotation - 1:
            rotation = 0
            direction_rot =1
            break

    pos_y += direction_y * step_y

    if pos_y > max_displacement * 1.01:
        pos_y = - step_y
        direction_y = -1
    elif pos_y < -max_displacement * 1.01:
        break

plt.show()
