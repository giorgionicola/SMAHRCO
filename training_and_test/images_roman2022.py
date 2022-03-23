import os.path

import matplotlib.pyplot as plt
import numpy as np
import cv2
from skimage.transform import resize_local_mean
import apriltag


crop_region = ((0, 1080), (210, 1750))
final_shape = (128, 128)

options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)


def grayConversion(image):
    grayValue = 0.07 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.21 * image[:, :, 0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


class Data:
    def __init__(self, folder, filename, grasp_combination):
        """
        Class to open acquired data a get all relevant infos

        filename is defined as follows: y_x_z_angle_n.npz
        :param folder:
        :param filename:
        :param grasp_combination:
        """

        self.grasp1 = int(grasp_combination[0])
        self.grasp2 = int(grasp_combination[-1])

        self.folder = folder
        self.filename = filename
        self.name = filename[:-4]

        start = 0
        index = []
        while True:
            i = filename.find('_', start)
            if i != -1:
                index.append(i)
            else:
                break
            start = index[-1] + 1

        self.pos_x = int(filename[index[0] + 1: index[1]])
        self.pos_y = int(filename[: index[0]])
        self.pos_z = int(filename[index[1] + 1: index[2]])
        self.angle = int(filename[index[2] + 1: index[3]])

        self.n = int(filename[index[3] + 1:-4])

        self.grasp_combination = (self.grasp1, self.grasp2)
        self.position_combination = (self.pos_x, self.pos_y, self.pos_z)
        self.combination = self.grasp_combination + self.position_combination

    def get_data(self):
        data = np.load(os.path.join(self.folder,
                                    os.path.join(f'{self.grasp_combination[0]}_{self.grasp_combination[1]}',
                                                 self.filename)))
        self.rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)

        self.depth = data['depth']

    def delete_data(self):
        del self.rgb, self.depth

def open_and_mask_rgb_depth(data: Data, image: bool = False):
    depth_threshold = 1000

    data.get_data()
    depth_image = data.depth

    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(depth_image)

    depth_mask = (depth_image > 0) * (depth_image < depth_threshold)

    depth_image = depth_image * depth_mask

    plt.figure()
    plt.imshow(depth_image)

    rgb_image = data.rgb
    gray_image = grayConversion(rgb_image)

    data.delete_data()

    tags_detected = detector.detect(gray_image)

    if len(tags_detected) == 2:
        tag1_corners = tags_detected[0].corners
        tag2_corners = tags_detected[1].corners

        bottom_point1 = tag1_corners[tag1_corners[:, 1].argmax()]
        bottom_point2 = tag2_corners[tag2_corners[:, 1].argmax()]

        a = (bottom_point1[1] - bottom_point2[1]) / (bottom_point1[0] - bottom_point2[0])
        b = bottom_point1[1] - a * bottom_point1[0]
    elif len(tags_detected) == 1:
        tag1_corners = tags_detected[0].corners
        a = 0
        b = tag1_corners[tag1_corners[:, 1].argmax(), 1]
    elif len(tags_detected) > 2:
        print(
            f'WTF!!! More than 2 tags!!!! tags detected: {len(tags_detected)}, current method not likely to work with more than 3 tags')
        while len(tags_detected) > 2:
            avg_y_center = np.mean([tags_detected[i].center[1] for i in range(len(tags_detected))])
            delta_y = [tags_detected[i].center[1] - avg_y_center for i in range(len(tags_detected))]
            tags_detected.pop(np.argmax([abs(d) for d in delta_y]))

        tag1_corners = tags_detected[0].corners
        tag2_corners = tags_detected[1].corners

        bottom_point1 = tag1_corners[tag1_corners[:, 1].argmax()]
        bottom_point2 = tag2_corners[tag2_corners[:, 1].argmax()]

        a = (bottom_point1[1] - bottom_point2[1]) / (bottom_point1[0] - bottom_point2[0])
        b = bottom_point1[1] - a * bottom_point1[0]

    if len(tags_detected) > 0:
        bottom_line = lambda x: a * x + b
        for i in range(depth_mask.shape[1]):
            depth_mask[:int(bottom_line(i) + 100), i] = False

    depth_image = depth_image * depth_mask

    depth_image = depth_image / depth_threshold

    if image:

        depth_mask_to_rgb = np.zeros(shape=(*depth_mask.shape, 3))
        for i in range(3):
            depth_mask_to_rgb[:, :, i] = depth_mask
        rgb_image[:, :, :3] = rgb_image[:, :, :3] * depth_mask_to_rgb

        return depth_image, rgb_image
    else:
        return depth_image, True


acquisition_path = '/mnt/data/drapebot_dataset/2022_02_17'

data = Data(folder=acquisition_path, filename='500_0_0_0_0.npz', grasp_combination='1_4')

depth_image, rgb_image = open_and_mask_rgb_depth(data, image=True)

plt.figure()
plt.imshow(depth_image)
plt.tight_layout()


depth_image = depth_image[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1]]
depth = resize_local_mean(depth_image, final_shape)
plt.figure()
plt.imshow(depth)
plt.tight_layout()

