import os
import numpy as np
from skimage.transform import resize_local_mean
from copy import deepcopy
from tqdm import tqdm
import apriltag


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
        data = np.load(os.path.join(self.folder, self.filename))
        self.rgb = data['color']
        self.depth = data['depth']

    def delete_data(self):
        del self.rgb, self.depth


acquisition_path = '/media/mullis/Volume/drapebot_dataset/2022_02_17'

crop_region = ((0, 1080), (210, 1750))
final_shape = (128, 128)

options = apriltag.DetectorOptions(families="tag36h11")
detector = apriltag.Detector(options)


def grayConversion(image):
    grayValue = 0.07 * image[:, :, 2] + 0.72 * image[:, :, 1] + 0.21 * image[:, :, 0]
    gray_img = grayValue.astype(np.uint8)
    return gray_img


def open_and_mask_rgb_depth(data: Data, image: bool = False):
    depth_threshold = 1000

    data.get_data()
    depth_image = data.depth


    # fig = plt.figure()
    # fig.add_subplot(1, 2, 1)
    # plt.imshow(depth_image)

    depth_mask = (depth_image > 0) * (depth_image < depth_threshold)

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
        print(f'WTF!!! More than 2 tags!!!! tags detected: {len(tags_detected)}, current method not likely to work with more than 3 tags')
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
            depth_mask[:int(bottom_line(i)) + 100, i] = False

    depth_image = depth_image * depth_mask

    # fig.add_subplot(1, 2, 2)
    # plt.imshow(depth_image)
    #
    # plt.show()

    depth_image = depth_image / depth_threshold

    if image:

        depth_mask_to_rgb = np.zeros(shape=(*depth_mask.shape, 3))
        for i in range(3):
            depth_mask_to_rgb[:, :, i] = depth_mask
        rgb_image[:, :, :3] = rgb_image[:, :, :3] * depth_mask_to_rgb

        return depth_image, rgb_image, True
    else:
        return depth_image, True


grasp_combinations = os.listdir(acquisition_path)
if 'training.npz' in grasp_combinations:
    grasp_combinations.remove('training.npz')
if 'test.npz' in grasp_combinations:
    grasp_combinations.remove('test.npz')
trials = os.listdir(os.path.join(acquisition_path, grasp_combinations[0]))

data = []
for g in grasp_combinations:
    for t in trials:
        folder = os.path.join(acquisition_path, os.path.join(g, t))
        filenames = os.listdir(folder)
        for f in filenames:
            data.append(Data(folder=folder, filename=f, grasp_combination=g))

pos_combinations = []
data_per_combinations = []

for d in data:
    if d.position_combination not in pos_combinations:
        pos_combinations.append(d.position_combination)
        data_per_combinations.append(1)
    else:
        i = pos_combinations.index(d.position_combination)
        data_per_combinations[i] += 1

training_combinations = []
data_per_training_combinations = []
test_combinations = []
data_per_test_combinations = []
counter = 0
n_combination_test = len(pos_combinations) // 5
while counter < n_combination_test:
    if len(pos_combinations) == 0:
        raise RuntimeError('pos_combination finished before fillinf test_combinations')
    if len(training_combinations) >= 4 * n_combination_test:
        test_combinations += deepcopy(pos_combinations)
        data_per_test_combinations += deepcopy(data_per_combinations)
        pos_combinations = []
        data_per_combinations = []
        break
    i = np.random.randint(len(pos_combinations))
    too_close = False
    if len(test_combinations) >= 1:
        current_comb = pos_combinations[i]
        for c in test_combinations:
            if np.linalg.norm(np.array(current_comb) - np.array(c)) < 75:
                too_close = True
                break
    if not too_close:
        test_combinations.append(pos_combinations.pop(i))
        data_per_test_combinations.append(data_per_combinations.pop(i))
        counter += 1
    else:
        training_combinations.append(pos_combinations.pop(i))
        data_per_training_combinations.append(data_per_combinations.pop(i))

if len(pos_combinations) > 0:
    training_combinations += pos_combinations
    data_per_training_combinations += data_per_combinations

assert sum(data_per_training_combinations) + sum(data_per_test_combinations) == len(data), 'fuck'

ntest_combinatios= 0

test_depth_images = []
training_depth_images = []

test_depth_names = []
training_depth_names = []

i = 0
ii = 0
for d in tqdm(data):
    depth_image, ok = open_and_mask_rgb_depth(d, image=False)
    depth_image = depth_image[crop_region[0][0]:crop_region[0][1], crop_region[1][0]:crop_region[1][1]]
    depth_image = resize_local_mean(depth_image, final_shape)
    if d.position_combination in test_combinations:
        test_depth_images.append(depth_image)
        test_depth_names.append(d.name)
    else:
        training_depth_images.append(depth_image)
        training_depth_names.append(d.name)

np.savez(os.path.join(acquisition_path, f'test_{final_shape[0]}_{final_shape[1]}'),
         depths=test_depth_images,
         names=test_depth_names)
np.savez(os.path.join(acquisition_path, f'training_{final_shape[0]}_{final_shape[1]}'),
         depths=training_depth_images,
         names=training_depth_names)