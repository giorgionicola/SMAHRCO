from torch.utils.data import Dataset
import torch
import os
from skimage.util import random_noise
from skimage.transform import rotate, AffineTransform, warp
import numpy as np


class CarbonFiberDataset(Dataset):
    """
    name y_x_z_angle_n
    """

    def __init__(self,
                 path,
                 img_shape=(128, 128),
                 training=True,
                 max_rotation=45,
                 max_translation_x=10,
                 max_tranlsation_y=10,
                 prob_flip_lr=0.5,
                 prob_flip_ud=0.5):
        self.training = training
        self.img_shape = img_shape
        if self.training:
            self.path = os.path.join(path, f'training_{self.img_shape[0]}_{self.img_shape[1]}.npz')
        else:
            self.path = os.path.join(path, f'test_{self.img_shape[0]}_{self.img_shape[1]}.npz')

        data = np.load(self.path)
        image_names = data['names']
        images = data['depths']
        self.labels = [[0 for _ in range(3)] for _ in range(len(image_names))]
        self.images = [None for _ in range(len(image_names))]
        for i in range(len(image_names)):
            image_name = image_names[i]
            start = 0
            index = []
            while True:
                j = image_name.find('_', start)
                if j != -1:
                    index.append(j)
                else:
                    break
                start = index[-1] + 1

            self.labels[i][0] = float(image_name[index[0] + 1: index[1]]) / 1000
            self.labels[i][1] = float(image_name[: index[0]]) / 1000
            self.labels[i][2] = float(image_name[index[1] + 1: index[2]]) / 1000

            image = images[i]
            image = np.expand_dims(image, 0)
            self.images[i] = image

        self.max_rotation = max_rotation  # degree
        self.max_translation_x = max_translation_x
        self.max_translation_y = max_tranlsation_y
        self.prob_flip_lr = prob_flip_lr
        self.prob_flip_ud = prob_flip_ud

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.training:
            img = random_noise(self.images[idx], mode='pepper')
            mask = img > 0
            img = random_noise(img, mode='gaussian') * mask
            # img = self.images[idx]

            if self.max_translation_x and self.max_translation_y:
                transformation = AffineTransform(translation=(np.random.randint(low=-self.max_translation_x,
                                                                                high=self.max_translation_x),
                                                              np.random.randint(low=-self.max_translation_y + 1,
                                                                                high=self.max_translation_y + 1)))
                img[0] = warp(img[0], transformation, output_shape=self.img_shape)

            img[0] = rotate(img[0], angle=self.max_rotation)

            if self.prob_flip_lr:
                if np.random.uniform() > self.prob_flip_lr:
                    img[0] = np.fliplr(img[0])
            if self.prob_flip_ud:
                if np.random.uniform() > self.prob_flip_ud:
                    img[0] = np.flipud(img[0])

        else:
            img = self.images[idx]

        label = self.labels[idx]
        if self.training:
            return (torch.tensor(img, requires_grad=True, dtype=torch.float32),
                    torch.tensor(label, requires_grad=True, dtype=torch.float32))
        else:
            return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)
