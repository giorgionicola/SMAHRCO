import time

import torch
from CarbonFiberDataset import CarbonFiberDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folder_path = '/media/mullis/Volume/drapebot_dataset/2022_02_17'

resolution = (128, 128)

training_dataset = CarbonFiberDataset(path=folder_path,
                                      img_shape=resolution,
                                      training=True,
                                      max_rotation=0,
                                      max_translation_x=0,
                                      max_tranlsation_y=0,
                                      prob_flip_lr=None,
                                      prob_flip_ud=None
                                      )
test_dataset = CarbonFiberDataset(path=folder_path,
                                  img_shape=resolution,
                                  training=False, )

### Model 2

model1 = torch.load('/media/mullis/Volume/drapebot_dataset/2022_02_17/training_2022_03_14_16_55_23/epoch_17/model.pth')
# model1.to(device)
model1.eval()

### Model 2

model2 = torch.load('/media/mullis/Volume/drapebot_dataset/2022_02_17/training_2022_03_14_17_26_23/epoch_28/model.pth')
# model2.to(device)
model2.eval()

#### Model 3

model3 = torch.load('/media/mullis/Volume/drapebot_dataset/2022_02_17/training_2022_03_21_15_26_42/epoch_32/model.pth')
model3.eval()
# model3.to(device)

avg_err = 0
total_time = 0

with torch.no_grad():
    for i in range(len(test_dataset)):
        x, y = test_dataset.__getitem__(i)
        x = x[None, :]
        x = x.to(device)
        t0 = time.time()
        pred = model1(x)/3 + model2(x)/3 + model3(x)/3
        total_time += time.time() - t0
        pred = pred.to('cpu')

        err = y - pred[0]


        avg_err += torch.sum(err**2).detach()

avg_err /= len(test_dataset)
avg_time = total_time/len(test_dataset)

print(avg_err.numpy())
print(avg_time)





