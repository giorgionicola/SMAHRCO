import torch
from torch import nn
from torch.utils.data import DataLoader
from CarbonFiberDataset import CarbonFiberDataset
from tqdm import tqdm, trange
import os
import numpy as np
from datetime import datetime
from collections import OrderedDict
from csv import writer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

folder_path = '/media/mullis/Volume/drapebot_dataset/2022_02_17'
dt = datetime.now()
save_folder = os.path.join(folder_path, 'training_' + dt.strftime("%Y_%m_%d_%H_%M_%S"))
os.mkdir(save_folder)
print('Saving to ', save_folder)

resolution = (128, 128)

training_dataset = CarbonFiberDataset(path=folder_path,
                                      img_shape=resolution,
                                      training=True,
                                      max_rotation=0,
                                      max_translation_x=5,
                                      max_tranlsation_y=5,
                                      prob_flip_lr=None,
                                      prob_flip_ud=None
                                      )
test_dataset = CarbonFiberDataset(path=folder_path,
                                  img_shape=resolution,
                                  training=False, )

epochs = 100
lr =  4.6049098266881956e-05
batch_size = 64

test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

training_dataloader = DataLoader(training_dataset,
                                 batch_size=batch_size,
                                 shuffle=True)


od = OrderedDict()
od['batch_norm_0'] = nn.BatchNorm2d(1)
od['conv_0'] = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
od['conv_0_act'] = nn.ReLU()
od['batch_norm_1'] = nn.BatchNorm2d(32)
od['conv_1'] = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
od['conv_1_act'] = nn.ReLU()
od['max_pool_0'] = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

od['batch_norm_2'] = nn.BatchNorm2d(64)
od['conv_2'] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
od['conv_2_act'] = nn.ReLU()
od['batch_norm_3'] = nn.BatchNorm2d(64)
od['conv_3'] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
od['conv_3_act'] = nn.ReLU()
od['max_pool_1'] = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

od['batch_norm_4'] = nn.BatchNorm2d(64)
od['conv_4'] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
od['conv_4_act'] = nn.ReLU()
od['batch_norm_5'] = nn.BatchNorm2d(64)
od['conv_5'] = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
od['conv_5_'] = nn.ReLU()
od['max_pool_2'] = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

od['flatten'] = nn.Flatten(start_dim=1)
od['linear_0'] = nn.Linear(in_features=16 * 16 * 64, out_features=500)
od['linear_0_act'] = nn.ReLU()
od['dropout_0'] = nn.Dropout(p=0.25)
od['linear_1'] = nn.Linear(in_features=500, out_features=100)
od['linear_1_act'] = nn.ReLU()
od['dropout_1'] = nn.Dropout(p=0.25)
od['linear_2'] = nn.Linear(in_features=100, out_features=3)
model = nn.Sequential(od)
model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)

best_so_far = np.inf
epoch_with_no_improvement = 0
patience = 10
for ep in trange(epochs, desc="Training"):
    training_dataset.validation = False

    running_loss = 0
    # net.model.train()
    model.train()
    for batch, (X, y) in enumerate(training_dataloader):
        # Compute prediction and loss
        X = X.to(device)
        y = y.to(device)
        # pred = net.model(X)
        pred = model(X)
        errors = y - pred
        loss = torch.mean(torch.sum(errors ** 2, dim=1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(training_dataloader)

    running_loss = 0
    # net.model.eval()
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)
            # pred = net.model(X)
            pred = model(X)
            errors = y - pred
            loss = torch.mean(torch.sum(errors ** 2, dim=1))

            running_loss += loss.item()
        avg_test_loss = running_loss / len(test_dataloader)

    if avg_test_loss < best_so_far:
        best_so_far = avg_test_loss
        epoch_with_no_improvement = 0
        model_save_folder = os.path.join(save_folder, f'epoch_{ep}')
        os.mkdir(model_save_folder)
        torch.save(model, os.path.join(model_save_folder, 'model.pth'))
        # net.save(path=model_save_folder)
        if ep == 0:
            with open(os.path.join(save_folder, 'results.csv'), 'w') as fd:
                writer_object = writer(fd)
                writer_object.writerow(['Epoch', 'test_loss'])
        with open(os.path.join(save_folder, 'results.csv'), 'a') as fd:
            writer_object = writer(fd)
            writer_object.writerow([ep, best_so_far])
    else:
        epoch_with_no_improvement += 1

    msg = f'epoch: {ep}\tTrain Loss: {avg_train_loss:.4f} \tTest Loss loss: {avg_test_loss:.4f}\t ' \
          f'Best so far: {best_so_far:.4f}'
    tqdm.write(msg)

    if epoch_with_no_improvement >= patience:
        break

print('Training is over')
