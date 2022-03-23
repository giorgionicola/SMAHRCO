import optuna
from optuna.trial import Trial, TrialState
import torch
from torch import nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from CarbonFiberDataset import CarbonFiberDataset
from tqdm import tqdm, trange
import numpy as np
from sklearn.model_selection import KFold
from collections import OrderedDict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
folder_path = '/media/mullis/Volume/drapebot_dataset/2022_02_17'


def objective(trial: Trial):
    n_folds = 5
    epochs = 100
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_int('batch size', low=32, high=256, log=True)

    training_dataset = CarbonFiberDataset(path=folder_path,
                                          img_shape=(128, 128),
                                          training=True,
                                          max_rotation=10,
                                          max_translation_x=10,
                                          max_tranlsation_y=10,
                                          prob_flip_lr=None,
                                          prob_flip_ud=None
                                          )

    od = OrderedDict()
    od['batch_norm_0'] = nn.BatchNorm2d(1)
    od['conv_0'] = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    od['conv_0_act'] = nn.ReLU()
    od['batch_norm_1'] = nn.BatchNorm2d(32)
    od['conv_1'] = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    od['conv_1_act'] = nn.ReLU()
    od['max_pool_0'] = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

    od['batch_norm_2'] = nn.BatchNorm2d(64)
    od['conv_2'] = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    od['conv_2_act'] = nn.ReLU()
    od['batch_norm_3'] = nn.BatchNorm2d(128)
    od['conv_3'] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    od['conv_3_act'] = nn.ReLU()
    od['max_pool_1'] = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

    od['batch_norm_4'] = nn.BatchNorm2d(128)
    od['conv_4'] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    od['conv_4_act'] = nn.ReLU()
    od['batch_norm_5'] = nn.BatchNorm2d(128)
    od['conv_5'] = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding=(2, 2))
    od['conv_5_act'] = nn.ReLU()
    od['max_pool_2'] = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))

    od['flatten'] = nn.Flatten(start_dim=1)
    od['linear_0'] = nn.Linear(in_features=36 * 128, out_features=256)
    od['linear_0_act'] = nn.ReLU()
    od['dropout_0'] = nn.Dropout(p=0.25)
    od['linear_1'] = nn.Linear(in_features=256, out_features=128)
    od['linear_1_act'] = nn.ReLU()
    od['dropout_1'] = nn.Dropout(p=0.25)
    od['linear_2'] = nn.Linear(in_features=128, out_features=32)
    od['linear_2_act'] = nn.ReLU()
    od['dropout_2'] = nn.Dropout(p=0.25)
    od['linear_3'] = nn.Linear(in_features=32, out_features=3)

    best_losses = [np.inf for _ in range(n_folds)]
    best_epoch = [np.inf for _ in range(n_folds)]
    splits = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(splits.split(training_dataset)):
        print(f'Fold : {fold + 1}')
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        training_dataloader = DataLoader(training_dataset,
                                         batch_size=batch_size,
                                         sampler=train_sampler)
        validation_dataloader = DataLoader(training_dataset,
                                           batch_size=batch_size,
                                           sampler=valid_sampler, )

        model = nn.Sequential(od)
        model.to(device)

        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=1e-5)

        best_so_far = np.inf
        epoch_with_no_improvement = 0
        patience = 10
        for ep in trange(epochs, desc="Training"):

            running_loss = 0
            model.train()
            for batch, (X, y) in enumerate(training_dataloader):
                # Compute prediction and loss
                X = X.to(device)
                y = y.to(device)
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
            model.eval()
            with torch.no_grad():
                for batch, (X, y) in enumerate(validation_dataloader):
                    # Compute prediction and loss
                    X = X.to(device)
                    y = y.to(device)
                    pred = model(X)
                    errors = y - pred
                    loss = torch.mean(torch.sum(errors ** 2, dim=1))

                    running_loss += loss.item()
                avg_valid_loss = running_loss / len(validation_dataloader)

            if avg_valid_loss < best_so_far:
                best_so_far = avg_valid_loss
                epoch_with_no_improvement = 0
                best_ep = ep
            else:
                epoch_with_no_improvement += 1

            msg = f'Fold: {fold + 1}\tEpoch: {ep}\tTrain Loss: {avg_train_loss:.4f}\tValidation Loss: {avg_valid_loss:.4f}\t' \
                  f'Best so far: {best_so_far:.4f}'
            tqdm.write(msg)

            if epoch_with_no_improvement >= patience:
                best_losses[fold] = best_so_far
                best_epoch[fold] = best_ep
                break

    trial.report(np.mean(best_losses), int(np.mean(best_ep)))

    return np.mean(best_losses)


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, timeout= 60*60*12, show_progress_bar=False)
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
