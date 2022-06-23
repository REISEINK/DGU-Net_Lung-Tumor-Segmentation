import numpy as np
from data import load_image_tensor, load_image
from utils import torch_dice_coef_loss, dice_coef
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import model
import sys
import os
import other_model
import gc
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda")
nb_epoch = 20
times = 1
data_folder = 'S:/dataset/LIDC'
save_path = 'S:/model'
# x_total, y_total = load_image_tensor(data_folder)
x_train, y_train = load_image(data_folder, 'train')
x_valid, y_valid = load_image(data_folder, 'valid')

model_name = 'DGU-Net'
if model_name == 'U-Net':
    model = model.UNet3D()
if model_name == 'DGU-Net':
    model = model.UNet3D_DualGCN()
if model_name == 'AttU-Net':
    model = other_model.AttU_Net()
if model_name == 'R2U-Net':
    model = other_model.R2U_Net()
if model_name == 'U-Net++':
    model = other_model.NestedUNet()

for time in range(times):

    model_dict = model.state_dict()
    weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    for key in unParalled_state_dict.keys():
        if key in model_dict:
            model_dict[key] = unParalled_state_dict[key]
    model.load_state_dict(model_dict)

    model.to(device)
    model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
    criterion = torch_dice_coef_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_losses = []
    valid_losses = []
    train_dices = []
    valid_dices = []
    best_loss = 100000
    initial_epoch = 0
    num_epoch_no_improvement = 0
    sys.stdout.flush()
    k = 0
    # N = x_total.shape[0]

    for epoch in range(initial_epoch, nb_epoch):

        # train_dataset1 = TensorDataset(x_total[0: int(N * k / 5)], y_total[0: int(N * k / 5)])
        # train_dataset2 = TensorDataset(x_total[int(N * (k + 1) / 5): N], y_total[int(N * (k + 1) / 5): N])
        # valid_dataset = TensorDataset(x_total[int(N * k / 5): int(N * (k + 1) / 5)],
        #                               y_total[int(N * k / 5): int(N * (k + 1) / 5)])
        # if k != 0:
        #     train_loader1 = DataLoader(train_dataset1, batch_size=12, shuffle=True, drop_last=True)
        # if k != 4:
        #     train_loader2 = DataLoader(train_dataset2, batch_size=12, shuffle=True, drop_last=True)
        # valid_loader = DataLoader(valid_dataset, batch_size=12, shuffle=True, drop_last=True)

        # train_dataset = TensorDataset(x_total[0: int(N * 8 / 9)], y_total[0: int(N * 8 / 9)])
        # valid_dataset = TensorDataset(x_total[int(N * 8 / 9): N], y_total[int(N * 8 / 9): N])

        train_dataset = TensorDataset(x_train, y_train)
        valid_dataset = TensorDataset(x_valid, y_valid)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, drop_last=True)

        gc.collect()
        torch.cuda.empty_cache()

        model.train()
        for batch_ndx, (x, y) in enumerate(train_loader):
            # x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(x).float().to(device)
            x, y = x.float().to(device), y.float().to(device)
            pred = model(x)
            # loss1 = FocalLoss().focal_loss(pred, y)
            # loss2 = torch_dice_coef_loss(pred, y)
            # loss = loss1 + loss2
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (batch_ndx + 1) % 10 == 0:
                print('Epoch [{}/{}], iteration {}, Batch Loss: {:.6f}'.format(
                    epoch + 1, nb_epoch, batch_ndx + 1, loss.item()))
                sys.stdout.flush()

        with torch.no_grad():
            model.eval()
            print("validating...")
            for batch_ndx, (x, y) in enumerate(train_loader):
                x, y = x.float().to(device), y.float().to(device)
                pred = model(x)
                loss = criterion(pred, y)
                train_losses.append(loss.item())

            for batch_ndx, (x, y) in enumerate(valid_loader):
                x, y = x.float().to(device), y.float().to(device)
                pred = model(x)
                loss = criterion(pred, y)
                dice = dice_coef(pred, y)
                valid_losses.append(loss.item())
                valid_dices.append(dice.item())

        # scheduler.step()

        valid_loss = np.average(valid_losses)
        valid_dice = np.average(valid_dices)
        train_loss = np.average(train_losses)
        print("Epoch {}, training loss is {:.4f}, validation loss is {:.4f}, valid dice is {:.4f}".format(epoch + 1, train_loss, valid_loss, valid_dice))
        train_losses = []
        valid_losses = []
        valid_dices = []
        if valid_loss < best_loss:
            print("Validation loss decreases from {:.4f} to {:.4f}".format(best_loss, valid_loss))
            best_loss = valid_loss
            num_epoch_no_improvement = 0
            # save model
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(save_path, model_name + "_" + str(time) + "_Last" + ".pt"))
            print("Saving model ", os.path.join(save_path, model_name + "_" + str(time) + "_Genesis" + ".pt"))
        else:
            print("Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}".
                  format(best_loss, num_epoch_no_improvement))
            num_epoch_no_improvement += 1
        if num_epoch_no_improvement == 20:
            print("Early Stopping")
            break
        sys.stdout.flush()
