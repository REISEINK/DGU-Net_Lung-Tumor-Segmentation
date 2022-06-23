import numpy as np
from data import load_image
from utils import torch_dice_coef_loss, dice_coef, iou
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import model
import sys
import other_model


device = torch.device("cuda")
data_folder = 'S:/dataset/LIDC'
save_path = 'S:/model'
x_test, y_test = load_image(data_folder, 'test')

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

model_dict = model.state_dict()
weight_dir = 'S:/model/DGU-Net_2_Genesis.pt'
checkpoint = torch.load(weight_dir)
state_dict = checkpoint['state_dict']
unParalled_state_dict = {}
for key in state_dict.keys():
    unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
model.load_state_dict(unParalled_state_dict)

model.to(device)
model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
criterion = torch_dice_coef_loss


for i in range(10):

    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    test_losses = []
    test_dices = []
    test_ious = []
    best_loss = 100000
    initial_epoch = 0
    num_epoch_no_improvement = 0
    sys.stdout.flush()

    model.eval()
    if i == 0 :
        print("Testing...")

    with torch.no_grad():
        for batch_ndx, (x, y) in enumerate(test_loader):
            x, y = x.float().to(device), y.float().to(device)
            pred = model(x)

            loss = criterion(pred, y)
            dice = dice_coef(pred, y)
            IoU = iou(pred, y)
            test_losses.append(loss.item())
            test_dices.append(dice.item())
            test_ious.append(IoU.item())

    test_loss = np.average(test_losses)
    test_dice = np.average(test_dices)
    test_iou = np.average(test_ious)

    print(str(i+1) + ": Test loss is {:.4f}, test dice is {:.4f}, test iou is {:.4f}".format(test_loss, test_dice, test_iou))
