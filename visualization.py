import numpy as np
from data import load_image
from utils import torch_dice_coef_loss, dice_coef, iou
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import model
import sys
import matplotlib.pyplot as plt


def plot_image_truth_prediction(x, y, p, rows=12, cols=12):
    x, y, p = np.squeeze(x), np.squeeze(y), np.squeeze(p > 0.5)
    plt.rcParams.update({'font.size': 30})
    plt.figure(figsize=(25 * 3, 25))

    large_image = np.zeros((rows * x.shape[0], cols * x.shape[1]))
    for b in range(rows * cols):
        large_image[(b // rows) * x.shape[0]:(b // rows + 1) * x.shape[0],
        (b % cols) * x.shape[1]:(b % cols + 1) * x.shape[1]] = np.transpose(np.squeeze(x[:, :, b]))
    plt.subplot(1, 3, 1)
    plt.imshow(large_image, cmap='gray', vmin=0, vmax=1);
    plt.axis('off')

    large_image = np.zeros((rows * x.shape[0], cols * x.shape[1]))
    for b in range(rows * cols):
        large_image[(b // rows) * y.shape[0]:(b // rows + 1) * y.shape[0],
        (b % cols) * y.shape[1]:(b % cols + 1) * y.shape[1]] = np.transpose(np.squeeze(y[:, :, b]))
    plt.subplot(1, 3, 2)
    plt.imshow(large_image, cmap='gray', vmin=0, vmax=1);
    plt.axis('off')

    large_image = np.zeros((rows * p.shape[0], cols * p.shape[1]))
    for b in range(rows * cols):
        large_image[(b // rows) * p.shape[0]:(b // rows + 1) * p.shape[0],
        (b % cols) * p.shape[1]:(b % cols + 1) * p.shape[1]] = np.transpose(np.squeeze(p[:, :, b]))
    plt.subplot(1, 3, 3)
    plt.imshow(large_image, cmap='gray', vmin=0, vmax=1);
    plt.axis('off')

    plt.show()


device = torch.device("cuda")
data_folder = 'S:/dataset/LIDC'
save_path = 'S:/model'
x_test, y_test = load_image(data_folder, 'test')

model_name = 'DGU-Net'
model = model.UNet3D_DualGCN()

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


test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16)
x_v = x_test.numpy()
y_v = y_test.numpy()
p_v = np.zeros_like(x_v)

model.eval()

with torch.no_grad():
    for batch_ndx, (x, y) in enumerate(test_loader):
        x, y = x.float().to(device), y.float().to(device)
        p = model(x)
        p_v[batch_ndx * 16: batch_ndx * 16 + 16] = p.cpu()

for i in range(0, x_v.shape[0], 80):
    plot_image_truth_prediction(x_v[i], y_v[i], p_v[i], rows=1, cols=1)

