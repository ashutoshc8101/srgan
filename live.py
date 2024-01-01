import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import Adam
from torchvision.utils import make_grid, save_image

from matplotlib import pyplot as plt
from PIL import Image
import os
import cv2
from typing import Any
import numpy as np
import math
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter

device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
print("Using device {}".format(device))

# Fixed random number seed
SEED = 99

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def make_directory(dir_path: str) -> None:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)



class RCB(nn.Module):

    def __init__(self, num_channels):
        super(RCB, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(num_channels),
            nn.PReLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.BatchNorm2d(num_channels)
        )

    def forward(self, lr):
        return lr + self.model(lr)

class UpSampleBlock(nn.Module):

    def __init__(self, channels, upsample_factor):
        super(UpSampleBlock, self).__init__()

        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * upsample_factor * upsample_factor,  (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upsample_factor),
            nn.PReLU(),
        )

    def forward(self, x):
        return self.upsample_block(x)


class Generator(nn.Module):

  def __init__(self, num_rcbs = 16):
    super(Generator, self).__init__()

    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = 9, stride = 1, padding = 4),
        nn.PReLU()
    )

    # High frequency information extraction RCB Layer
    self.rcb_blocks = []
    for num in range(num_rcbs):
        self.rcb_blocks.append(RCB(64))

    self.rcb_blocks = nn.Sequential(*self.rcb_blocks)

    # High frequency information fusion layer
    self.fusion = nn.Sequential(
        nn.Conv2d(64, 64, kernel_size = 3, stride = 1, padding = 1, bias = False),
        nn.BatchNorm2d(64),
    )

    self.upsample = nn.Sequential(
        UpSampleBlock(64, 2),
        UpSampleBlock(64, 2)
    )

    self.reconstruction = nn.Conv2d(64, 3, kernel_size = 9, stride = 1, padding = 4)

    # Initialize neural network weights
    for module in self.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)


  def forward(self, lr):
    x = self.conv1(lr)

    x = x + self.fusion(self.rcb_blocks(x))

    up_x = self.upsample(x)

    return self.reconstruction(up_x)


netG = Generator().to(device)
# EXP_NAME = "sr_resnet_4X"
EXP_NAME = "sr_resnet_4X_L1"

# Create the folder where the model weights are saved
samples_dir = os.path.join("samples", EXP_NAME)
results_dir = os.path.join("results", EXP_NAME)
make_directory(samples_dir)
make_directory(results_dir)

load_epoch = 5

load_path = os.path.join(samples_dir, "epoch_{}.pth".format(load_epoch))
checkpoint = torch.load(load_path)

netG.load_state_dict(checkpoint["state_dict"])

print("Loaded checkpoint with epoch {}, PSNR {:.4f} and SSIM {:.4f}".format(
    checkpoint['epoch'], checkpoint["psnr"], checkpoint["ssim"]))

# # %%
# matlab_lr = transforms.ToTensor()(Image.open("./profile_pic.png"))

# plt.imshow(matlab_lr.permute(1, 2, 0))

# matlab_lr = matlab_lr.to(device)
# # matlab_lr.unsqueeze(0)
# sr_img = netG(matlab_lr.unsqueeze(0))

# sr_img = sr_img.squeeze(0).detach().cpu()
# save_image(sr_img, "./test.png")

# plt.imshow(sr_img.permute(1, 2, 0))

import time
import numpy as np

netG.eval()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
  print("Cannot open camera.")
count = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    # cv2.imshow('frame', frame)
    # frame = cv2.resize(frame, (320, 240))
    # cv2.imshow('input', frame)


    with torch.no_grad():
        frame = torch.tensor(frame) / 255.
        frame = frame.permute(2, 0, 1).unsqueeze(0)
        frame = frame.to(device)

        output = netG(frame)

        output = output[0].permute(1, 2, 0)
        if count == 1000:
            break

    # cv2.imshow('sr', out)

    # break
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
out.release()
cv2.destroyAllWindows()