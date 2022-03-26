import os
import random

import pandas
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import skimage

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class GraspDataset(Dataset):
    def __init__(self, datafolder, datatype='train', transform=None):
        self.datafolder = datafolder
        self.image_files_list = []
        self.grasp_files_list = []
        for root, dirs, files in os.walk(ROOT_DIR + "/Data/training"):
            for file in files:
                # Count the file
                if file.endswith('RGB.png'):
                    self.image_files_list.append(root + "/" + file)
                elif file.endswith('.txt'):
                    self.grasp_files_list.append(root + "/" + file)
        # print(self.image_files_list)
        # print(self.grasp_files_list)

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_files_list[idx])
        image = skimage.io.imread(img_name)

        search_term = img_name[0:-8]

        for file in self.grasp_files_list:
            if file.startswith(search_term):
                grasps_file = os.path.join(self.datafolder, file)

        grasps = pandas.read_csv(grasps_file, sep=';', names=["x", "y", "t", "h", "w"])
        # Pick a random grasp to return as the ground truth.
        grasp = grasps.sample()
        grasp_list = grasp.values.tolist()

        return image, grasp_list[0][0], grasp_list[0][1], grasp_list[0][2], grasp_list[0][3], grasp_list[0][4]


trainSet = GraspDataset(datafolder=ROOT_DIR + "/Data/training/", datatype='train')
testSet = GraspDataset(datafolder=ROOT_DIR + "/Data/testing/")

# Direct Regression Grasp Model:
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Initial image size: 1024*1024*3
        self.first_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2)  # Output dim: 511*511*3
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)  # 127*127*64
        self.second_conv = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)  # 63*63*128
        # Second pooling -> 31*31*128
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 31*31*128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 31*31*128
        )
        self.last_conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)  # 15*15*256
        # Third pooling -> 7*7*256

        self.fc1 = nn.Linear(in_features=7 * 7 * 256, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=5)  # 5 Output Neurons: [x, y, θ, h, w]

    def forward(self, x):
        x = self.first_conv(x)
        x = self.pool(F.relu(x))

        x = self.second_conv(x)
        x = self.pool(F.relu(x))

        x = self.last_conv(x)
        x = self.pool(F.relu(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc1(x))
        x = self.fc3(x)
        return x


model = NeuralNetwork()
loss_fn = nn.MSELoss()
