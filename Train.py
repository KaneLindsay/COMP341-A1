import math
import os
import random

import numpy
import pandas
import torch
import pandas as pd
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import skimage
import cv2
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


# Dataset class for retrieving data from resource files.
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
        image = cv2.imread(img_name)
        transform = transforms.ToTensor()
        imageTensor = transform(image)

        search_term = img_name[0:-8]

        for file in self.grasp_files_list:
            if file.startswith(search_term):
                grasps_file = os.path.join(self.datafolder, file)

        grasps = pandas.read_csv(grasps_file, sep=';', names=["x", "y", "t", "h", "w"])
        # Pick a random grasp to return as the ground truth.
        grasp = grasps.sample()
        grasp_list = grasp.values.tolist()

        return imageTensor, grasp_list[0][0], grasp_list[0][1], grasp_list[0][2], grasp_list[0][3], grasp_list[0][4]


trainSet = GraspDataset(datafolder=ROOT_DIR + "/Data/training/", datatype='train')
testSet = GraspDataset(datafolder=ROOT_DIR + "/Data/testing/")

trainLoader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=0)
testLoader = DataLoader(trainSet, batch_size=1, shuffle=True, num_workers=0)


# Direct Regression Grasp Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Initial image size -> 1024*1024*3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2)  # Output dim: 510 * 510 * 3
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)  # 254 * 254 * 64

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)  # 126*126*128
        # Second pooling -> 62 * 62 * 128
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2)  # 62*62*128
        # Repeated -> 62*62*128

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)  # 30*30*256
        # Third pooling -> 14 * 14 * 256
        self.fc1 = nn.Linear(in_features=57600, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=5)  # 5 Output Neurons: [x, y, θ, h, w]

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.relu(x))

        x = self.conv2(x)
        x = self.pool(F.relu(x))

        # Same convolution used twice.
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv3(x))

        x = self.conv4(x)
        x = self.pool(F.relu(x))

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


model = NeuralNetwork()
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def plotCorners(topLeft, topRight, bottomLeft, bottomRight):
    # Top left to top right
    plt.plot([topLeft[0], topRight[0]], [topLeft[1], topRight[1]])
    # Top left to bottom left
    plt.plot([topLeft[0], bottomLeft[0]], [topLeft[1], bottomLeft[1]])
    # Top right to bottom right
    plt.plot([topRight[0], bottomRight[0]], [topRight[1], bottomRight[1]])
    # Bottom left to bottom right
    plt.plot([bottomLeft[0], bottomRight[0]], [bottomLeft[1], bottomRight[1]])


def showImageGrasp(image, x, y, t, h, w, rotation):
    reshapedImage = image.reshape(3, 1024, 1024).permute(1, 2, 0)
    halfHeight = h / 2
    halfWidth = w / 2
    print("PERMUTED IMAGE SHAPE", reshapedImage.shape)

    plt.imshow(reshapedImage)

    topLeft = [x - halfWidth, y - halfHeight]
    topRight = [x + halfWidth, y - halfHeight]
    bottomLeft = [x - halfWidth, y + halfHeight]
    bottomRight = [x + halfWidth, y + halfHeight]

    if rotation:
        topLeftRotated = rotate([x, y], topLeft, t)
        topRightRotated = rotate([x, y], topRight, t)
        bottomLeftRotated = rotate([x, y], bottomLeft, t)
        bottomRightRotated = rotate([x, y], bottomRight, t)

        # plot the center point
        plt.plot([x], [y], 'x')
        plotCorners(topLeftRotated, topRightRotated, bottomLeftRotated, bottomRightRotated)
    else:
        plt.plot([x], [y], 'x')
        plotCorners(topLeft, topRight, bottomLeft, bottomRight)
    plt.show()
    return


# Training Loop
for epoch in range(5):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        image, x, y, t, h, w = data
        print("IMAGE SHAPE", image.shape)

        optimizer.zero_grad()

        outputs = model(image)

        targetList = [x, y, t, h, w]
        targetTensor = torch.FloatTensor(targetList)
        targetTensor = targetTensor.unsqueeze(0)

        showImageGrasp(image, x, y, t, h, w, rotation=True)

        print("OUTPUT_TENSOR: ", outputs.data)
        print("TARGET_TENSOR: ", targetTensor)

        loss = loss_fn(outputs, targetTensor)
        loss.backward()

        optimizer.step()

print('Finished Training')
