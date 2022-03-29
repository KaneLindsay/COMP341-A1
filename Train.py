import os
import random

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

        self.fc1 = nn.Linear(in_features=50176, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=5)  # 5 Output Neurons: [x, y, θ, h, w]

    def forward(self, x):
        x = self.first_conv(x)
        x = self.pool(F.relu(x))

        x = self.second_conv(x)
        x = self.pool(F.relu(x))
        
        #x = self.conv_stack(x)
        
        x = self.last_conv(x)
        x = self.pool(F.relu(x))

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


<<<<<<< Updated upstream
=======
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
    # print("PERMUTED IMAGE SHAPE", reshapedImage.shape)

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
# generated and actual in format (x,y,t,h,w)


def rectangleMetricEval(generated, groundTruth):
    similarityThreshold = 10 
    intersect = 0
    if abs(groundTruth[2]-generated[2]) > 30:
        return False
    for x in range(len(generated)):
        if x != 2:
            if abs(groundTruth[x]-generated[x]) > similarityThreshold:
                intersect += 1
                
    return abs(intersect/(8-intersect)) > 0.25
                
                
    

            
            
            
    
    
    
    
    

>>>>>>> Stashed changes
model = NeuralNetwork()
loss_fn = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.05)
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainLoader, 0):
        # get the inputs; data is a list of [inputs, labels]
        image, x, y, t, h, w = data
        #print(image.shape)


        # zero the parameter gradients
        targetArray = [x, y, t, h, w]
        target = torch.FloatTensor(targetArray) 
        #print(target)
        # forward + backward + optimize
        outputs = model(image)
        print(outputs)

        loss = loss_fn(outputs, target)
        optimizer.zero_grad()

<<<<<<< Updated upstream
        loss.backward()
=======
        loss = loss_fn(outputs, targetTensor)




        loss.backward()
        

        # print("CURRENT LOSS: ", loss.data)
        # print("\nOUTPUT_TENSOR: ", outputs.data)
        # print("TARGET_TENSOR: ", targetTensor)

>>>>>>> Stashed changes
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

<<<<<<< Updated upstream
print('Finished Training')
=======
print('Finished Training.')

print('Evaluating...')
evaluations = []
with torch.no_grad():
    model.eval()
    for i, data in enumerate(testLoader, 0):
        image, x, y, t, h, w = data
        outputs = model(image)

        targetList = [x, y, t, h, w]
        targetTensor = torch.FloatTensor(targetList)
        targetTensor = targetTensor.unsqueeze(0)

        output_data = outputs.data
        output_data = output_data.tolist()[0]
        
        showImageGrasp(image, output_data[0], output_data[1], output_data[2], output_data[3], output_data[4], rotation=True)
        showImageGrasp(image, x, y, t, h, w, rotation=True)
        if rectangleMetricEval(output_data,targetList):
            evaluations.append(1)
        else:
            evaluations.append(0)

        print(output_data)


print('Finished Evaluating.')
accuracy = sum(evaluations)/len((evaluations))
print("Overall Accuracy: ", accuracy)


>>>>>>> Stashed changes
