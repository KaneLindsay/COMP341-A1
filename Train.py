import math
import os
import random
import pandas
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt

# Training configs
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_CLASSES = 10
CLASS_NAMES = os.listdir(ROOT_DIR + "/Data/training/")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Map each class to a unique ground truth array, e.g. [0,0,0,1,0,0,0,0,0,0]
classMap = {}
class_array_zeros = []
for i in range(NUM_CLASSES):
    class_array_zeros.append(0)
for c in range(NUM_CLASSES):
    class_gt = class_array_zeros.copy()
    class_gt[c] = 1
    classMap[CLASS_NAMES[c]] = class_gt
print(classMap)


# Dataset class for retrieving TRAINING data from resource files.
class GraspTrainDataset(Dataset):
    def __init__(self, datafolder):
        self.datafolder = datafolder
        self.image_files_list = []
        self.grasp_files_list = []

        for root, dirs, files in os.walk(datafolder):
            for file in files:
                # Find RGB and .txt (grasp) files
                if file.endswith('RGB.png'):
                    self.image_files_list.append(root + "/" + file)
                elif file.endswith('.txt'):
                    self.grasp_files_list.append(root + "/" + file)

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_path = self.image_files_list[idx]
        image = cv2.imread(img_path)
        transform = transforms.ToTensor()
        image_tensor = transform(image)

        # Find corresponding grasp file for image
        search_term = img_path[0:-8]
        object_class = search_term.split("_")[1]

        for file in self.grasp_files_list:
            if file.startswith(search_term):
                grasps_file = os.path.join(self.datafolder, file)

        grasps = pandas.read_csv(grasps_file, sep=';', names=["x", "y", "t", "h", "w"])

        return image_tensor, grasps.to_numpy(), object_class


# FIXME: Redo the layer size maths - it's still slightly off.
# Direct Regression Grasp Model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Initial image size -> 1024*1024*3
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2)  # Output dim: 510 * 510 * 64
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)  # 254 * 254 * 64

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)  # 126*126*128
        # Second pooling -> 62 * 62 * 128
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2)  # 62*62*128
        # Repeated -> 62*62*128

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2)  # 30*30*256
        # Third pooling -> 14 * 14 * 256
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(in_features=57600, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=5)  # 5 Output Neurons: [x, y, θ, h, w]
        self.fc3class = nn.Linear(in_features=512, out_features=NUM_CLASSES)  # 10 Output Neurons (Class Num)
        self.softmax = torch.nn.Softmax(dim=1)

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
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        g = self.fc3(x)

        c = self.fc3class(x)
        c = self.softmax(c)

        return g, c


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


def plotCorners(topLeft, topRight, bottomLeft, bottomRight):
    # Top left to top right
    plt.plot([topLeft[0], topRight[0]], [topLeft[1], topRight[1]], color="yellow")
    # Top left to bottom left
    plt.plot([topLeft[0], bottomLeft[0]], [topLeft[1], bottomLeft[1]], color="green")
    # Top right to bottom right
    plt.plot([topRight[0], bottomRight[0]], [topRight[1], bottomRight[1]], color="green")
    # Bottom left to bottom right
    plt.plot([bottomLeft[0], bottomRight[0]], [bottomLeft[1], bottomRight[1]], color="yellow")


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


def findClosestGrasp(possible_grasps, x1, y1):
    closest_grasp = possible_grasps[0]
    closest_dist = 100000
    for g in range(len(possible_grasps)):
        x2 = possible_grasps[g][0]
        y2 = possible_grasps[g][1]
        dist = abs(x1 - x2) + abs(y1 - y2)
        if dist < closest_dist:
            closest_dist = dist
            closest_grasp = possible_grasps[g]
    return closest_grasp


def rectangleMetricEval(generated, ground_truth):
    similarity_threshold = 10
    intersect = 0
    if abs(ground_truth[2] - generated[2]) > 30:
        return False
    for x in range(len(generated)):
        if x != 2:
            if abs(ground_truth[x] - generated[x]) > similarity_threshold:
                intersect += 1

    return abs(intersect / (8 - intersect)) > 0.25


# Training Loop
def TrainNetwork(num_epochs=100):
    train_set = GraspTrainDataset(datafolder=ROOT_DIR + "/Data/training/")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
    model = NeuralNetwork()
    model.to(device)
    box_loss_fn = nn.MSELoss()
    class_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.001)
    print('Starting training...')
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        rectangle_metric_pass = 0
        rectangle_metric_fail = 0
        print("Training Epoch: ", epoch)
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [image, x-coord, y-coord, theta (rotation), height, width]
            image, gt_grasps, object_class = data

            grasp_prediction, class_prediction = model(image.to(device))
            grasp_prediction_list = grasp_prediction.to('cpu').data.tolist()[0]

            gt_grasps = gt_grasps.numpy()[0]

            grasp_target_list = gt_grasps[random.randrange(0, len(gt_grasps))]
            # grasp_list = findClosestGrasp(gt_grasps, grasp_prediction_list[0], grasp_prediction_list[1])

            grasp_target_tensor = torch.FloatTensor(grasp_target_list)
            grasp_target_tensor = grasp_target_tensor.unsqueeze(0)

            class_target = classMap[object_class[0]]
            class_target_tensor = torch.FloatTensor(class_target)
            class_target_tensor = class_target_tensor.unsqueeze(0)

            if rectangleMetricEval(grasp_prediction_list, grasp_target_list):
                rectangle_metric_pass += 1
            else:
                rectangle_metric_fail += 1
            print("Rectangle Metric Pass-Rate: ",
                  round(rectangle_metric_pass/((rectangle_metric_pass+rectangle_metric_fail)*100), 2)
                  )

            # Train on only image classification for _ epochs.
            if epoch >= 20:
                # Grasp regression
                box_loss = box_loss_fn(grasp_prediction.to('cpu'), grasp_target_tensor)
                box_loss.backward(retain_graph=True)

            # Image classification
            # print(class_prediction.to('cpu'))
            # print(class_target_tensor)
            class_loss = class_loss_fn(class_prediction.to('cpu'), class_target_tensor)
            class_loss.backward()

            # print("\nBox Loss: ", box_loss)
            # print("Class Loss: ", class_loss)

            optimizer.step()

            optimizer.zero_grad()

            if epoch == num_epochs - 1:
                output_data = grasp_prediction.to('cpu').data.tolist()[0]
                print(output_data)
                print("PING")
                showImageGrasp(image, output_data[0], output_data[1], output_data[2], output_data[3], output_data[4],
                               rotation=True)

    torch.save(model.state_dict(), "modelface")
    print('Finished Training.')

