from Train import NeuralNetwork
from Train import showImageGrasp

import os
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Dataset class for retrieving TESTING data (image only) from resource files.
class GraspTestDataset(Dataset):
    def __init__(self, datafolder):
        self.datafolder = datafolder
        self.image_files_list = []

        for file in os.listdir(datafolder):
            if file.endswith('RGB.png'):
                self.image_files_list.append(file)

        print(self.image_files_list)

    def __len__(self):
        return len(self.image_files_list)

    def __getitem__(self, idx):
        img_path = os.path.join(ROOT_DIR, self.datafolder, self.image_files_list[idx])
        image = cv2.imread(img_path)
        transform = transforms.ToTensor()
        image_tensor = transform(image)

        return image_tensor


def TestNetwork():
    print('Testing...')
    testSet = GraspTestDataset(datafolder=ROOT_DIR + "/Data/testing/")
    testLoader = DataLoader(testSet, batch_size=1, shuffle=True, num_workers=4)

    # Testing
    model = NeuralNetwork()
    model.load_state_dict(torch.load("modelface"))
    model.to(device)
    for i, data in enumerate(testLoader, 0):
        image = data
        outputs, object_class = model(image.to(device))

        output_data = outputs.to('cpu').data
        output_data = output_data.tolist()[0]
        print(output_data)

        showImageGrasp(image, output_data[0], output_data[1], output_data[2], output_data[3], output_data[4],
                       rotation=True)

    print('Finished Testing.')
