import os
import sys

drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))

from utils.utils import imagenet_train_transform, imagenet_test_transform
import torch
import torchvision
import natsort
from PIL import Image

# --------------------------------------------------------------------------------------------
imagenet_path = os.path.join(drive, os.sep, *path.split(os.sep)[:-1], 'datasets', 'imagenet')
train_path = os.path.join(imagenet_path, 'data', 'train')
validation_path = os.path.join(imagenet_path, 'data', 'val')
test_path = os.path.join(imagenet_path, 'data', 'test')


# --------------------------------------------------------------------------------------------

# Using Pytorch's ImageFolder function to access the train and validation folders
class Imagenet_Train_Val():

    def __init__(self) -> None:
        self.train_dir = train_path
        self.train_transform = imagenet_train_transform()
        self.validation_dir = validation_path
        self.validation_transform = imagenet_test_transform()

    def train_dataset(self):
        return torchvision.datasets.ImageFolder(self.train_dir, self.train_transform)

    def validation_dataset(self):
        return torchvision.datasets.ImageFolder(self.validation_dir, self.validation_transform)


# Describing a custom dataloader for the test folder
class Imagenet_test(torch.utils.data.Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.main_dir = test_path
        self.transform = imagenet_test_transform()
        all_imgs = os.listdir(test_path)
        self.total_imgs = natsort.natsorted(all_imgs)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert('RGB')
        tensor_image = self.transform(image)
        return tensor_image
