import os
import sys

drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))

import ssl
import torch
import torchvision
from utils.utils import cifar10_image_transform

ssl._create_default_https_context = ssl._create_unverified_context

class Cifar10datasets:

    def __init__(self):
        self.dataset_transform = cifar10_image_transform()

    def train_dataset(self):
        return torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=self.dataset_transform)

    def validation_dataset(self):
        return torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=self.dataset_transform)