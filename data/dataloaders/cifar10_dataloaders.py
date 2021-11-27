import os, sys
drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))
from auxiliary.settings import batch_size, num_workers, seed
from data.dataloaders.cifar10_datasets import Cifar10datasets
import torch
import numpy as np

# --------------------------------------------------------------------------------------------
torch.manual_seed(seed)
np.random.seed(seed)


# --------------------------------------------------------------------------------------------

class Cifar10Dataloader:

    def __init__(self):
        self.datasets = Cifar10datasets()
        self.train_dataset = self.datasets.train_dataset()
        self.validation_dataset = self.datasets.validation_dataset()

    def train_loader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

    def validation_loader(self):
        return torch.utils.data.DataLoader(self.validation_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=num_workers)

if __name__ == '__main__':

    dataloader = Cifar10Dataloader()
    for data in dataloader.train_loader():
        images, labels = data
        print(images.shape)
        print(cifar10_class_from_idx(labels))
        break