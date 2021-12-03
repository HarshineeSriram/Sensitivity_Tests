from auxiliary.settings import batch_size, num_workers, seed
from data.dataloaders.imagenet_datasets import Imagenet_Train_Val, Imagenet_test
import os
import sys
import torch
import numpy as np

drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))

# --------------------------------------------------------------------------------------------
torch.manual_seed(seed)
np.random.seed(seed)
# --------------------------------------------------------------------------------------------


class ImagenetDataloader:

    def __init__(self) -> None:
        self.train_val = Imagenet_Train_Val()

        self.train_dataset = self.train_val.train_dataset()
        self.validation_dataset = self.train_val.validation_dataset()
        self.test_dataset = Imagenet_test()
        self.external_test_dataset = self.train_val.external_test_dataset()

    def train_loader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

    def validation_loader(self):
        return torch.utils.data.DataLoader(self.validation_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

    def test_loader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=num_workers)

    def external_test_loader(self):
        return torch.utils.data.DataLoader(self.external_test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=num_workers)


if __name__ == '__main__':
    '''
    print("Hello")
    p = Imagenet_dataloader()
    train_loader = p.train_loader()
    validation_loader = p.validation_loader()
    test_loader = p.test_loader()
    this_path = r'E:\sensitivity_tests\tasks\image_classification\data\dataloaders\imgs'
    idx = 0

    #for idx, img in enumerate(test_loader):
    #for img, label in validation_loader:
        if idx<5:
            print(img.shape)
            #this_label = list(p.validation_dataset.class_to_idx.keys())[list(p.validation_dataset.class_to_idx.values()).index(label)]
            idx += 1
            plt.imsave(fname=os.path.join(this_path, str(idx)+'.jpeg'), 
                        arr=normalize(np.array(torch.permute(img[0], (1, 2, 0)))))
        else:
            break
    '''