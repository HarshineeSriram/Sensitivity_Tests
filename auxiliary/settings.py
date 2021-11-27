import torch
from models.inception_v3.inceptionv3 import inception_v3
from models.mobilenet_v3large.mobilenetv3large import mobilenet_v3_large
from models.vgg16.vgg16 import vgg16

# Global variables and functions
seed = 27


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Task: Image Classification

batch_size = 1
num_workers = 1
models = {'inception_v3': inception_v3,
          'mobilenet_v3': mobilenet_v3_large,
          'vgg_16': vgg16}

model_paths = {
    'inception_v3': r'E:/sensitivity_tests/models/inception_v3/inceptionv3_baseline.pth',
    'mobilenet_v3': r'E:/sensitivity_tests/models/mobilenet_v3large/mobilenetv3large_baseline.pth',
    'vgg_16': r'E:/sensitivity_tests/models/vgg16/vgg16_baseline.pth'
}
