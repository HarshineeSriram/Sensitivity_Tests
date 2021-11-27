import os
import json
import torch
import numpy as np
import torchvision
from auxiliary.paths import path_to_imagenet_labels
from utils.visualization import visualize_image_attr

"""--------------------------------------------------------------
  Image Classification with ImageNet
--------------------------------------------------------------"""


def normalize(x):
    x = x.detach().numpy()
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x)))


def imagenet_normalization():
    return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])


def imagenet_train_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        imagenet_normalization()
    ])


def imagenet_test_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        imagenet_normalization()
    ])


def imagenet_class_from_idx(class_number):
    with open(path_to_imagenet_labels) as f:
        labels_json = json.load(f)
    return labels_json[class_number]


"""--------------------------------------------------------------
  Image Classification with Cifar10
--------------------------------------------------------------"""


def cifar10_normalization():
    return torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))


def cifar10_image_transform():
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           cifar10_normalization()])


def cifar10_class_from_idx(class_number):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return classes[class_number]


"""--------------------------------------------------------------
  Explanation methods utilities
--------------------------------------------------------------"""


def original_img_transform(img):
    return np.transpose((img.squeeze().cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))


def attribution_transform(attribution):
    return np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))


def save_explanation_viz(original_image, explanation, destination, file_name, **kwargs):
    this_viz, indices = visualize_image_attr(explanation, original_image, **kwargs)
    this_viz.savefig(os.path.join(destination, file_name))
