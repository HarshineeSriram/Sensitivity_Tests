import os
import json
import torch
import numpy as np
import torchvision
from scipy.spatial.distance import cdist
from auxiliary.paths import path_to_imagenet_labels
from utils.visualization import visualize_image_attr

"""--------------------------------------------------------------
  Temporary functions
--------------------------------------------------------------"""


def convert_tensor_str_to_float(string):
    pos1, pos2 = (-1, -1)
    for i in range(len(string)):
        if '(' == string[i]:
            pos1 = i + 1
        if ',' == string[i]:
            pos2 = i
            break
        if ')' == string[i]:
            pos2 = i
            break

    return float(string[pos1:pos2])

"""--------------------------------------------------------------
  Image Utilities
--------------------------------------------------------------"""


def normalize(x):
    x = x.detach().numpy()
    return np.array((x - np.min(x)) / (np.max(x) - np.min(x) + 1e-13))


def image_normalization():
    return torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])


def get_input_shape(model_type):
    if model_type == 'inception_v3':
        input_shape = (299, 299)
    else:
        input_shape = (224, 224)
    return input_shape


"""--------------------------------------------------------------
  Image Classification with ImageNet
--------------------------------------------------------------"""


def imagenet_train_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        image_normalization()
    ])


def imagenet_test_transform():
    return torchvision.transforms.Compose([
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor(),
        image_normalization()
    ])


def imagenet_class_from_idx(class_number):
    with open(path_to_imagenet_labels) as f:
        labels_json = json.load(f)
    return labels_json[class_number]


"""--------------------------------------------------------------
  Image Classification with Cifar10
--------------------------------------------------------------"""


def cifar10_image_transform():
    return torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                           image_normalization()])


def cifar10_class_from_idx(class_number):
    classes = ('plane', 'car', 'bird',
               'cat', 'deer', 'dog',
               'frog', 'horse', 'ship',
               'truck')
    return classes[class_number]


"""--------------------------------------------------------------
  Explanation methods utilities
--------------------------------------------------------------"""


def original_img_transform(img):
    return np.transpose((img.squeeze().cpu().detach().numpy() / 2) + 0.5, (1, 2, 0))


def attribution_transform(attribution):
    return np.transpose(attribution.squeeze().cpu().detach().numpy(), (1, 2, 0))


def get_last_layer(model_type='inception_v3', model=None):
    if model_type == 'inception_v3':
        return model.Mixed_7c
    elif model_type == 'mobilenet_v2':
        return model._modules['features'][17]
    elif model_type == 'vgg_16':
        return model._modules['features'][28]
    elif 'resnet' in model_type:
        return model.layer4


def get_last_layer_outputs_tf(model_type='inception_v3', model=None):
    if model_type == 'inception_v3':
        return model.get_layer('conv2d_93').output
    elif model_type == 'mobilenet_v2':
        return model.get_layer('Conv_1').output
    elif model_type == 'vgg_16':
        return model.get_layer('block5_conv3').output


def save_explanation_viz(original_image, explanation, destination, file_name, **kwargs):
    this_viz, indices = visualize_image_attr(explanation, original_image, **kwargs)
    this_viz.savefig(os.path.join(destination, file_name))


def sidu_kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


def sidu_sim_differences(pred_org, preds):
    diff = abs(pred_org - preds)
    weights = sidu_kernel(diff, 0.25)
    return weights, diff


def sidu_uniqness_measure(masks_predictions):
    sum_all_cdist = cdist(masks_predictions, masks_predictions).sum(axis=1)
    sum_all_cdist = normalize(sum_all_cdist)
    return sum_all_cdist
