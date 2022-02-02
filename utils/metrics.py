import os
import sys

drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))

import math
from utils.core.Loss import Loss
import torch
from pytorch_msssim import SSIM
from torch import Tensor
from scipy.spatial.distance import jensenshannon
from auxiliary.settings import get_device
from pytorch_msssim import ms_ssim


def scale(x: Tensor, eps: float = 0.0000000000001) -> Tensor:
    """
    Scales all values of a batched tensor between 0 and 1. Source:
    https://discuss.pytorch.org/t/how-to-efficiently-normalize-a-batch-of-tensor-to-0-1/65122/10
    """
    shape = x.shape
    x = x.to(get_device())
    x = x.reshape(x.shape[0], -1)
    x = x - x.min(1, keepdim=True)[0]
    x = x / (x.max(1, keepdim=True)[0] + Tensor([eps]).to(get_device()))
    return x.reshape(shape)


def jsd(p: list, q: list) -> float:
    """
    Jensen-Shannon Divergence (JSD) between two probability distributions as square of scipy's JS distance. Refs:
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.jensenshannon.html
    - https://stackoverflow.com/questions/15880133/jensen-shannon-divergence
    """
    return jensenshannon(p, q) ** 2


class SSIMLoss(Loss):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.__ssim_loss = SSIM(data_range=1, channel=1)

    def _compute(self, img1: Tensor, img2: Tensor) -> Tensor:
        return self._one - self.__ssim_loss(scale(img1), scale(img2)).to(self._device)


def layer_distance(base_model, model, layer):
    base_model_sd = base_model.state_dict()
    model_sd = model.state_dict()
    sub_layer_weights = []

    for key in model_sd.keys():
        if layer.lower() in key.lower():
            if 'weight' in key:
                sub_layer_weights.append(layer_weights_diff(tensor1=base_model_sd[key].float(),
                                                            tensor2=model_sd[key].float()))

    mean_sub_layer_diff = sum(sub_layer_weights) / len(sub_layer_weights)
    return 1.0 - (1 / (1 + mean_sub_layer_diff))


def layer_weights_diff(tensor1: Tensor, tensor2: Tensor) -> float:
    return float(torch.linalg.norm(tensor1 - tensor2))


# def module_weights_diff(module, model) -> float:

def ssim(img1, img2):
    data_range = img1.max() - img1.min()
    ssim_score = abs(float(ms_ssim(img1, img2,
                             data_range=data_range,
                             size_average=True)))

    if math.isnan(ssim_score):
        ssim = 0
    return ssim_score
