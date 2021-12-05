import os
import sys
drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))
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


def layer_weights_diff(tensor1: Tensor, tensor2: Tensor) -> float:
    return float(torch.linalg.norm(tensor1 - tensor2))


def ssim_loss(img1, img2):

    data_range = img1.max() - img2.min()
    return 1 - ms_ssim(img1, img2,
                       data_range=data_range,
                       size_average=True)
