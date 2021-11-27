import os
import sys

drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))
import numpy as np
from skimage.transform import resize
from scipy.spatial.distance import cdist
from auxiliary.settings import batch_size

class Sidu:

    def __init__(self, model, model_type='inception_v3'):

        self.model = model
        self.model_type = model_type
        self.input_shape = self.this_input_shape()

    def this_input_shape(self):
        if self.model_type == 'inception_v3':
            input_shape = (299, 299)
        else:
            input_shape = (224, 224)
        return input_shape



def generate_masks_conv_output(input_size, last_conv_output, s=8):
    cell_size = np.ceil(np.array(input_size) / s)
    up_size = s * cell_size
    grid = np.rollaxis(last_conv_output, 2, 0)

    N = len(grid)
    masks = np.empty((*input_size, N))

    for i in range(N):
        conv_out = last_conv_output[:, :, i]
        conv_out = conv_out > 0.5
        conv_out = conv_out.astype('float32')
        final_resize = resize(conv_out, up_size, order=1, mode='reflect',
                              anti_aliasing=False)
        masks[:, :, i] = final_resize
    #        masks = masks.reshape(-1, *input_size, 1)
    return masks, grid, cell_size, up_size


def kernel(d, kernel_width):
    return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))


def sim_differences(pred_org, preds):
    diff = abs(pred_org - preds)
    weights = kernel(diff, 0.25)
    return weights, diff


def normalize(array):
    return (array - array.min()) / (array.max() - array.min() + 1e-13)

def uniqness_measure(masks_predictions):
    sum_all_cdist =(cdist(masks_predictions, masks_predictions)).sum(axis=1)
    sum_all_cdist = normalize(sum_all_cdist)
    return sum_all_cdist


def explain_SIDU(model, inp, N, p1, masks, input_size):
    preds = []
    masked = inp * masks
    pred_org = model.predict(inp)

    for i in range(0, N, batch_size):
        preds.append(model.predict(masked[i:min(i + batch_size, N)]))
    preds = np.concatenate(preds)

    weights, diff = sim_differences(pred_org, preds)
    interactions = uniqness_measure(preds)
    new_interactions = interactions.reshape(-1, 1)
    diff_interactions = np.multiply(weights, new_interactions)

    sal = diff_interactions.T.dot(masks.reshape(N, -1)).reshape(-1, *input_size)
    sal = sal / N / p1
    return sal, weights, new_interactions, diff_interactions, pred_org
