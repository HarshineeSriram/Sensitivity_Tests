import os
import sys

drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))

import numpy as np

from skimage.transform import resize
from scipy.spatial.distance import cdist

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inceptionv3
from data.dataloaders.imagenet_dataloaders import ImagenetDataloader

from auxiliary.settings import batch_size, models_tf
from utils.utils import sidu_sim_differences, sidu_uniqness_measure


class Sidu:

    def __init__(self, img, model_type='inception_v3'):
        self.input_img = img[0]
        print("\n\n\n\n\n\n\n before resize", np.array(self.input_img).shape)
        self.model_type = model_type
        self.input_shape = self.this_input_shape()

    def get_last_layer(self):
        if self.model_type == 'mobilenet_v2':
            return self.model.get_layer('Conv_1').output
        elif self.model_type == 'inception_v3':
            return self.model.get_layer('conv2d_93').output
        elif self.model_type == 'vgg_16':
            return self.model.get_layer('block5_conv3').output

    def get_model(self):
        self.model = models_tf[self.model_type]()
        self.model_last_layer = self.get_last_layer()

    def process_input_image(self):
        self.img = tf.image.resize(self.input_img, size=self.this_input_shape())
        print("\n\n after resize", np.array(self.img).shape)
        x = image.img_to_array(self.img)
        x = np.expand_dims(x, axis=0)
        if self.model_type == 'mobilenet_v2':
            self.transformed_input = preprocess_mobilenetv2(x)
        elif self.model_type == 'inception_v3':
            self.transformed_input = preprocess_inceptionv3(x)
        elif self.model_type == 'vgg_16':
            self.transformed_input = preprocess_vgg16(x)

    def get_input_shape(self):
        if self.model_type == 'inception_v3':
            input_shape = [299, 299]
        else:
            input_shape = [224, 224]
        return input_shape

    def get_features_model(self):
        self.features_model = Model(inputs=self.model.input, outputs=self.model_last_layer)

    def get_last_conv_output(self):
        print(np.array(self.transformed_input).shape)
        feature_activation_maps = self.features_model.predict(self.transformed_input)
        self.last_conv_output = np.squeeze(feature_activation_maps)

    def generate_masks_conv_output(self, s=8):
        self.cell_size = np.ceil(np.array(self.input_shape) / s)
        self.up_size = s * self.cell_size
        self.grid = np.rollaxis(self.last_conv_output, 2, 0)

        N = len(self.grid)
        self.masks = np.empty((*self.input_shape, N))

        for i in range(N):
            conv_out = self.get_last_conv_output()[:, :, i]
            conv_out = conv_out > 0.5
            conv_out = conv_out.astype('float32')
            final_resize = resize(conv_out, self.up_size, order=1, mode='reflect',
                                  anti_aliasing=False)
            self.masks[:, :, i] = final_resize

    def get_feature_activation_masks(self):
        mask_ind = self.masks[:, :, 500]
        grid_ind = self.grid[500, :, :]
        new_mask = np.reshape(mask_ind, (self.input_shape[0], self.input_shape[1]))
        new_masks = np.rollaxis(self.masks, 2, 0)
        size = new_masks.shape
        self.data = new_masks.reshape(size[0], size[1], size[2], 1)
        masked = self.transformed_input * self.data
        self.N = len(new_masks)

    def attribute(self, p1=0.5):
        preds = []
        masked = self.transformed_input * self.masks
        pred_org = self.model.predict(self.transformed_input)

        for i in range(0, self.N, batch_size):
            preds.append(self.model.predict(masked[i:min(i + batch_size, self.N)]))
        preds = np.concatenate(preds)

        weights, diff = sidu_sim_differences(pred_org, preds)
        interactions = sidu_uniqness_measure(preds)
        new_interactions = interactions.reshape(-1, 1)
        diff_interactions = np.multiply(weights, new_interactions)

        sal = diff_interactions.T.dot(self.masks.reshape(self.N, -1)).reshape(-1, *self.input_shape)
        sal = sal / self.N / p1
        return sal, weights, new_interactions, diff_interactions, pred_org

    def figure_this_out(self):
        pred_vec = self.model.predict(self.transformed_input)
        print(np.array(pred_vec).shape)

    def run_pipeline(self):
        self.get_model()
        self.get_features_model()
        self.process_input_image()
        self.get_last_conv_output()
        self.generate_masks_conv_output()
        self.get_feature_activation_masks()
        self.attribute()
        self.figure_this_out()

if __name__ == "__main__":

    ids = [1]
    imagenet_dataloader = ImagenetDataloader()
    #cifar10_dataloader = Cifar10Dataloader()

    for idx, img in enumerate(imagenet_dataloader.test_loader()):
        # for idx, data in enumerate(cifar10_dataloader.train_loader()):
        # img, labels = data
        if ids:
            if idx in ids:
                sidu = Sidu(img=np.transpose(img, (0, 2, 3, 1)), model_type='inception_v3')
                sidu.run_pipeline()
                ids.remove(idx)
        else:
            break