import os
import sys
drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg16
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as preprocess_mobilenetv2
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_inceptionv3
from tensorflow.keras.models import Model
import numpy as np
from skimage.transform import resize
from auxiliary.settings import models_tf
from utils.utils import get_last_layer_outputs_tf


class Sidu:

    def __init__(self, model_type="mobilenet_v2", img=None):

        self.model_type = model_type
        self.input_shape = (299, 299) if model_type == "inception_v3" else (224, 224)
        self.input_img = img
        self.base_model = models_tf[self.model_type](weights="imagenet")
        self.features_model = Model(inputs=self.base_model.input,
                                    outputs=get_last_layer_outputs_tf(self.model_type, self.base_model))

        self.img, self.x = self.load_img(img=self.input_img)
        self.masks, self.grid, self.cell_size, self.up_size = self.generate_masks_conv_output()


    def load_img(self, img):
        """ to load the image for the pretrained model which are trained from imagenet """
        img = np.resize(img, (1, self.input_shape[0], self.input_shape[1], 3))
        if self.model_type == 'mobilenet_v2':
            x = preprocess_mobilenetv2(img)
        elif self.model_type == 'inception_v3':
            x = preprocess_inceptionv3(img)
        elif self.model_type == 'vgg_16':
            x = preprocess_vgg16(img)
        return np.squeeze(img), x

    def get_last_conv_output(self):
        feature_activation_maps = self.features_model.predict(self.x)
        return np.squeeze(feature_activation_maps)

    def generate_masks_conv_output(self, s=8):
        last_conv_output = self.get_last_conv_output()
        cell_size = np.ceil(np.array(self.input_shape) / s)
        up_size = self.input_shape
        grid = np.rollaxis(last_conv_output, 2, 0)
        N = len(grid)
        masks = np.empty((*self.input_shape, N))
        for i in range(N):
            conv_out = last_conv_output[:, :, i]
            conv_out = conv_out > 0.5
            conv_out = conv_out.astype('float32')
            final_resize = resize(conv_out, up_size, order=1, mode='reflect',
                                  anti_aliasing=False)
            masks[:, :, i] = final_resize
        return masks, grid, cell_size, up_size