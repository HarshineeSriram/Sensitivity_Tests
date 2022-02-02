import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))

from collections import defaultdict

import math
from auxiliary.settings import seed, models
from auxiliary.paths import model_paths_imagenet
from utils.metrics import ssim
from utils.utils import get_last_layer
from explanation_methods.core.gradient import Gradient
from explanation_methods.core.integrated_gradients import IntegratedGradients
from explanation_methods.core.guided_gradcam import GuidedGradCam
from explanation_methods.core.guided_backprop import GuidedBackprop
from explanation_methods.core.input_x_gradient import InputXGradient
from explanation_methods.core.deep_lift import DeepLift
from explanation_methods.core.lime import Lime, featuremask
from data.dataloaders.imagenet_dataloaders import ImagenetDataloader

import torch
from tqdm import tqdm
import pandas as pd


class SImage:

    def __init__(self, imgs, model_type, explanations, destination):
        self.model_type = model_type
        self.destination = destination
        self.base_model = models[self.model_type](init_weights=True)
        self.base_model.load_state_dict(torch.load(model_paths_imagenet[self.model_type]))
        self.base_model.eval()

        self.explanation_methods = {'gradients': self.gradients,
                                    'integrated_gradients': self.integrated_gradients,
                                    'guided_gradcam': self.guided_gradcam,
                                    'guided_backprop': self.guided_backprop,
                                    'inputxgradient': self.inputxgradient,
                                    'deeplift': self.deeplift,
                                    'lime': self.lime}

        self.explanation_ssim = defaultdict(list)
        self.layer_ssim = defaultdict(list)
        self.s_layer = defaultdict(list)

        print("\n\nGenerating S Layer and S Dataset scores. . . ")
        self.generate_sensitivity_scores(images=imgs, explanations=explanations)
        print("\n\n Process complete! Kindly find the generated reports in {}".format(self.destination))

    def generate_sensitivity_scores(self, images, explanations):

        for this_img in tqdm(images, desc="Processing each image. . ."):
            self.explanation_ssim['image'].append(this_img)
            predicted_class = self.get_predicted_class(image=this_img)

            randomized_dir = os.path.join(
                os.path.dirname(model_paths_imagenet[self.model_type]),
                self.model_type + "_randomized")

            for randomized_layer in tqdm(os.listdir(randomized_dir),
                                         desc="Processing each model variation for this image . .. "):

                randomized_path = os.path.join(randomized_dir, randomized_layer)
                randomlayer_model = models[self.model_type](init_weights=False)
                randomlayer_model.load_state_dict(torch.load(os.path.join(randomized_path, "model.pth")))
                randomlayer_model.eval()

                for e in explanations:
                    if e in self.explanation_methods.keys():
                        self.update_layer_data(model1=self.base_model,
                                               model2=randomlayer_model,
                                               random_layer=randomized_layer,
                                               explanation=e,
                                               image=this_img,
                                               predicted_class=predicted_class)

                        self.update_image_data(model1=self.base_model,
                                               model2=randomlayer_model,
                                               explanation=e,
                                               image=this_img,
                                               predicted_class=predicted_class)

            self.calc_s_image(explanations=explanations)

        self.calc_s_layer()

        df_s_layer = pd.DataFrame(self.s_layer)
        df_s_layer.to_csv(os.path.join(self.destination, self.model_type + "_s_layer.csv"),
                          columns=self.s_layer.keys())

        df_s_image = pd.DataFrame(self.explanation_ssim)
        df_s_image.to_csv(os.path.join(self.destination, self.model_type + "_s_image.csv"),
                          columns=self.explanation_ssim.keys())

    def update_layer_data(self, model1, model2, random_layer, explanation, image, predicted_class):
        attributions1 = self.explanation_methods[explanation](img=image,
                                                              predicted_class=predicted_class,
                                                              model=model1)

        attributions2 = self.explanation_methods[explanation](img=image,
                                                              predicted_class=predicted_class,
                                                              model=model2)

        this_ssim = ssim(img1=attributions1, img2=attributions2)
        self.layer_ssim[explanation + '_' + random_layer].append(this_ssim)

    def calc_s_layer(self):
        for explanation_layer_config in self.layer_ssim.keys():
            self.s_layer[explanation_layer_config].append(1.0 - (sum(self.layer_ssim[explanation_layer_config]) /
                                                                 len(self.layer_ssim[explanation_layer_config])))

    def update_image_data(self, model1, model2, explanation, image, predicted_class):
        attributions1 = self.explanation_methods[explanation](img=image,
                                                              predicted_class=predicted_class,
                                                              model=model1)

        attributions2 = self.explanation_methods[explanation](img=image,
                                                              predicted_class=predicted_class,
                                                              model=model2)

        this_ssim = ssim(img1=attributions1, img2=attributions2)
        self.explanation_ssim[explanation + '_ssim'].append(this_ssim)

    def calc_s_image(self, explanations):
        print(self.explanation_ssim)
        for explanation in explanations:
            this_ssim_list = []
            for ssim_score in self.explanation_ssim[explanation + "_ssim"]:
                if math.isnan(ssim_score):
                    ssim_score = 0
                if ssim_score < 0.95:
                    this_ssim_list.append(ssim_score)
            self.explanation_ssim[explanation + "_s_image"].append(1.0 - (sum(this_ssim_list) / len(this_ssim_list)))
            del self.explanation_ssim[explanation + "_ssim"]

    def get_predicted_class(self, image):
        predictions = self.base_model(image)
        predicted_class = torch.argmax(predictions).int()
        return predicted_class

    def guided_gradcam(self, img, predicted_class, model):
        last_layer = get_last_layer(model_type=self.model_type, model=model)
        guided_gc = GuidedGradCam(model, last_layer)
        attributions = guided_gc.attribute(img, predicted_class)
        return attributions

    @staticmethod
    def gradients(img, predicted_class, model):
        gradient = Gradient(forward_func=model)
        attributions = gradient.attribute(img, target=predicted_class)
        return attributions

    @staticmethod
    def integrated_gradients(img, predicted_class, model):
        baseline = torch.zeros(size=img.shape)
        integrated_grad = IntegratedGradients(forward_func=model)
        attributions = integrated_grad.attribute(img,
                                                 baseline,
                                                 target=predicted_class,
                                                 return_convergence_delta=False)
        return attributions

    @staticmethod
    def guided_backprop(img, predicted_class, model):
        guided_bp = GuidedBackprop(model)
        attributions = guided_bp.attribute(img, predicted_class)
        return attributions

    @staticmethod
    def inputxgradient(img, predicted_class, model):
        inpxgrad = InputXGradient(model)
        attributions = inpxgrad.attribute(img, predicted_class)
        return attributions

    @staticmethod
    def deeplift(img, predicted_class, model):
        dl = DeepLift(model)
        attributions = dl.attribute(img, target=predicted_class)
        return attributions

    @staticmethod
    def lime(img, predicted_class, model):
        lime = Lime(model)
        superpixels, num_pixels = featuremask(img=img)
        attributions = lime.attribute(img,
                                      predicted_class,
                                      feature_mask=superpixels)
        return attributions


if __name__ == "__main__":

    imagenet_dataloader = ImagenetDataloader()
    images_list = []
    for idx, (img, label) in enumerate(imagenet_dataloader.validation_loader()):
        if idx < 100:
            images_list.append(img)
        else:
            break

    obj = SImage(imgs=images_list,
                 model_type="resnet_152",
                 explanations=['deeplift',
                               'gradients',
                               'integrated_gradients',
                               'guided_gradcam',
                               'guided_backprop',
                               'inputxgradient',
                               'lime'],
                 destination=r'E:\sensitivity_tests\generate_reports\s_image')
