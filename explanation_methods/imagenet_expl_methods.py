import os
import sys

drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))
from auxiliary.settings import seed, models, model_paths_imagenet
from utils.utils import imagenet_class_from_idx, original_img_transform, \
    attribution_transform, save_explanation_viz, get_last_layer, \
    cifar10_class_from_idx, get_input_shape
import numpy as np
import torch
from explanation_methods.core.gradient import Gradient
from explanation_methods.core.integrated_gradients import IntegratedGradients
from explanation_methods.core.guided_gradcam import GuidedGradCam
from explanation_methods.core.guided_backprop import GuidedBackprop
from explanation_methods.core.input_x_gradient import InputXGradient
from explanation_methods.core.deep_lift import DeepLift
from explanation_methods.core.lime import Lime, featuremask
from data.dataloaders.imagenet_dataloaders import ImagenetDataloader
from data.dataloaders.cifar10_dataloaders import Cifar10Dataloader

import matplotlib.pyplot as plt
from tqdm import tqdm

# ----------------------------------------
torch.manual_seed(seed)
np.random.seed(seed)
# ----------------------------------------


class ImagenetExplanation:

    def __init__(self, source_img, model_type, explanations, save_viz_path, viz_type="baseline"):
        # self.dataset = dataset
        self.input_img = source_img
        self.transformed_input_img = original_img_transform(img=self.input_img)
        self.model_type = model_type
        self.model = models[self.model_type](init_weights=True)
        self.viz_type = viz_type

        if self.viz_type == "baseline":
            self.model.load_state_dict(torch.load(model_paths_imagenet[self.model_type]))
            self.model.eval()

        self.destination_path = save_viz_path
        self.last_layer = get_last_layer(model_type=self.model_type, model=self.model)
        self.explanation_methods = {'gradients': self.gradients,
                                    'integrated_gradients': self.integrated_gradients,
                                    'guided_gradcam': self.guided_gradcam,
                                    'guided_backprop': self.guided_backprop,
                                    'inputxgradient': self.inputxgradient,
                                    'deeplift': self.deeplift,
                                    'lime': self.lime}

        for this_explanation in tqdm(explanations, desc="Generating explanations"):
            self.function_to_execute(this_explanation=this_explanation)

    def get_predictions(self):
        self.predictions = self.model(self.input_img)
        self.predicted_class = torch.argmax(self.predictions).int()
        # self.predicted_class_confidence = self.predictions[0, self.predicted_class] * 100
        self.predicted_label = imagenet_class_from_idx(self.predicted_class)
        # else:
        #    self.predicted_label = cifar10_class_from_idx(self.predicted_class)

    def function_to_execute(self, this_explanation):

        if self.viz_type == "baseline":
            if this_explanation in self.explanation_methods.keys():
                self.get_predictions()
                self.explanation_methods[this_explanation](model=self.model)

        elif self.viz_type == "parameter_randomization":
            if this_explanation in self.explanation_methods.keys():
                this_dir = os.path.join(
                    os.path.dirname(model_paths_imagenet[self.model_type]),
                    self.model_type + "_randomized")

                for folder in os.listdir(this_dir):
                    self.model.load_state_dict(torch.load(os.path.join(this_dir, folder, 'model.pth')))
                    self.model.eval()
                    self.get_predictions()

                    this_destination = os.path.join(
                        self.destination_path,
                        self.model_type + "_randomized",
                        folder,
                        this_explanation
                    )
                    os.makedirs(this_destination, exist_ok=True)
                    self.explanation_methods[this_explanation](model=self.model,
                                                               destination=this_destination,
                                                               folder=folder)

        elif this_explanation not in self.explanation_methods.keys():
            print("Invalid explanation method entered - {}. \n "
                  "Available explanation methods are {}"
                  .format(this_explanation, self.explanation_methods.keys()))

    def save_this_explanation(self,
                              explanation_type="Gradients",
                              expl_attributions=None,
                              method="blended_heat_map",
                              sign="absolute_value",
                              show_colorbar=True,
                              use_pyplot=False,
                              destination=None,
                              folder=None
                              ):

        this_viz_destination = self.destination_path
        this_title = "Explanation Method = {}, " \
                     "Model = {}, \n " \
                     "Predicted Class = {}".format(explanation_type, self.model_type, self.predicted_label)

        if self.viz_type == "parameter_randomization":
            this_viz_destination = destination
            this_title = "Explanation Method = {}, " \
                         "Model = {}, \n Layer randomized = {} \n" \
                         "Predicted Class = {}".format(explanation_type, self.model_type, folder, self.predicted_label)

        save_explanation_viz(original_image=self.transformed_input_img,
                             explanation=expl_attributions,
                             destination=this_viz_destination,
                             file_name="{}_{}_{}.jpeg".format(self.predicted_label, explanation_type, self.model_type),
                             method=method,
                             sign=sign,
                             show_colorbar=show_colorbar,
                             title=this_title,
                             use_pyplot=use_pyplot)

    def gradients(self, model, **kwargs):
        self.gradient = Gradient(forward_func=model)
        self.grads = self.gradient.attribute(self.input_img, target=self.predicted_class)
        self.grads = attribution_transform(attribution=self.grads)
        self.save_this_explanation(expl_attributions=self.grads, **kwargs)

    def integrated_gradients(self, model, **kwargs):
        self.baseline = torch.zeros(size=self.input_img.shape)
        self.integrated_grad = IntegratedGradients(forward_func=model)
        self.attributions, self.delta = self.integrated_grad.attribute(self.input_img,
                                                                       self.baseline,
                                                                       target=self.predicted_class,
                                                                       return_convergence_delta=True)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="Integrated Gradients",
                                   expl_attributions=self.attributions,
                                   sign="all", **kwargs)

    def guided_gradcam(self, model, **kwargs):
        self.guided_gc = GuidedGradCam(model, self.last_layer)
        self.attributions = self.guided_gc.attribute(self.input_img, self.predicted_class)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="Guided GradCAM",
                                   expl_attributions=self.attributions,
                                   **kwargs)

    def guided_backprop(self, model, **kwargs):
        self.guided_bp = GuidedBackprop(model)
        self.attributions = self.guided_bp.attribute(self.input_img, self.predicted_class)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="Guided Back Propagation",
                                   expl_attributions=self.attributions,
                                   sign="all",
                                   **kwargs)

    def inputxgradient(self, model, **kwargs):
        self.inpxgrad = InputXGradient(model)
        self.attributions = self.inpxgrad.attribute(self.input_img, self.predicted_class)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="Input X Gradient",
                                   expl_attributions=self.attributions,
                                   sign="all",
                                   **kwargs)

    def deeplift(self, model, **kwargs):
        self.dl = DeepLift(model)
        self.attributions = self.dl.attribute(self.input_img, target=self.predicted_class)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="DeepLIFT",
                                   expl_attributions=self.attributions,
                                   sign="all",
                                   **kwargs)

    def lime(self, model, **kwargs):
        self.lime_exp = Lime(model)
        superpixels, num_pixels = featuremask(img=self.input_img)
        self.attributions = self.lime_exp.attribute(self.input_img,
                                                    self.predicted_class,
                                                    feature_mask=superpixels)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="LIME",
                                   expl_attributions=self.attributions,
                                   sign="all",
                                   **kwargs)


if __name__ == '__main__':

    ids = [1]
    idx = 1
    imagenet_dataloader = ImagenetDataloader()
    # cifar10_dataloader = Cifar10Dataloader()

    for img, label in imagenet_dataloader.external_test_loader():
        # for idx, data in enumerate(cifar10_dataloader.train_loader()):
        # img, labels = data
        if ids:
            if idx in ids:
                explanation = ImagenetExplanation(source_img=img,
                                                  model_type='vgg_16',
                                                  explanations=['deeplift',
                                                                'gradients',
                                                                'integrated_gradients',
                                                                'guided_gradcam',
                                                                'guided_backprop',
                                                                'inputxgradient',
                                                                'lime'
                                                                ],
                                                  save_viz_path=r'E:\sensitivity_tests\viz',
                                                  viz_type="parameter_randomization")
                ids.remove(idx)
        else:
            break
