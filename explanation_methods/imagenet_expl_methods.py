import os
import sys
drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))
from auxiliary.settings import seed, models, model_paths
from utils.utils import imagenet_class_from_idx, original_img_transform, attribution_transform, save_explanation_viz
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

# ----------------------------------------
torch.manual_seed(seed)
np.random.seed(seed)


# ----------------------------------------

class ImagenetExplanation:

    def __init__(self, source_img, model, explanations, save_viz_path):
        self.input_img = source_img
        self.transformed_input_img = original_img_transform(img=self.input_img)
        self.model_type = model
        self.model = models[self.model_type](init_weights=True)
        self.model.load_state_dict(torch.load(model_paths[self.model_type]))
        self.model.eval()
        self.destination_path = save_viz_path
        self.last_layer = self.get_last_layer()
        self.explanation_methods = {'gradients': self.gradients,
                                    'integrated_gradients': self.integrated_gradients,
                                    'guided_gradcam': self.guided_gradcam,
                                    'guided_backprop': self.guided_backprop,
                                    'inputxgradient': self.inputxgradient,
                                    'deeplift': self.deeplift,
                                    'lime': self.lime}

        for this_explanation in explanations:
            self.get_predictions()
            self.function_to_execute(this_explanation=this_explanation)

    def get_last_layer(self):

        if self.model_type == 'inception_v3':
            return self.model.Mixed_7c
        elif self.model_type == 'mobilenet_v3':
            return list(self.model.children())[-3][-2]
        elif self.model_type == 'vgg_16':
            return list(self.model.children())[-3][-3]

    def get_predictions(self):
        self.predictions = self.model(self.input_img)
        self.predicted_class = torch.argmax(self.predictions).int()
        #self.predicted_class_confidence = self.predictions[0, self.predicted_class] * 100
        self.predicted_label = imagenet_class_from_idx(self.predicted_class)

    def function_to_execute(self, this_explanation):

        if this_explanation in self.explanation_methods.keys():
            self.explanation_methods[this_explanation]()
        else:
            print("Invalid explanation method entered - {}. \n "
                  "Available explanation methods are {}"
                  .format(this_explanation, self.explanation_methods.keys()))

    def save_this_explanation(self,
                              explanation_type="Gradients",
                              expl_attributions=None,
                              method="blended_heat_map",
                              sign="absolute_value",
                              show_colorbar=True,
                              use_pyplot=False
                              ):

        save_explanation_viz(original_image=self.transformed_input_img,
                             explanation=expl_attributions,
                             destination=self.destination_path,
                             file_name="{}_{}_{}.jpeg".format(self.predicted_label, explanation_type, self.model_type),
                             method=method,
                             sign=sign,
                             show_colorbar=show_colorbar,
                             title="Explanation Method = {}, Model = {}, \n Predicted Class = {}"
                             .format(explanation_type, self.model_type, self.predicted_label),
                             use_pyplot=use_pyplot)

    def gradients(self):
        self.gradient = Gradient(forward_func=self.model)
        self.grads = self.gradient.attribute(self.input_img, target=self.predicted_class)
        self.grads = attribution_transform(attribution=self.grads)
        self.save_this_explanation(expl_attributions=self.grads)

    def integrated_gradients(self):
        self.baseline = torch.zeros(size=self.input_img.shape)
        self.integrated_grad = IntegratedGradients(forward_func=self.model)
        self.attributions, self.delta = self.integrated_grad.attribute(self.input_img,
                                                                       self.baseline,
                                                                       target=self.predicted_class,
                                                                       return_convergence_delta=True)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="Integrated Gradients",
                                   expl_attributions=self.attributions,
                                   sign="all")

    def guided_gradcam(self):
        self.guided_gc = GuidedGradCam(self.model, self.last_layer)
        self.attributions = self.guided_gc.attribute(self.input_img, self.predicted_class)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="Guided GradCAM",
                                   expl_attributions=self.attributions)

    def guided_backprop(self):
        self.guided_bp = GuidedBackprop(self.model)
        self.attributions = self.guided_bp.attribute(self.input_img, self.predicted_class)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="Guided Back Propagation",
                                   expl_attributions=self.attributions,
                                   sign="all")

    def inputxgradient(self):
        self.inpxgrad = InputXGradient(self.model)
        self.attributions = self.inpxgrad.attribute(self.input_img, self.predicted_class)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="Input X Gradient",
                                   expl_attributions=self.attributions,
                                   sign="all")

    def deeplift(self):
        self.dl = DeepLift(self.model)
        self.attributions = self.dl.attribute(self.input_img, target=self.predicted_class)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="DeepLIFT",
                                   expl_attributions=self.attributions,
                                   sign="all")

    def lime(self):
        self.lime_exp = Lime(self.model)
        superpixels, num_pixels = featuremask(img=self.input_img)
        self.attributions = self.lime_exp.attribute(self.input_img,
                                                    self.predicted_class,
                                                    feature_mask=superpixels)
        self.attributions = attribution_transform(attribution=self.attributions)
        self.save_this_explanation(explanation_type="LIME",
                                   expl_attributions=self.attributions,
                                   sign="all")

if __name__ == '__main__':

    ids = [1]
    imagenet_dataloader = ImagenetDataloader()

    for idx, img in enumerate(imagenet_dataloader.test_loader()):
        if ids:
            if idx in ids:
                explanation = ImagenetExplanation(source_img=img,
                                                  model='vgg_16',
                                                  explanations=['deeplift',
                                                                'gradients',
                                                                'integrated_gradients',
                                                                'guided_gradcam',
                                                                'guided_backprop',
                                                                'inputxgradient',
                                                                'lime'
                                                                ],
                                                  save_viz_path=r'E:\sensitivity_tests\viz')
                ids.remove(idx)
        else:
            break
