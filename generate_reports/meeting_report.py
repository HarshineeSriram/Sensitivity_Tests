import os
import sys

drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))
from utils.metrics import SSIMLoss, ssim_loss, layer_weights_diff
from utils.utils import get_last_layer
from auxiliary.settings import models, model_paths_imagenet, get_device
from data.dataloaders.imagenet_dataloaders import ImagenetDataloader

from explanation_methods.core.gradient import Gradient
from explanation_methods.core.integrated_gradients import IntegratedGradients
from explanation_methods.core.guided_gradcam import GuidedGradCam
from explanation_methods.core.guided_backprop import GuidedBackprop
from explanation_methods.core.input_x_gradient import InputXGradient
from explanation_methods.core.deep_lift import DeepLift
from explanation_methods.core.lime import Lime, featuremask

import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


class GenerateReports:

    def __init__(self,
                 model_type="vgg_16",
                 dataset="imagenet",
                 folder="validation",
                 destination=r'E:\sensitivity_tests\generate_reports'):
        self.model_type = model_type
        self.dataset = dataset
        self.imagenet_dataloader = ImagenetDataloader()
        self.folder = folder
        self.destination = destination
        self.reset_values()

        self.dataloader = self.get_dataloader_items(folder=self.folder)
        self.run_pipeline()

    def reset_values(self):
        self.images = []
        self.layer_weights_distance = []
        self.deeplift_diff = []
        self.gradients_diff = []
        self.guided_backprop_diff = []
        self.inputxgrad_diff = []
        self.lime_diff = []

    def get_dataloader_items(self, folder):
        if folder == "train":
            return self.imagenet_dataloader.train_loader()
        elif folder == "validation":
            return self.imagenet_dataloader.validation_loader()
        elif folder == "test":
            return self.imagenet_dataloader.test_loader()

    def run_pipeline(self):

        # Load the base model
        self.base_model = models[self.model_type](init_weights=True)
        self.base_model.load_state_dict(torch.load(model_paths_imagenet[self.model_type]))
        self.base_model.eval()
        self.last_layer = get_last_layer(model_type=self.model_type, model=self.base_model)

        # Iteratively load all model variations
        this_randomized_folder = os.path.join(os.path.dirname(model_paths_imagenet[self.model_type]),
                                              self.model_type +
                                              "_randomized")
        for randomized_layer in tqdm(os.listdir(this_randomized_folder),
                                     desc="Processing each model variation for {}.".format(self.model_type)):
            this_destination_idx = randomized_layer.split("_")[0][0]
            this_randomized_layer = randomized_layer.split("_")[1:][0]
            this_model = models[self.model_type](init_weights=True)
            this_model.load_state_dict(
                torch.load(os.path.join(this_randomized_folder, randomized_layer, 'model.pth')))
            this_model.eval()

            # For each model variation, load the data folder
            idx = 0
            for img, _ in tqdm(self.dataloader, desc="Processing images in {} folder".format(self.folder)):
                # Get the base prediction to determine the target class for explanations
                print("\n\n\n\nThis img size is {}".format(np.array(img).shape))
                if idx < 1000:
                    with torch.no_grad():
                        base_preds = self.base_model(img)
                    base_pred_class = torch.argmax(base_preds)

                    self.store_attributions(img=img,
                                            model=this_model,
                                            randomized_layer=this_randomized_layer,
                                            target=base_pred_class)

                    idx += 1
                else:
                    break

            self.save_csv(randomized_layer=this_randomized_layer, this_destination_idx=this_destination_idx)

    def store_attributions(self, img, model, randomized_layer, target):

        print("Processing grad...")
        # Computing Gradient Values
        base_gradient = Gradient(forward_func=self.base_model)
        base_grad_attributions = base_gradient.attribute(img, target=target)
        this_gradient = Gradient(forward_func=model)
        this_grad_attributions = this_gradient.attribute(img, target=target)

        print("Processing guided backprop...")
        # Computing Guided BackProp Values
        base_guided_backprop = GuidedBackprop(self.base_model)
        base_guided_backprop_attributions = base_guided_backprop.attribute(img, target)
        this_guided_backprop = GuidedBackprop(model)
        this_guided_backprop_attributions = this_guided_backprop.attribute(img, target)

        print("Processing input x grad...")
        # Computing Input X Gradient Values
        base_inputxgrad = InputXGradient(self.base_model)
        base_inputxgrad_attributions = base_inputxgrad.attribute(img, target)
        this_inputxgrad = InputXGradient(model)
        this_inputxgrad_attributions = this_inputxgrad.attribute(img, target)

        print("Processing deeplift...")
        # Computing DeepLIFT Values
        base_deeplift = DeepLift(self.base_model)
        base_deeplift_attributions = base_deeplift.attribute(img, target=target)
        this_deeplift = DeepLift(model)
        this_deeplift_attributions = this_deeplift.attribute(img, target=target)

        print("Processing lime...")
        # Computing LIME Values
        superpixels, num_pixels = featuremask(img=img)

        base_lime = Lime(self.base_model)
        base_lime_attributions = base_lime.attribute(img,
                                                     target,
                                                     feature_mask=superpixels)
        this_lime = Lime(model)
        this_lime_attributions = this_lime.attribute(img,
                                                     target,
                                                     feature_mask=superpixels)

        self.images.append(img)
        self.layer_weights_distance.append(layer_weights_diff(tensor1=self.base_model.state_dict()[randomized_layer],
                                                              tensor2=model.state_dict()[randomized_layer]))
        self.gradients_diff.append(ssim_loss(img1=base_grad_attributions, img2=this_grad_attributions))
        self.guided_backprop_diff.append(
            ssim_loss(img1=base_guided_backprop_attributions, img2=this_guided_backprop_attributions))
        self.inputxgrad_diff.append(ssim_loss(img1=base_inputxgrad_attributions, img2=this_inputxgrad_attributions))
        self.deeplift_diff.append(ssim_loss(img1=base_deeplift_attributions, img2=this_deeplift_attributions))
        self.lime_diff.append(ssim_loss(img1=base_lime_attributions, img2=this_lime_attributions))

    def save_csv(self, randomized_layer, this_destination_idx):
        this_save_destination = os.path.join(self.destination,
                                             self.dataset,
                                             self.model_type,
                                             str(this_destination_idx)+'_'+randomized_layer)
        os.makedirs(this_save_destination, exist_ok=True)

        df = pd.DataFrame(list(zip(self.images, self.layer_weights_distance,
                                   self.deeplift_diff, self.gradients_diff,
                                   self.guided_backprop_diff, self.inputxgrad_diff,
                                   self.lime_diff)),
                          columns=['image', 'layer_weights_distance',
                                   'deeplift_diff', 'gradients_diff',
                                   'guided_backprop_diff', 'inputxgrad_diff',
                                   'lime_diff']
                          )
        df.to_csv(os.path.join(this_save_destination, "{}_{}_{}_{}.csv".format(self.model_type,
                                                                               randomized_layer,
                                                                               self.dataset,
                                                                               self.folder)))
        self.reset_values()


if __name__ == "__main__":
    rg = GenerateReports(model_type="vgg_16", folder="train")
    rg_2 = GenerateReports(model_type="mobilenet_v2", folder="train")
