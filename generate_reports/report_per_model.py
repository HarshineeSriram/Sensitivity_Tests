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
from tqdm import tqdm


class GenerateReports:

    def __init__(self, dataset="imagenet", folder="validation"):

        self.dataset = dataset
        self.imagenet_dataloader = ImagenetDataloader()
        self.folder = folder
        self.get_dataloader_items(folder=self.folder)

        self.model_name = []
        self.model_variation = []
        self.images = []
        self.layer_weights_distance = []

        self.deeplift_diff = []
        self.gradients_diff = []
        # self.integrated_gradients_diff = []
        # self.guided_gradcam_diff = []
        self.guided_backprop_diff = []
        self.inputxgrad_diff = []
        self.lime_diff = []

        self.run_pipeline(dataloader=self.dataloader)

    def get_dataloader_items(self, folder):
        if folder == "train":
            self.dataloader = self.imagenet_dataloader.train_loader()
        elif folder == "validation":
            self.dataloader = self.imagenet_dataloader.validation_loader()
        elif folder == "test":
            self.dataloader = self.imagenet_dataloader.test_loader()

    def run_pipeline(self, dataloader):

        idx = 0
        for img, _ in tqdm(dataloader, desc="Processing images in {} folder".format(self.folder)):
            if idx <= 3:
                # for key in tqdm(models.keys(), desc="Processing this image for model variations."):
                self.store_variations(key="vgg_16", img=img)
                idx += 1

        df = pd.DataFrame(list(zip(
            self.model_name, self.model_variation,
            self.images, self.layer_weights_distance,
            self.deeplift_diff, self.gradients_diff,
            # self.integrated_gradients_diff, self.guided_gradcam_diff,
            self.guided_backprop_diff, self.inputxgrad_diff,
            self.lime_diff)),
            columns=['model_name', 'model_variation',
                     'image', 'layer_weights_distance',
                     'deeplift_diff', 'gradients_diff',
                     # 'integrated_gradients_diff', 'guided_gradcam_diff',
                     'guided_backprop_diff', 'inputxgrad_diff',
                     'lime_diff']
        )
        df.to_csv(path=os.path.join(r'E:\sensitivity_tests\generate_reports', "{}_{}_{}".format(self.current_key,
                                                                                                self.dataset,
                                                                                                self.folder)))

    def store_variations(self, key="inception_v3", img=None):

        self.current_key = key
        self.current_img = img

        self.base_model = models[self.current_key](init_weights=True)
        self.base_model.load_state_dict(torch.load(model_paths_imagenet[self.current_key]))
        self.base_model.eval()
        with torch.no_grad():
            base_preds = self.base_model(self.current_img)
        base_pred_class = torch.argmax(base_preds)

        this_randomized_folder = os.path.join(os.path.dirname(model_paths_imagenet[self.current_key]),
                                              self.current_key +
                                              "_randomized")

        for randomized_layer in tqdm(os.listdir(this_randomized_folder),
                                     desc="Processing each model variation for this architecture. . ."):
            this_randomized_layer = randomized_layer.split("_")[1:]
            this_model = models[self.current_key](init_weights=True)
            this_model.load_state_dict(torch.load(os.path.join(this_randomized_folder, randomized_layer, 'model.pth')))

            self.store_attributions(
                model=this_model,
                last_layer=get_last_layer(model_type=self.current_key, model=this_model),
                target=base_pred_class,
                randomized_layer=this_randomized_layer
            )

    def store_attributions(
            self,
            model=None,
            last_layer=None,
            target=None,
            randomized_layer=None):

        baseline = torch.zeros(size=self.current_img.shape)
        model.eval()

        print("Processing gradients...")
        # Computing Gradient Values
        base_gradient = Gradient(forward_func=self.base_model)
        base_grad_attributions = base_gradient.attribute(self.current_img, target=target)
        this_gradient = Gradient(forward_func=model)
        this_grad_attributions = this_gradient.attribute(self.current_img, target=target)
        '''
        print("Processing integrated gradients...")
        # Computing Integrated Gradients Values
        base_ig = IntegratedGradients(forward_func=self.base_model)
        base_ig_attributions = base_ig.attribute(
            self.current_img, baseline,
            target=target, return_convergence_delta=False)
        this_ig = IntegratedGradients(forward_func=model)
        this_ig_attributions = this_ig.attribute(
            self.current_img, baseline,
            target=target, return_convergence_delta=False
        )
        print("Grad ssim loss {}".format(ssim_loss(img1=base_ig_attributions, img2=this_ig_attributions)))
        
        print("Processing guided gradcam...")
        # Computing Guided GradCAM Values
        base_guided_gradcam = GuidedGradCam(self.base_model, last_layer)
        base_guided_gradcam_attributions = base_guided_gradcam.attribute(self.current_img, target)
        this_guided_gradcam = GuidedGradCam(model, last_layer)
        this_guided_gradcam_attributions = this_guided_gradcam.attribute(self.current_img, target)
        print("Grad ssim loss {}".format(ssim_loss(img1=base_guided_gradcam_attributions, img2=this_guided_gradcam_attributions)))
        '''
        print("Processing guided backprop...")
        # Computing Guided BackProp Values
        base_guided_backprop = GuidedBackprop(self.base_model)
        base_guided_backprop_attributions = base_guided_backprop.attribute(self.current_img, target)
        this_guided_backprop = GuidedBackprop(model)
        this_guided_backprop_attributions = this_guided_backprop.attribute(self.current_img, target)

        print("Processing input x grad...")
        # Computing Input X Gradient Values
        base_inputxgrad = InputXGradient(self.base_model)
        base_inputxgrad_attributions = base_inputxgrad.attribute(self.current_img, target)
        this_inputxgrad = InputXGradient(model)
        this_inputxgrad_attributions = this_inputxgrad.attribute(self.current_img, target)

        print("Processing deeplift...")
        # Computing DeepLIFT Values
        base_deeplift = DeepLift(self.base_model)
        base_deeplift_attributions = base_deeplift.attribute(self.current_img, target=target)
        this_deeplift = DeepLift(model)
        this_deeplift_attributions = this_deeplift.attribute(self.current_img, target=target)

        print("Processing lime...")
        # Computing LIME Values
        superpixels, num_pixels = featuremask(img=self.current_img)

        base_lime = Lime(self.base_model)
        base_lime_attributions = base_lime.attribute(self.current_img,
                                                     target,
                                                     feature_mask=superpixels)
        this_lime = Lime(model)
        this_lime_attributions = this_lime.attribute(self.current_img,
                                                     target,
                                                     feature_mask=superpixels)

        # Saving all values

        ssim = SSIMLoss(get_device())
        self.model_name.append(self.current_key)
        self.model_variation.append(randomized_layer)
        self.images.append(self.current_img)
        """
        self.layer_weights_distance.append(layer_weights_diff(tensor1=self.base_model.state_dict()[randomized_layer[0]],
                                                              tensor2=model.state_dict()[randomized_layer[0]]))

        self.gradients_diff.append(ssim._compute(img1=base_grad_attributions, img2=this_grad_attributions))
        self.integrated_gradients_diff.append(ssim._compute(img1=base_ig_attributions, img2=this_ig_attributions))
        self.guided_gradcam_diff.append(
            ssim._compute(img1=base_guided_gradcam_attributions, img2=this_guided_gradcam_attributions))
        self.guided_backprop_diff.append(
            ssim._compute(img1=base_guided_backprop_attributions, img2=this_guided_backprop_attributions))
        self.inputxgrad_diff.append(ssim._compute(img1=base_inputxgrad_attributions, img2=this_inputxgrad_attributions))
        self.deeplift_diff.append(ssim._compute(img1=base_deeplift_attributions, img2=this_deeplift_attributions))
        self.lime_diff.append(ssim._compute(img1=base_lime_attributions, img2=this_lime_attributions))
        
        """
        self.gradients_diff.append(ssim_loss(img1=base_grad_attributions, img2=this_grad_attributions))
        # self.integrated_gradients_diff.append(ssim_loss(img1=base_ig_attributions, img2=this_ig_attributions))
        # self.guided_gradcam_diff.append(
        #    ssim_loss(img1=base_guided_gradcam_attributions, img2=this_guided_gradcam_attributions))
        self.guided_backprop_diff.append(
            ssim_loss(img1=base_guided_backprop_attributions, img2=this_guided_backprop_attributions))
        self.inputxgrad_diff.append(ssim_loss(img1=base_inputxgrad_attributions, img2=this_inputxgrad_attributions))
        self.deeplift_diff.append(ssim_loss(img1=base_deeplift_attributions, img2=this_deeplift_attributions))
        self.lime_diff.append(ssim_loss(img1=base_lime_attributions, img2=this_lime_attributions))


if __name__ == "__main__":
    gr = GenerateReports()
