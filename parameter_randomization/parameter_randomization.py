import os
import sys
drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))
from auxiliary.settings import models, model_paths_imagenet, seed
import torch
from tqdm import tqdm

# ----------------------------------------
torch.manual_seed(seed)
# ----------------------------------------


class ParameterRandomization:

    def __init__(self, model_type="inception_v3", dataset="imagenet"):

        self.model_type = model_type
        self.base_model = models[self.model_type](init_weights=True)

        self.dataset = dataset
        if self.dataset == "imagenet":
            self.base_model.load_state_dict(torch.load(model_paths_imagenet[self.model_type]))

        self.destination_path = os.path.join(os.path.dirname(
            model_paths_imagenet[self.model_type]),
            self.model_type+"_randomized")
        os.makedirs(self.destination_path, exist_ok=True)

        self.store_original_parameters()
        self.save_randomized_models()

    def store_original_parameters(self):

        self.important_layers = {}

        for layer, param in self.base_model.named_parameters():
            if param.requires_grad and 'weight' in layer and 'bn' not in layer:
                self.important_layers[layer] = param

    def save_randomized_models(self):
        folder_idx = 1
        temp_dict = self.base_model.state_dict().copy()
        for key in tqdm(self.base_model.state_dict().keys(),
                        desc="Randomizing important layers of {} and saving these models".format(self.model_type)):
            if key in self.important_layers.keys():
                random_weights_layer = torch.randn(size=self.important_layers[key].size())
                temp_dict[key] = random_weights_layer
                this_destination = os.path.join(self.destination_path, str(folder_idx) + '_' + key)
                os.makedirs(this_destination, exist_ok=True)
                torch.save(temp_dict, os.path.join(this_destination, 'model.pth'))
                temp_dict[key] = self.important_layers[key]
                folder_idx += 1

        print("\n\n Process complete! Kindly find the models in the subfolders of {}".format(self.destination_path))

if __name__ == "__main__":

    param_random = ParameterRandomization(model_type="vgg_16")