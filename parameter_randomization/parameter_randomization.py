import os
import sys
drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))
from auxiliary.paths import model_paths_imagenet, model_paths_imdb
from auxiliary.settings import models, seed
from data.dataloaders import load_data
import torch
from tqdm import tqdm

# ----------------------------------------
torch.manual_seed(seed)
# ----------------------------------------


class ParameterRandomization:

    def __init__(self, model_type="inception_v3"):

        self.model_type = model_type

        if 'lstm' in self.model_type:
            self.dataset = "imdb"
            TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()
            batch_size = 32
            output_size = 2
            hidden_size = 256
            embedding_length = 300
            self.base_model = models[self.model_type](batch_size, output_size, hidden_size,
                                                      vocab_size, embedding_length, word_embeddings)
            self.base_model.load_state_dict(torch.load(model_paths_imdb[self.model_type]))
            self.destination_path = os.path.join(os.path.dirname(
                model_paths_imdb[self.model_type]),
                self.model_type + "_randomized")

        else:
            self.dataset = "imagenet"
            self.base_model = models[self.model_type](init_weights=False)
            self.base_model.load_state_dict(torch.load(model_paths_imagenet[self.model_type]))
            self.destination_path = os.path.join(os.path.dirname(
                model_paths_imagenet[self.model_type]),
                self.model_type + "_randomized")

        os.makedirs(self.destination_path, exist_ok=True)
        self.important_layers = {}

        self.store_original_parameters()
        self.save_randomized_models()

    def store_original_parameters(self):

        model_blocks = []

        if self.model_type == "inception_v3":
            model_blocks = ["conv2d_1a", "conv2d_2a", "conv2d_2b", "conv2d_3b",
                                "conv2d_4a", "mixed_5b", "mixed_5c", "mixed_5d",
                                "mixed_6a", "mixed_6b", "mixed_6c", "mixed_6d",
                                "mixed_6e", "mixed_7a", "mixed_7b", "mixed_7c",
                                "logits"]

        elif self.model_type == "mobilenet_v2":
            model_blocks = ["features." + str(i) for i in range(0, 19)]

        elif self.model_type == "vgg_16":
            model_blocks = ["features.0", "features.2", "features.5",
                            "features.7", "features.10", "features.12",
                            "features.14", "features.17", "features.19",
                            "features.21", "features.24", "features.26",
                            "features.28"]

        elif self.model_type == "resnet_18":
            model_blocks = []
            for first_idx in range(1, 5):
                for second_idx in range(0, 2):
                    model_blocks.append('layer'+str(first_idx)+'.'+str(second_idx))

        elif self.model_type == "resnet_50":
            model_blocks = []
            for first_idx in range(1, 5):
                if first_idx == 1:
                    for second_idx in range(0, 3):
                        model_blocks.append('layer' + str(first_idx) + '.' + str(second_idx))
                elif first_idx == 2:
                    for second_idx in range(0, 4):
                        model_blocks.append('layer' + str(first_idx) + '.' + str(second_idx))
                elif first_idx == 3:
                    for second_idx in range(0, 6):
                        model_blocks.append('layer' + str(first_idx) + '.' + str(second_idx))
                elif first_idx == 4:
                    for second_idx in range(0, 3):
                        model_blocks.append('layer' + str(first_idx) + '.' + str(second_idx))

        elif self.model_type == "resnet_152":
            model_blocks = []
            for first_idx in range(1, 5):
                if first_idx == 1:
                    for second_idx in range(0, 3):
                        model_blocks.append('layer' + str(first_idx) + '.' + str(second_idx))
                elif first_idx == 2:
                    for second_idx in range(0, 8):
                        model_blocks.append('layer' + str(first_idx) + '.' + str(second_idx))
                elif first_idx == 3:
                    for second_idx in range(0, 36):
                        model_blocks.append('layer' + str(first_idx) + '.' + str(second_idx))
                elif first_idx == 4:
                    for second_idx in range(0, 3):
                        model_blocks.append('layer' + str(first_idx) + '.' + str(second_idx))

        elif self.model_type == "lstm_softatt":
            model_blocks = ['l0']

        elif self.model_type == "bilstm3_att":
            model_blocks = ['l0', 'l1', 'l2']

        elif self.model_type == "bilstm6_att":
            model_blocks = ['l0', 'l1', 'l2', 'l3', 'l4', 'l5']

        for element in model_blocks:
            self.important_layers[element] = {}

        for layer, param in self.base_model.named_parameters():
            if param.requires_grad:
                this_layer = layer.lower()
                for layer_idx in range(len(model_blocks)):
                    if model_blocks[layer_idx] in this_layer:
                        self.important_layers[model_blocks[layer_idx]][layer] = param

    def save_randomized_models(self):
        folder_idx = 1
        temp_dict = self.base_model.state_dict().copy()

        for block_key in tqdm(self.important_layers.keys(), desc="Randomizing each block of {}".format(self.model_type)):

            for layer in self.important_layers[block_key].keys():
                random_weights_layer = torch.randn(size=self.important_layers[block_key][layer].size())
                temp_dict[layer] = random_weights_layer

            this_destination = os.path.join(self.destination_path, block_key)
            os.makedirs(this_destination, exist_ok=True)
            torch.save(temp_dict, os.path.join(this_destination, 'model.pth'))

            for layer in self.important_layers[block_key].keys():
                temp_dict[layer] = self.important_layers[block_key][layer]

            folder_idx += 1

        print("\n\n Process complete! Kindly find the models in the subfolders of {}".format(self.destination_path))


if __name__ == "__main__":

    param_random = ParameterRandomization(model_type="bilstm6_att")