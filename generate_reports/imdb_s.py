import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
drive, path = os.path.splitdrive(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, os.path.join(drive, os.sep, *path.split(os.sep)[:2]))

from collections import defaultdict
from auxiliary.settings import seed, models
from auxiliary.paths import model_paths_imdb
from data.dataloaders import load_data
from scipy.spatial import distance

import torch
import pandas as pd
from tqdm import tqdm
import numpy

class SSentence:

    def __init__(self, data_iter, model_type, destination):
        self.model_type = model_type
        self.destination = destination

        self.TEXT, self.vocab_size, self.word_embeddings, self.train_iter, self.valid_iter, self.test_iter = load_data. \
            load_dataset()
        self.batch_size = 32
        self.output_size = 2
        self.hidden_size = 256
        self.embedding_length = 300

        self.base_model = models[self.model_type](self.batch_size, self.output_size,
                                                  self.hidden_size, self.vocab_size,
                                                  self.embedding_length, self.word_embeddings)

        self.base_model.load_state_dict(torch.load(model_paths_imdb[self.model_type]))
        self.base_model = self.base_model.cuda()
        self.base_model.eval()

        if data_iter == "train":
            self.data_iter = self.train_iter
        elif data_iter == "validation":
            self.data_iter = self.valid_iter
        elif data_iter == "test":
            self.data_iter = self.test_iter

        self.attention_jsd = defaultdict(list)

    def random_layer_processing(self):
        for idx, batch in tqdm(enumerate(self.data_iter), desc="Processing each sentence . . ."):
            text = batch.text[0]
            if text.size()[0] != 32:
                continue
            if torch.cuda.is_available():
                text = text.cuda()

            for i in range(len(text)):
                self.attention_jsd['sentence'].append(text[i].cpu().detach().numpy())

            _, att_weights_base = self.base_model(text)

            randomized_dir = os.path.join(
                os.path.dirname(model_paths_imdb[self.model_type]),
                self.model_type + "_randomized")

            for randomized_layer in os.listdir(randomized_dir):
                randomized_path = os.path.join(randomized_dir, randomized_layer)
                randomlayer_model = models[self.model_type](self.batch_size, self.output_size,
                                                            self.hidden_size, self.vocab_size,
                                                            self.embedding_length, self.word_embeddings)
                randomlayer_model.load_state_dict(torch.load(os.path.join(randomized_path, "model.pth")))
                randomlayer_model = randomlayer_model.cuda()
                randomlayer_model.eval()

                _, att_weights_randomized = randomlayer_model(text)

                self.store_jsd(weights1=att_weights_base,
                               weights2=att_weights_randomized,
                               layer=randomized_layer,
                               sentences=text)

        jsd_df = pd.DataFrame(self.attention_jsd)
        jsd_df.to_csv(os.path.join(self.destination, 'attention_jsd_{}.csv'.format(self.model_type)))

        print("\n\n All JSD scores have been saved and the generated file can be found in {}".format(self.destination))

    def store_jsd(self, weights1, weights2, layer, sentences):
        for i in range(len(sentences)):
            if 'bilstm' in self.model_type:
                weightsA = weights1[i][0]
                weightsB = weights2[i][0]
            else:
                weightsA = weights1[i]
                weightsB = weights2[i]
            self.attention_jsd[layer].append(self.calc_jsd(weightsA, weightsB))

    @staticmethod
    def calc_jsd(weights1, weights2):
        weights1 = weights1.cpu().detach().numpy()
        weights2 = weights2.cpu().detach().numpy()
        return distance.jensenshannon(weights1, weights2)


if __name__ == "__main__":

    s = SSentence(data_iter='validation', model_type='lstm_softatt', destination=r'E:\sensitivity_tests\generate_reports\s_sentence')
    s.random_layer_processing()
