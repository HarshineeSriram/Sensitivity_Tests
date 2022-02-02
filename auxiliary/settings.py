import torch
from models.inception_v3.inception_v3_imagenet.inceptionv3 import inception_v3
from models.mobilenet_v2.mobilenet_v2_imagenet.mobilenetv2 import mobilenet_v2
from models.vgg_16.vgg_16_imagenet.vgg16 import vgg16
from models.resnet_18.resnet_18_imagenet.resnet18 import resnet18
from models.resnet_50.resnet_50_imagenet.resnet_50 import resnet50
from models.resnet_152.resnet_152_imagenet.resnet_152 import resnet152
from models.lstm_soft_attention.lstm_softatt_imdb.lstm_soft_att import LSTMSoftAttention
from models.bilstm3_attention.bilstm3_attention_imdb.bilstm3_attention import BiLSTM3Attention
from models.bilstm6_attention.bilstm6_attention_imdb.bilstm6_attention import BiLSTM6Attention

# Global variables and functions
seed = 27


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Task: Image Classification

batch_size = 1
num_workers = 1

models = {
    # Following for image classification with ImageNet
    'inception_v3': inception_v3,
    'mobilenet_v2': mobilenet_v2,
    'vgg_16': vgg16,
    'resnet_18': resnet18,
    'resnet_50': resnet50,
    'resnet_152': resnet152,
    # Following for sentiment analysis with IMdB
    'lstm_softatt': LSTMSoftAttention,
    'bilstm3_att': BiLSTM3Attention,
    'bilstm6_att': BiLSTM6Attention
}
