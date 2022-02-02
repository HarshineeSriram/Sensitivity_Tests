path_to_imagenet_labels = r'E:\sensitivity_tests\data\datasets\imagenet\Annotations\imagenet-simple-labels.json'

model_paths_imagenet = {
    'inception_v3': r'E:/sensitivity_tests/models/inception_v3/inception_v3_imagenet/inceptionv3_baseline.pth',
    'mobilenet_v2': r'E:/sensitivity_tests/models/mobilenet_v2/mobilenet_v2_imagenet/mobilenetv2_baseline.pth',
    'vgg_16': r'E:/sensitivity_tests/models/vgg_16/vgg_16_imagenet/vgg16_baseline.pth',
    'resnet_18': r'E:/sensitivity_tests/models/resnet_18/resnet_18_imagenet/resnet18_baseline.pth',
    'resnet_50': r'E:/sensitivity_tests/models/resnet_50/resnet_50_imagenet/resnet50_baseline.pth',
    'resnet_152': r'E:/sensitivity_tests/models/resnet_152/resnet_152_imagenet/resnet152_baseline.pth',
}

model_paths_imdb = {
    'lstm_softatt': r'E:/sensitivity_tests/models/lstm_soft_attention/lstm_softatt_imdb/lstm_soft_att_baseline.pth',
    'bilstm3_att': r'E:/sensitivity_tests/models/bilstm3_attention/bilstm3_attention_imdb/bilstm3_attention_baseline.pth',
    'bilstm6_att': r'E:/sensitivity_tests/models/bilstm6_attention/bilstm6_attention_imdb/bilstm6_attention_baseline.pth'
}
