import torch
import torchvision.models as models
import torch.nn as nn


layer_names = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
               'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
               'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
               'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
               'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4', 'pool5']
content_layers = ['relu1_1', 'relu2_1', 'relu3_1']

class VGG(nn.Module):

    def __init__(self):
        super(VGG, self).__init__()
        features = models.vgg19(pretrained=True).features

        self.features = nn.Sequential()
        for i, module in enumerate(features):
            name = layer_names[i]
            self.features.add_module(name, module)

    def forward(self, x):
        batch_size = x.size(0)
        all_outputs, output = [], x
        for name, module in self.features.named_children():
            output = module(output)
            if name in content_layers: 
                all_outputs.append(output.view(batch_size, -1))
                if name == content_layers[-1]: break

        return all_outputs