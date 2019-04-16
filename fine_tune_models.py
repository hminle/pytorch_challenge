import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision.models as models
from collections import OrderedDict


class FineTuneModel(nn.Module):

    def __init__(self, model_arch, n_classes, is_pretrained=True, is_freezed=False):
        super(FineTuneModel, self).__init__()
        if model_arch.startswith('vgg'):
            self.model = models.__dict__[model_arch](pretrained=is_pretrained)
            self.__set_requires_grad(self.model, value=is_freezed)

            n_inputs = self.model.classifier[0].in_features

            classifier = nn.Sequential(OrderedDict([
                ('fc1', nn.Linear(n_inputs, 4096)),
                ('relu', nn.ReLU()),
                ('fc2', nn.Linear(4096, n_classes))
            ]))
            self.model.classifier = classifier
        elif model_arch.startswith('resnet'):
            if model_arch == 'resnet34':
                self.model = models.__dict__[
                    model_arch](pretrained=is_pretrained)
                self.__set_requires_grad(self.model, value=is_freezed)
                n_inputs = self.model.fc.in_features
                self.model.fc = nn.Linear(n_inputs, n_classes)

        else:
            print('Not valid model')
            exit()

    def __set_requires_grad(self, model, value=False):
        for param in model.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x
