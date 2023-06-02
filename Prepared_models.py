# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 05:57:09 2022

@author: arh
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.transforms as transforms
import torchvision.models as models

# use cuda if it is available
if torch.cuda.device_count() != 0:
    device = torch.device('cuda:0')
else :
    device = torch.device('cpu')
    
class CIFAR_10_net(nn.Module):
    def __init__(self):
        super(CIFAR_10_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1)
        
        x = torch.randn(3,32,32).view(-1,3,32,32)
        self._to_linear = None
        self.convs(x)
        
        self.fc1 = nn.Linear(self._to_linear, 256)
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256, 10)

        self.dropout = nn.Dropout(p=0.5)

    def convs(self, x):
        # max pooling over 2x2
        x,ind = F.relu(self.conv1(x)), (2, 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x,ind = F.relu(self.conv3(x)), (2, 2)
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        
        if self._to_linear is None:
            self._to_linear = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.reshape(-1, self._to_linear)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        logit = self.fc3(x)
        return logit

class BaseBlock(nn.Module):
    alpha = 1

    def __init__(self, input_channel, output_channel, t = 6, downsample = False):
        """
            t:  expansion factor, t*input_channel is channel of expansion layer
            alpha:  width multiplier, to get thinner models
            rho:    resolution multiplier, to get reduced representation
        """ 
        super(BaseBlock, self).__init__()
        self.stride = 2 if downsample else 1
        self.downsample = downsample
        self.shortcut = (not downsample) and (input_channel == output_channel) 

        # apply alpha
        input_channel = int(self.alpha * input_channel)
        output_channel = int(self.alpha * output_channel)
        
        # for main path:
        c  = t * input_channel
        # 1x1   point wise conv
        self.conv1 = nn.Conv2d(input_channel, c, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(c)
        # 3x3   depth wise conv
        self.conv2 = nn.Conv2d(c, c, kernel_size = 3, stride = self.stride, padding = 1, groups = c, bias = False)
        self.bn2 = nn.BatchNorm2d(c)
        # 1x1   point wise conv
        self.conv3 = nn.Conv2d(c, output_channel, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(output_channel)
        

    def forward(self, inputs):
        # main path
        x = F.relu6(self.bn1(self.conv1(inputs)), inplace = True)
        x = F.relu6(self.bn2(self.conv2(x)), inplace = True)
        x = self.bn3(self.conv3(x))

        # shortcut path
        x = x + inputs if self.shortcut else x

        return x

class MobileNetV2(nn.Module):
    def __init__(self, output_size, alpha = 1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False),
            BaseBlock(16, 24, downsample = False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample = False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample = True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample = False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample = True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample = False))

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

        # weights init
        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, inputs):
        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x
    
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.nirmalizing_input = transforms.Normalize(mean, std)

    def forward(self, input):
        return self.nirmalizing_input(input)
    
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.nirmalizing_input = transforms.Normalize(mean, std)

    def forward(self, input):
        return self.nirmalizing_input(input)

    # model for cifar 10 dataset
def pretrained_model(model_type, path_or_model_name=None):
    if model_type == 'convnet':
        net = CIFAR_10_net().to(device)
        net.load_state_dict(torch.load(path_or_model_name))
        normalize_layer= Normalize((0.5, 0.5, 0.5), (1, 1, 1)).to(device)
        
    # model for cifar 10 dataset
    elif model_type == 'mobilenetv2':
        net = nn.DataParallel(MobileNetV2(10, alpha = 1)).to(device)
        net.load_state_dict(torch.load(path_or_model_name))
        normalize_layer= Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)).to(device)

    # robust model for cifar 10 dataset
    elif model_type == 'robustnet':
        from robustbench.utils import load_model
        # Load a model from the model zoo
        net = load_model(model_name=path_or_model_name,
                           dataset='cifar10',
                           threat_model='Linf').to(device)
        normalize_layer= Normalize((0, 0, 0), (1, 1, 1)).to(device)

    # model for ImageNet dataset
    elif model_type == 'inceptionv3':
        net = models.inception_v3(pretrained=True).to(device)
        normalize_layer= Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(device)

    # model for ImageNet dataset
    elif model_type == 'vits16' :
        import timm
        net = timm.create_model('vit_small_patch16_224', pretrained=True).to(device) 
        normalize_layer= Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)).to(device)

        

    normalized_net = nn.Sequential(
        normalize_layer,
        net
        )
    return normalized_net
