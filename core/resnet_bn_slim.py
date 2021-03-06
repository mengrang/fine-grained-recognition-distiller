# coding:utf-8
import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from config import *
from dropblock import DropBlock2D, LinearScheduler


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

def L1_penalty(var):
    return torch.abs(var).sum()

def BatchNorm2d_no_b(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
    bn = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
    bn.bias.requires_grad = False
    return bn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d_no_b(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d_no_b(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d_no_b(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d_no_b(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_slimmed(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, block_channels, stride=1, downsample=None):
        super(Bottleneck_slimmed, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, block_channels[0], kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(block_channels[0])
        self.conv2 = nn.Conv2d(block_channels[0], block_channels[1], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(block_channels[1])
        self.conv3 = nn.Conv2d(block_channels[1], planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck_db(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck_db, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d_no_b(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d_no_b(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d_no_b(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=0.3, block_size=5),
            start_value=0.,
            stop_value=0.3,
            nr_steps=5e3
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(BN_W_INIT)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm2d_no_b(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
       
        x = self.maxpool(x)       
        x = self.layer1(x)
        x = self.dropblock(self.layer1(x))
        g1 = x
        x = self.layer2(x)
        x = self.dropblock(self.layer2(x))
        g2 = x       
        
        x = self.layer3(x)
        g3 = x
        x = self.layer4(x)
        g4 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = nn.Dropout(p=0.5)(x)
        x = self.fc(x)
        
        return x, (g1, g2, g3, g4)


class ResNet_slimmed(nn.Module):
    def __init__(self, block, layers, num_classes=1000, net_channels=None):
        self.inplanes = 64
        super(ResNet_slimmed, self).__init__()
        if net_channels == None:
            ## Standard Resnet Structure
            net_channels = [
                    [[64]], 
                    [[64, 64]]*layers[0], 
                    [[128, 128]]*layers[1], 
                    [[256, 256]]*layers[2], 
                    [[512, 512]]*layers[3]
                    ]
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = BatchNorm2d_no_b(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_0, self.layer1 = self._make_layer(block, 64, layers[0], net_channels[1])
        self.layer2_0, self.layer2 = self._make_layer(block, 128, layers[1], net_channels[2], stride=2)
        self.layer3_0, self.layer3 = self._make_layer(block, 256, layers[2], net_channels[3], stride=2)
        self.layer4_0, self.layer4 = self._make_layer(block, 512, layers[3], net_channels[4], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                # m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, layer_channels, stride=1):
        downsample = None
        
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, layer_channels[0], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, layer_channels[i]))

        return nn.Sequential(*layers[:2]), nn.Sequential(*layers[2:])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1_0(x)
        g1_0 = x
        x = self.layer1(x)        
        g1 = x
        x = self.layer2_0(x)        
        g2_0 = x
        x = self.layer2(x)
        g2 = x
        x = self.layer3_0(x)       
        g3_0 = x
        x = self.layer3(x)
        g3 = x
        x = self.layer4_0(x)  
        g4_0 = x
        x = self.layer4(x)      
        g4 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)        
        x = nn.Dropout(p=0.5)(x)
        x = self.fc(x)
        
        return x, (g1_0, g2_0, g3_0, g4_0), (g1, g2, g3, g4)

class ResNet_fps(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet_fps, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_0, self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2_0, self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3_0, self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4_0, self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=0.3, block_size=5),
            start_value=0.,
            stop_value=0.3,
            nr_steps=5e3
        )
        # weights_init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, math.sqrt(2. / m.out_features))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers[:2]), nn.Sequential(*layers[2:])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1_0(x)
        g1_0 = x
        x = self.layer1(x)
        x = self.dropblock(self.layer1(x))        
        g1 = x
        x = self.layer2_0(x)        
        g2_0 = x
        x = self.layer2(x)
        x = self.dropblock(self.layer2(x))
        g2 = x
        x = self.layer3_0(x)       
        g3_0 = x
        x = self.layer3(x)
        x = self.dropblock(self.layer3(x))
        g3 = x
        x = self.layer4_0(x)  
        g4_0 = x
        x = self.layer4(x)   
        x = self.dropblock(self.layer4(x))   
        g4 = x
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)        
        x = nn.Dropout(p=0.5)(x)
        x = self.fc(x)
        
        return x, (g1_0, g2_0, g3_0, g4_0), (g1, g2, g3, g4)


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        net_dict = slim_resume(model, model_zoo.load_url(model_urls['resnet50']))
        model.load_state_dict(net_dict)
    return model

def resnet50_fps(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_fps(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        net_dict = slim_resume(model, model_zoo.load_url(model_urls['resnet50']))
        model.load_state_dict(net_dict)
    return model

def resnet50_slimmed(net_channels, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_slimmed(Bottleneck_slimmed, [3, 4, 6, 3], net_channels=net_channels, **kwargs)
    # model.load_state_dict(net_dict)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

def slim_resume(model, ckpt_dict):
    net_dict = model.state_dict()    
    bn_b_dict = dict()
    for k, v in model.state_dict().items():
        for i in k.split('.'):
            if (i == 'bn1' or i == 'bn2') and k.endswith('bias'):
                bn_b_dict[k] = v
    pre_dict = {k: v for k, v in ckpt_dict.items() if k in net_dict and k not in bn_b_dict}        
    net_dict.update(pre_dict)
    return net_dict
