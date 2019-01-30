# coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet, resnet_bn_slim
import numpy as np
from config import *
from collections import OrderedDict

class student(nn.Module):
    def __init__(self):
        super(student, self).__init__()
        self.pretrained_model = resnet.resnet50(pretrained=False)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)

    def forward(self, x):
        logits, fms0, fms = self.pretrained_model(x)    
        return logits, fms0, fms

class student_slim_fps(nn.Module):
    def __init__(self):
        super(student_slim_fps, self).__init__()
        self.pretrained_model = resnet_bn_slim.resnet50_fps(pretrained=False)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)

    def forward(self, x):
        logits, fms0, fms = self.pretrained_model(x)    
        return logits, fms0, fms

class student_fps_attention(nn.Module):
    def __init__(self):
        super(student_slim_fps, self).__init__()
        self.pretrained_model = resnet_bn_slim.resnet50_fps(pretrained=False)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)

    def forward(self, x):
        logits, fms0, fms = self.pretrained_model(x)    
        return logits, fms0, fms






class student_slimmed_fps(nn.Module):
    def __init__(self):
        super(student_slimmed_fps, self).__init__()
        self.real_prune = RealPrune(resume, RATIO, ResNet50_LAYERS)
        self.pretrained_model = resnet_bn_slim.resnet50_slimmed(self.real_prune.slim_channels())
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)

    def forward(self, x):
        logits, fms0, fms = self.pretrained_model(x)    
        return logits, fms0, fms

class RealPrune(object):
    def __init__(self, ckpt_dir, ratio, layers, model='ResNet50'):
        super(RealPrune, self).__init__()
        self.ckpt_dir = ckpt_dir
        self.ratio = ratio
        self.bn_w_dict = self.bn_weights()
        self.bnw_state = self.bn_state()
        self.layers = layers
        self.model = model
 
    def ckpt_dict(self):
        ckpt = torch.load(self.ckpt_dir)       
        return ckpt['net_state_dict']

    def bn_weights(self):
        w_dict = OrderedDict()
        for k, v in self.ckpt_dict().items():
            for i in k.split('.'):
                if (i == 'bn1' or i == 'bn2') and k.endswith('weight'):
                    w_dict[k] = v
        return w_dict

    def mask(self, var, thr):
        return list(var.cpu().abs().gt(thr).float().numpy())

    def bn_channels(self, var, thr):
        var = self.mask(var, thr)
        return sum(var)
        
    def threshold_adap(self, arr):
        arr.sort()
        num = arr.size
        elm = arr[int(num*(1-self.ratio))]
        return elm

    def bn_state(self):
        bnw_state = OrderedDict()
        bnw_state = {k:[v.abs().cpu().numpy().max(), 
                    v.abs().cpu().numpy().min(), 
                    v.abs().cpu().numpy().mean(), 
                    np.median(v.abs().cpu().numpy()),
                    self.threshold_adap(v.abs().cpu().numpy())] for k,v in self.bn_w_dict.items()}
        return bnw_state

    def slim_statistic(self):
        C_slim_ratio = []
        if self.model == 'ResNet50':

            flat_net_C = [c for layer in ResNet50_C for block in layer for c in block]
        flat_slim_C = [self.bn_channels(v, min(float(self.bnw_state[k][-1]), 0.05)) for k, v in self.bn_w_dict.items()]
        for p in zip(flat_slim_C, flat_net_C):
            C_slim_ratio.append(round(p[0]/p[1], 4))
        total_C_slim_ratio = sum(C_slim_ratio) / len(flat_net_C)

        return C_slim_ratio, total_C_slim_ratio

    def slim_channels(self):
        block_clust = []
        layer_clust = []
        if self.model == 'ResNet50':
            net_channels = [
                    [[64]], 
                    [[64, 64]]*self.layers[0], 
                    [[128, 128]]*self.layers[1], 
                    [[256, 256]]*self.layers[2], 
                    [[512, 512]]*self.layers[3]]           
        flat_channels = [int(self.bn_channels(v, min(float(self.bnw_state[k][-1]), 0.05))) for k, v in self.bn_w_dict.items()]
        net_channels[0][0] = [64]
        for i in range(1, len(flat_channels), 2):
            block_clust.append(flat_channels[i:i+2])
        layer_clust = [block_clust[:self.layers[0]]]
        for i in range(1, len(self.layers)): 
            layer_clust.append(block_clust[sum(self.layers[:i]):sum(self.layers[:i+1])])
        net_channels[1:] = layer_clust
        return net_channels

    def slim_bnws(self):
        bnw_index = OrderedDict()
        bnw_index = {k:np.argwhere(v.abs().cpu().numpy() > min(float(self.bnw_state[k][-1]), 0.05))
                    for k,v in self.bn_w_dict.items()}
        pruned_bn_w_dict = {k:v[bnw_index[k]].squeeze() for k,v in self.bn_w_dict.items()}
        
        slimmed_dict = self.ckpt_dict()
        for k, v in slimmed_dict.items():
            if k in pruned_bn_w_dict.keys():
                slimmed_dict[k] = pruned_bn_w_dict[k]
        
        return bnw_index, pruned_bn_w_dict, slimmed_dict