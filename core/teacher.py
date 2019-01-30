# coding:utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from core import resnet
import numpy as np
from config import *

class teacher(nn.Module):
    def __init__(self):
        super(teacher, self).__init__()
        self.pretrained_model = resnet.resnet50(pretrained=False)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, 200)

    def forward(self, x):
        logits, fms0, fms = self.pretrained_model(x)    
        return logits, fms0, fms

