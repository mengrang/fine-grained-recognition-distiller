# coding: utf-8
from __future__ import print_function
import os
import sys
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from config import *
from core import teacher, student, dataset
import collections


# ATTENTION TRANSFER
def at(x):   
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))

def at_loss(x, y):
    if not at(x).size() == at(y).size():
        print(at(y).size(), at(y).size())
    return ((at(x) - at(y)).pow(2)).mean()

def at_kd(t_logits, s_logits, t_fms, s_fms, kd):
    # with torch.no_grad():
    #     t_logits, t_fms = teacher_net(img)   
    kd_loss = kd(F.log_softmax(s_logits, dim=1), 
                    F.softmax(t_logits / T, dim=1))  
    at_losses = [at_loss(x, y) for x, y in zip(s_fms, t_fms)] 
    loss_at = [v.sum() for v in at_losses]
    loss_at = sum(loss_at)   
    return kd_loss, loss_at

#########
#  FSP  #
#########
def flat(x):
    n, c, h, w = x.size()
    x = x.view(n, c, h * w)
    return x

def gramm(x, y):   
        x = flat(x)
        hw = x.size(-1)
        y = flat(y)
        x = x.permute(0, 1, 2)
        y = y.permute(0, 2, 1)
        g = torch.bmm(x, y) / hw
        return g

def fsp(t_grams, s_grams, loss_fn):
    losses = []
    for g in zip(s_grams, t_grams):
        losses.append(loss_fn(g[0], g[1]))
    loss = sum(losses) / len(losses)
    return loss







