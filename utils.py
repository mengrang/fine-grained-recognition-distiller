# coding: utf-8
from __future__ import print_function
import os
import sys
import time
import logging
from functools import partial
import torch
import torch.nn.functional as F
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from config import *
from core import teacher, student, dataset
import collections
import shutil
# import tqdm


term_width = 200
TOTAL_BAR_LENGTH = 20.
last_time = time.time()
begin_time = last_time

#########################################
#               Functions               #
#########################################

def mk_save(dir, cfg_dir):
    make_dir = os.path.join(dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(make_dir):
        raise NameError('model dir exists!')
    os.makedirs(make_dir)
    
    cfg_file = os.path.join(cfg_dir, 'config.py')
    train_file = os.path.join(cfg_dir, 't_s_slim_v3.py')
    shutil.copy(cfg_file, make_dir)
    shutil.copy(train_file, make_dir)
    return make_dir

def dataloader(data_dir, num_workers):
    trainset = dataset.CUB(root=data_dir, is_train=True, data_len=None)
    testset = dataset.CUB(root=data_dir, is_train=False, data_len=None)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                            shuffle=True, num_workers=0, drop_last=False)    
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                            shuffle=False, num_workers=0, drop_last=False)
    return trainset, testset, trainloader, testloader

# def params_extract(model):
#     base_params = []
#     slim_params = []
#     bp_params = []
#     for name, param in model.named_parameters():
#         if name.endswith('weight'):
#             for j in name.split('.'):
#                 if (j == BN_NAMES[0] or j == BN_NAMES[1]):
#                     slim_params.append(param)
#         elif param.requires_grad:
#             base_params.append(param)
#     # base_params = filter(lambda p:p not in slim_params, bp_params)
#     return base_params, slim_params
# def params_extract(model):
#     base_params = []
#     base_names = []
#     base_bias_params = []
#     base_bias_names = []
#     slim_params = []
#     slim_names = []
    
#     for name, param in model.named_parameters():
#       if param.requires_grad:
#         if name.endswith('bias'):
#           base_bias_params.append(param)
#           base_bias_names.append(name)
#         elif name.endswith('weight'):
#           for j in name.split('.'):
#             if (j == 'bn1' or j == 'bn2'):
#               slim_params.append(param)
#               slim_names.append(name)
#     return slim_params
def params_extract(model):
    base_params = []
    base_names = []
    base_bias_params = []
    base_bias_names = []
    slim_params = []
    slim_names = []
    
    for name, param in model.named_parameters():
      if param.requires_grad:
        if name.endswith('bias'):
          base_bias_params.append(param)
          base_bias_names.append(name)
        elif name.endswith('weight'):
          for j in name.split('.'):
            if (j == 'bn1' or j == 'bn2'):
              slim_params.append(param)
              slim_names.append(name)

    for name, param in model.named_parameters():
      if param.requires_grad and name:
        if name not in base_bias_names and name not in slim_names:
          base_params.append(param)
    return base_params, base_bias_params, slim_params

def no_b_bn_resume(model, ckpt_dict):
    net_dict = model.state_dict()    
    bn_b_dict = dict()
    for k, v in model.state_dict().items():
        for i in k.split('.'):
            if (i == 'bn1' or i == 'bn2') and k.endswith('bias'):
                bn_b_dict[k] = v
    pre_dict = {k: v for k, v in ckpt_dict.items() if k in net_dict and k not in bn_b_dict}        
    net_dict.update(pre_dict)
    return net_dict


def params_count(student_net, teacher_net):
    s_n_parameters = sum(p.numel() for p in student_net.parameters())       
    t_n_parameters = sum(p.numel() for p in teacher_net.parameters())       
    print('-----Teacher Model Size: {:.5f}M'.format(t_n_parameters/1e6))
    print('-----Student Model Size: {:.5f}M'.format(s_n_parameters/1e6))
   
def L1_penalty(var):
    return torch.abs(var).sum()

def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)

def progress_bar(current, 
                total, 
                loss1, loss2=0, loss3=0, 
                arg4=None, arg5=None, arg6=None, arg7=None,
                msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    L.append(' | s_loss: {:.4f}'.format(loss1))
    L.append(' | kd_loss: {:.6f}'.format(loss2))
    L.append(' | at_loss: {:.6f}'.format(loss3))
    if arg4:
        L.append(' | fsp_loss: {:.4f}'.format(arg4))
    if arg5:
        L.append(' | total_loss: {:.4f}'.format(arg5))
    if arg6:
        L.append(' | L1:{:.6f}'.format(arg6))
    if arg7:
        L.append(' | lr:{:.6f}'.format(arg7))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2)):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def init_log(output_dir):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%Y%m%d-%H:%M:%S',
                        filename=os.path.join(output_dir, 'log.log'),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging
    
if __name__ == '__main__':
    pass
