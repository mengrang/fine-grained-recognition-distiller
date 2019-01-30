# coding:utf-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR

from config import *
from utils import *
from kd_utils import *
from core import teacher, student, dataset
import collections
import shutil
# import tqdm

def main():
    start_epoch = 0
    saver_dir = mk_save(save_dir, cfg_dir)
    logging = init_log(saver_dir)
    _print = logging.info
    _print('----save_dir:' + saver_dir)
       
    ################
    # read dataset #
    ################
    trainset, testset, trainloader, testloader = dataloader(data_dir, 0)
    
    ################                                         
    # define model #
    ################ 
    teacher_net = teacher.teacher()
    student_net = student.student_slim_fps()
    
    ##########                                      
    # resume #
    ##########

    if teacher_id:
        ckpt = torch.load(teacher_id)
        ckpt_dict = ckpt['net_state_dict']
        # fc1_list = ['pretrained_model.fc1.weight', 'pretrained_model.fc1.bias']
        # _ckpt_dict = {k:v for k, v in ckpt_dict.items()}
        t_net_dict = teacher_net.state_dict()
        for keys in zip(t_net_dict, ckpt_dict):
            # print(keys)
            t_net_dict[keys[0]] = ckpt_dict[keys[1]]
        # exit()
        teacher_net.load_state_dict(t_net_dict)
       

    if resume:
        ckpt = torch.load(resume)
        ckpt_dict = ckpt['net_state_dict']
        s_net_dict = student_net.state_dict()
        for keys in zip(s_net_dict, ckpt_dict):
            # print(keys)
            s_net_dict[keys[0]] = ckpt_dict[keys[1]]
        # exit()
        student_net.load_state_dict(s_net_dict)
        start_epoch = 75
        print('resume', start_epoch)

    
    #####################
    #    define Loss    #
    #####################
    criterion = nn.CrossEntropyLoss().cuda()
    kd = nn.KLDivLoss().cuda()
    L2_loss = nn.MSELoss().cuda()

    #####################
    # define optimizers #
    #####################
    t_parameters = list(teacher_net.parameters())
    s_parameters = list(student_net.parameters())

    #####################
    # num of parameters #
    #####################
    params_count(student_net, teacher_net)

    #############################
    # OPTIMIZER FOR BN SLIMMING #
    #############################
    base_params, base_bias_params, slim_params = params_extract(student_net)
    
    teacher_net = teacher_net.cuda()
    teacher_net = DataParallel(teacher_net)
    student_net = student_net.cuda()
    student_net = DataParallel(student_net)

    for epoch in range(start_epoch, 500): 
      
        """  
        ##########################  train the model  ###############################
        """
        total = 0
        _print('--' * 80)
        student_net.train()
        teacher_net.eval()
        for i, data in enumerate(trainloader):
            #########################
            # warm up Learning rate #
            #########################
            lr = warm_lr(i, epoch, trainloader)

            ########################
            # No bias weight decay #
            ########################
            s_optimizer = torch.optim.Adam([{'params':base_params},
                                            {'params':base_bias_params + slim_params, 'weight_decay':0}],
                                    lr=lr, weight_decay=WD) 

            img, label = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            s_optimizer.zero_grad()
            L1_norm = 0.

            s_logits, s_fms0, s_fms = student_net(img)

            ###################
            # teacher-student #
            ###################   
            if teacher_id:
                t_grams = []
                s_grams = []
                with torch.no_grad():
                    t_logits, t_fms0, t_fms = teacher_net(img)   
                ##at 
                
                train_kd_loss, train_at_loss = at_kd(t_logits, s_logits, t_fms, s_fms, kd)
                ##fsp
                
                for fms in zip(t_fms0, t_fms):             
                    t_grams.append(gramm(fms[0], fms[1]))
                for fms in zip(s_fms0, s_fms):
                    s_grams.append(gramm(fms[0], fms[1]))
                
                train_fsp_loss = fsp(t_grams, s_grams, L2_loss)
            s_loss = criterion(s_logits, label)

            #############
            # L1 penaty #
            #############
            L1_norm = sum([L1_penalty(m).cuda() for m in slim_params])
           
            total_loss = s_loss + ALPHA * train_kd_loss + \
                        BETA * train_at_loss + \
                        FSP_LAMBDA * train_fsp_loss + L1_LAMBDA_end * L1_norm
            total_loss.backward()
            s_optimizer.step()
            total += batch_size
            progress_bar(i, len(trainloader), 
                            s_loss.item(),
                            train_kd_loss.item(),
                            train_at_loss.item(),
                            train_fsp_loss.item(),
                            arg5=total_loss.item(),
                            arg6=L1_norm, arg7=lr, msg='train')
        
        """
        ##########################  evaluate net and save model  ###############################
        """
        #############################
        # evaluate net on train set #
        #############################
        train_loss = 0
        train_correct = 0
        total = 0
        student_net.eval()
        teacher_net.eval()
        for i, data in enumerate(trainloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                s_logits, s_fms0, s_fms = student_net(img)

                ###################
                # teacher-student #
                ################### 

                if teacher_id:
                    t_grams = []
                    s_grams = []
                    with torch.no_grad():
                        t_logits, t_fms0, t_fms = teacher_net(img)   
                    ##at 
                    
                    train_kd_loss, train_at_loss = at_kd(t_logits, s_logits, t_fms, s_fms, kd)
                    ##fsp
                    
                    for fms in zip(t_fms0, t_fms):             
                        t_grams.append(gramm(fms[0], fms[1]))
                    for fms in zip(s_fms0, s_fms):
                        s_grams.append(gramm(fms[0], fms[1]))
                    
                    train_fsp_loss = fsp(t_grams, s_grams, L2_loss)
            
                ##calculate loss
                train_loss = criterion(s_logits, label)
                # train_inter_loss = criterion(interm, label)

                ##calculate accuracy
                _, s_predict = torch.max(s_logits, 1)
                total += batch_size
                train_correct += torch.sum(s_predict.data == label.data)
                train_loss += train_loss.item() * batch_size
                train_kd_loss += train_kd_loss.item() * batch_size
                train_at_loss += train_at_loss.item() * batch_size
                train_fsp_loss += train_fsp_loss.item() * batch_size

                ##L1 penaty
                L1_norm = sum([L1_penalty(m) for m in slim_params])
                total_loss = train_loss + ALPHA * train_kd_loss + BETA * train_at_loss

                progress_bar(i, len(trainloader), 
                            train_loss / (i+1),
                            train_kd_loss / (i+1),
                            train_at_loss / (i+1),
                            train_fsp_loss / (i+1),
                            arg5=total_loss / (i+1), msg='eval train set')

        train_acc = float(train_correct) / total
        train_loss = train_loss / total
        train_kd_loss = train_kd_loss / total
        train_at_loss = train_at_loss / total
        train_fsp_loss = train_fsp_loss / total
        total_loss = total_loss / total

        _print(
            'epoch:{} - train_loss: {:.4f}, kd_loss: {:.6f}, at_loss: {:.6f}, fsp_loss: {:6f}, total_loss: {:.4f}, train acc: {:.4f}, lr:{:.6f}, L1:{:.4f}, total sample: {}'.format(
                epoch,
                train_loss,
                train_kd_loss,
                train_at_loss,
                train_fsp_loss,
                total_loss,
                train_acc,              
                lr,
                L1_norm,
                total))
       
        """
        ############################
        # evaluate net on test set #
        ############################
        """
 
        if epoch % TEST_FQ == 0:               
            test_loss = 0
            test_correct = 0
            total = 0
            for i, data in enumerate(testloader):
                with torch.no_grad():   
                    img, label = data[0].cuda(), data[1].cuda()
                    batch_size = img.size(0)                 
                    s_logits, s_fms0, s_fms = student_net(img)
                
                    # calculate loss
                    test_loss = criterion(s_logits, label)

                    # calculate accuracy
                    _, s_predict = torch.max(s_logits, 1)
                    total += batch_size
                    test_correct += torch.sum(s_predict.data == label.data)

                    test_loss += test_loss.item() * batch_size            

                    progress_bar(i, len(testloader), 
                            test_loss / (i+1), msg='eval test set')

            test_acc = float(test_correct) / total
            test_loss = test_loss / total


            _print('epoch:{} - test loss: {:.4f} and test acc: {:.4f} total sample: {}'.format(
                    epoch,
                    test_loss,
                    test_acc,
                    total))

            ##########################  save model  ###############################
        net_state_dict = student_net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_kd_loss': train_kd_loss,
            'train_at_loss': train_at_loss,
            'train_fsp_loss': train_fsp_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            
            'net_state_dict': net_state_dict},
            os.path.join(saver_dir, '%03d.ckpt' % epoch))  

    print('Training Done')




def warm_lr(i, epoch, dataloader):
    # warm up Learning rate
    lr = 0
    if epoch <= 5:
        lr = LR / (len(dataloader) * 5) * (i + len(dataloader) * epoch)
    elif epoch > 5 and epoch <= 10:
        lr = LR
    elif epoch > 10 and epoch <= 50:
        lr = LR /10
    elif epoch > 50 and epoch <= 90:
        lr = LR * 1e-2
    elif epoch > 90 and epoch <= 130:
        lr = LR * 1e-3
    elif epoch > 130 and epoch <= 170:
        lr = LR * 1e-3 - (LR*1e-3 - LR*1e-4) / 40.  * (epoch - 130.)
    else:
        lr =lr   
    return lr
def L1_lambda_schedule(i, epoch, dataloader):
    # decay
    L1_lamb = 0
    if epoch <= L1_lamb_eps:
        L1_lamb = 0
    else:
        L1_lamb = L1_LAMBDA_end
    return L1_lamb
    
if __name__ == '__main__':
  main()