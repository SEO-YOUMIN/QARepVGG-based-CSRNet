import sys
import os

import warnings

from model import CSRNet
from repvgg_csrnet import Repvgg_CSRNet

from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time
from collections import OrderedDict
#from torchsummary import summary

import wandb

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')
parser.add_argument('--wandb', type=str, default='CSRNet-QARepVGG')

parser.add_argument('--kd', action='store_true')
parser.add_argument('--online', action='store_true')
parser.add_argument('--kd-weight', type=float, default=1)
parser.add_argument('--gt-weight', type=float, default=1)

def main():
    
    global args,best_prec1
    
    best_prec1 = 1e6
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 4
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 30
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    NGPU = torch.cuda.device_count()
    print('Device:', device)
    print('Count of using GPUs:', NGPU)
    print('Current cuda device:', torch.cuda.current_device())
    print("task id: ", args.task)

    # wandb
    wandb.init(
            project = args.wandb
            )
    wandb.config.update(args)

    model_t = CSRNet()
    model_t = model_t.cuda()

    model = Repvgg_CSRNet(backbone_name='RepVGG-A0', backbone_file='./qarepvgg_pretrained_weight.pth', deploy=False, pretrained=True)
    model = model.cuda()

    wandb.watch(model)
    model = nn.DataParallel(model).to(device)
    if NGPU > 1:
        model = nn.DataParallel(model, device_ids=list(range(NGPU)))
    model.to(device)

    torch.cuda.manual_seed(args.seed)
    criterion = nn.MSELoss(size_average=False).cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    optimizer_t = torch.optim.SGD(model_t.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            # args.start_epoch = checkpoint['epoch']
            args.start_epoch = 0
            #best_prec1 = checkpoint['best_prec1']
            # model.load_state_dict(checkpoint['state_dict'])
            state_dict = checkpoint['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if 'module' in k:
                    k = k.replace('module.', '') 
                new_state_dict[k]=v
            model_t.load_state_dict(new_state_dict)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    if args.kd:
        print("check teacher model")
        prec1 = validate(val_list, model_t, criterion)

            
    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        if args.kd :
            distill(train_list, model, model_t, criterion, optimizer, epoch)
            train(train_list, model_t, criterion, optimizer_t, epoch)
        else:
            train(train_list, model, criterion, optimizer, epoch)

        prec1 = validate(val_list, model, criterion)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        
        print(' * best MAE {mae:.3f} '
              .format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)

def distill(train_list, model, model_t, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ]), 
            train=True, 
            seen=model.seen,
            batch_size=args.batch_size,
            num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    #if args.online :
        #model_t.train()
    #else :
    model_t.eval()
    end = time.time()

    # summary(model, (3,768,1024))
    
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        output = model(img)

        output_t = model_t(img)
        
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)
        ############################to check data shape#######################
        #print("image  : ",img.shape)
        #print("target : ",target.shape)
        #print("output : ",output.shape)
        #print("density sum : ", torch.sum(target))
        ######################################################################
        
       
#        print('*********************check************************')
#        print('output size:', output.size())
#        print('target size:', target.size())
#        print('*********************check************************')
        loss_distill = criterion(output, output_t)
        loss = criterion(output, target)
        loss = args.kd_weight * loss_distill + args.gt_weight * loss
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  

        #if args.online :
        #    loss_t = criterion(output_t, target)
        #    optimizer_t.zero_grad()
        #    loss_t.backward()
        #    optimizer_t.step()

        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            wandb.log({"epoch": epoch,
                        "train_loss":losses.avg})

def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                ]), 
            train=True, 
            seen=model.seen,
            batch_size=args.batch_size,
            num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()

    # summary(model, (3,768,1024))
    
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)
        
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        
        
        
        
        target = target.type(torch.FloatTensor).unsqueeze(1).cuda()
        target = Variable(target)
        ############################to check data shape#######################
        #print("image  : ",img.shape)
        #print("target : ",target.shape)
        #print("output : ",output.shape)
        #print("density sum : ", torch.sum(target))
        ######################################################################
        
       
        #print('*********************check************************')
        #print('output size:', output.size())
        #print('target size:', target.size())
        #print('*********************check************************')
        loss = criterion(output, target)
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
            wandb.log({"epoch": epoch,
                        "train_loss":losses.avg})
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)
#        print('***************check*********************')
#        print('img is ', img.size())
#        print('output is ', output.size())
#        print('target is ', target.size())
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
        
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))
    wandb.log({"mae": mae})

    return mae    
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        
