#!usr/bin/env python
#-*- coding:utf-8 _*-
"""
@version: python3.7
@author: zdy
@time: 2019-11-3 22:01
"""
from __future__ import print_function

import os
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
import dataset
import numpy as np
from args import args
from build_net import make_model
from transform import get_transforms

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig, get_optimizer, save_checkpoint
from collections import OrderedDict
import codecs
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
best_acc = 0  # best test accuracy

def load_network(load_path, network, strict=True):
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)

def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc
    fp = codecs.open('./pred.txt','a')
    filelist = list(open(args.testroot))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        _, pred = outputs.data.topk(5, 1, True, True)
        pred = pred.t()[:2]
        pred = pred.cpu().numpy()
        # print(pred[0])
        # image_name = filelist[batch_idx].strip().split('/')[-1].split(',')[0]
        image_name = filelist[batch_idx].strip().split('/')[-1]
        # print(image_name)
        # print(pred[0][0])
        fp.write(image_name + ',' + str(pred[0][0]) + '\n')

        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(val_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data # 调整数据的均值和方差
    transform = get_transforms(input_size=args.image_size, test_size=args.image_size, backbone=None)

    print('==> Preparing dataset %s' % args.trainval)

    valset = dataset.TestDataset(root=args.trainval, transform=transform['val_test'], Label = True)
    val_loader = data.DataLoader(valset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

    model = make_model(args)
    device = torch.device(type='cuda', index=0)
    # model = torch.load(args.modelpath)
    # model.module.load_state_dict(checkpoint['state_dict'])
    load_network(args.test_modelpath, model)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()  #
    optimizer = get_optimizer(model, args)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5,
                                                           verbose=False)

    # Resume
    title = 'ImageNet-' + args.arch

    print('\nEvaluation only')
    test_loss, test_acc, _ = test(val_loader, model, criterion, start_epoch, use_cuda)
    print(' Test Loss:  %.8f, Test Acc:  %.2f' % (test_loss, test_acc))

if __name__ == '__main__':
    main()

