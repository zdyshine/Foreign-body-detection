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

from utils import mkdir_p
from collections import OrderedDict

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


def load_network(load_path, network, strict=True):
    load_net = torch.load(load_path)
    load_net_clean = OrderedDict()
    for k, v in load_net.items():
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v
    network.load_state_dict(load_net_clean, strict=strict)

def main():
    global best_acc
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data # 调整数据的均值和方差
    transform = get_transforms(input_size=args.image_size, test_size=args.image_size, backbone=None)

    print('==> Preparing dataset %s' % args.testroot)
    filelist = list(open(args.testroot))

    testset = dataset.TestDataset(root=args.testroot, transform=transform['val_test'], Label = False)
    test_loader = data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers,
                                 pin_memory=True)

    model = make_model(args)
    device = torch.device(type='cuda', index=0)
    load_network(args.modelpath, model)
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)

    for batch_idx, inputs in enumerate(test_loader):
        # measure data loading time

        if use_cuda:
            inputs = inputs.cuda()
        inputs= torch.autograd.Variable(inputs)

        # compute output
        outputs = model(inputs)
        _, pred = outputs.data.topk(5, 1, True, True)
        pred = pred.t()[:2]
        pred = pred.cpu().numpy()
        print(pred)
        image_name = filelist[batch_idx].strip().split('/')[-1]
        # print('The pred of %s is %s' %(image_name, pred[0]))


if __name__ == '__main__':
    main()

