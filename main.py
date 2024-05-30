import sys
import os
import time
import argparse
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.backends import cudnn
import torch.nn.functional as F
import datetime
import math

from model.resnext import resnet50, resnext50
from model.resnet_small import resnet32
from dataset.cifar100 import Cifar100
from dataset.cifar10 import Cifar10
from dataset.imagenet import ImageNet_LT
from utils.util import *
from Trainer import Trainer

best_acc1 = 0

def get_model(args):
    print("=> creating model '{}'".format(args.arch))
    if args.arch == 'resnet32':
        net = resnet32(num_classes=args.num_classes)
    elif args.arch == 'resnet50':
        net = resnet50(num_classes=args.num_classes)
    elif args.arch == 'resnext50':
        net = resnext50(num_classes=args.num_classes)
    return net


def get_dataset(args):
    transform_train, transform_val = get_transform(args)
    if args.dataset == 'cifar10':
        trainset = Cifar10(transform=ThreeCropTransform(transform_train), imbanlance_rate=args.imbanlance_rate, train=True)
        testset = Cifar10(transform=transform_val, imbanlance_rate=args.imbanlance_rate, train=False)
        print("load cifar10")
        return trainset, testset

    if args.dataset == 'cifar100':
        trainset = Cifar100(transform=ThreeCropTransform(transform_train), imbanlance_rate=args.imbanlance_rate, train=True)
        testset = Cifar100(transform=transform_val, imbanlance_rate=args.imbanlance_rate, train=False)
        real_trainset = Cifar100(transform=ThreeCropTransform(transform_train), imbanlance_rate=args.imbanlance_rate, train=True, real_only=True)
        print("load cifar100")
        return trainset, testset, real_trainset

    if args.dataset == 'ImageNet-LT':
        trainset = ImageNet_LT(transform=ThreeCropTransform(transform_train), train=True)
        testset = ImageNet_LT(transform=transform_val, train=False)
        print("load ImageNet-LT")
        return trainset, testset
    

def main():
    args = parser.parse_args()
    print(args)
    curr_time = datetime.datetime.now()
    args.store_name = '#'.join(["dataset: " + args.dataset, "arch: " + args.arch,"imbanlance_rate: " + str(args.imbanlance_rate)
            ,datetime.datetime.strftime(curr_time, '%Y-%m-%d %H:%M:%S')])
    prepare_folders(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True
    main_worker(args.gpu, args)

def main_worker(gpu, args):

    global best_acc1

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))
    # create model
    num_classes = args.num_classes
    model = get_model(args)
    _ = print_model_param_nums(model=model)
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.root_log, args.store_name, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger()
    logger.addHandler(fh)

    # Data loading code
    train_dataset, val_dataset, real_dataset = get_dataset(args)
    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == args.num_classes
    real_per_class_num = train_dataset.get_real_per_class_num()

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=True, pin_memory=True)
    mix_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, persistent_workers=True, pin_memory=True)
    real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, persistent_workers=True, pin_memory=True)

    start_time = time.time()
    print("Training started!")
    trainer = Trainer(args, model=model, train_loader=train_loader, mix_loader=mix_loader, real_loader=real_loader, val_loader=val_loader, real_per_class_num=real_per_class_num, log=logging)
    trainer.train()
    end_time = time.time()
    print("It took {} to execute the program".format(hms_string(end_time - start_time)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ResNets for CIFAR100')
    parser.add_argument('--dataset', type=str, default='cifar100', help="cifar10, cifar100, ImageNet-LT")
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet32', choices=('resnet32', 'resnet50', 'resnext50'))
    parser.add_argument('--imbanlance_rate', default=0.01, type=float, help='imbalance factor')
    parser.add_argument('--num_classes', default=100, type=int, help='number of classes ')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=128, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight_decay', default=5e-3, type=float, metavar='W',help='weight decay (default: 5e-3、2e-4、1e-4)', dest='weight_decay')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-features_dim', type=int, default=128, help='dim of contrastive features')
    parser.add_argument('--loss_strategy', default='pos_neg', choices=('pos_neg', 'neg_only', 'drop'))
    parser.add_argument('--randaug_m', default=10, type=int, help='randaug-m')
    parser.add_argument('--randaug_n', default=2, type=int, help='randaug-n')
    
    parser.add_argument('--seed', default=35180, type=int, help='seed for initializing training. ')
    parser.add_argument('-p', '--print_freq', default=100, type=int, metavar='N', help='print frequency (default: 100)')
    parser.add_argument('--gpu', default=None, type=int,help='GPU id to use.')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--root_log', type=str, default='./output/root_log')
    parser.add_argument('--root_model', type=str, default='./output/root_model')
    parser.add_argument('--store_name', type=str, default='./output')
    main()