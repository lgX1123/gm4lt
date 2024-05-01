import sys
import os
import time
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
from torch.backends import cudnn
import torch.nn.functional as F
from utils import util
from utils.util import *
from loss.ContrastiveLoss import SCL
from loss.MixLoss import MixLoss
import datetime
import math
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import warnings


class Trainer(object):
    def __init__(self, args, model=None, train_loader=None, mix_loader=None, val_loader=None, real_per_class_num=[], log=None):
        self.args = args
        self.device = args.gpu
        self.print_freq = args.print_freq
        self.lr = args.lr
        self.epochs = args.epochs
        self.start_epoch = args.start_epoch
        self.use_cuda = True
        self.num_classes = args.num_classes
        self.real_per_class_num = np.array(real_per_class_num)
        self.log = log
        
        self.train_loader = train_loader
        self.mix_loader = mix_loader
        self.val_loader = val_loader
        self.model = model

        self.optimizer = torch.optim.SGD(self.model.parameters(), momentum=0.9, lr=self.lr, weight_decay=args.weight_decay)
        # self.train_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[60, 120, 160], gamma=0.2)
        self.train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.SCL = SCL().cuda(self.device)
        self.CE = nn.CrossEntropyLoss().cuda(self.device)
        self.ML = MixLoss().cuda(self.device)
        
    def train(self):
        best_acc1 = 0
        for epoch in range(self.start_epoch, self.epochs):
            print('current lr {:.5e}'.format(self.optimizer.param_groups[0]['lr']))
            self.train_per_epoch(epoch)
            self.train_scheduler.step()

            # evaluate on validation set
            acc1 = self.validate(epoch)

            # remember best prec@1 and save checkpoint
            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)
            output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
            print(output_best)
            save_checkpoint(self.args, {
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_acc1':  best_acc1,
            }, is_best, epoch + 1)
            

    def train_per_epoch(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        mix_loss = AverageMeter('MixLoss', ':.4e')
        cont_loss = AverageMeter('ContrastiveLoss', ':.4e')
        losses = AverageMeter('Loss', ':.4e')

        real_features, real_labels = self.get_real_features()
        knn = KNeighborsClassifier(n_neighbors=self.real_per_class_num[-1]*2)
        knn.fit(real_features, real_labels)

        # switch to train mode
        self.model.train()

        end = time.time()

        for i, ((input_1, target_1, is_syn), (input_2, target_2, _)) in enumerate(zip(self.train_loader, self.mix_loader)):
            batch_size = target_1.shape[0]
            input_1 = torch.cat([input_1[0], input_1[1]], dim=0)
            input_2 = torch.cat([input_2[0], input_2[1]], dim=0)
            if self.device is not None:
                input_1 = input_1.cuda(self.device)
                target_1 = target_1.cuda(self.device)
                input_2 = input_2.cuda(self.device)
                target_2 = target_2.cuda(self.device)
            
            input_view1, input_view2 = torch.split(input_1, [batch_size, batch_size], dim=0)
            input_mix_aug1, _ = torch.split(input_2, [batch_size, batch_size], dim=0)
            
            one_hot_target_1 = F.one_hot(target_1, num_classes=self.num_classes)
            one_hot_target_2 = F.one_hot(target_2, num_classes=self.num_classes)
            x_mixup, y_mixup = mixup(input_view1, input_mix_aug1, one_hot_target_1, one_hot_target_2)
            x_cutmix, y_cutmix = cutmix(input_view1, input_mix_aug1, one_hot_target_1, one_hot_target_2)

            _, output_mixup = self.model(x_mixup)
            _, output_cutmix = self.model(x_cutmix)
            z1, _ = self.model(input_view1)
            z2, _ = self.model(input_view2)
            z1_ = z1.cpu().detach().numpy()
            z2_ = z2.cpu().detach().numpy()
            target_1_ = target_1.cpu().detach().numpy()

            new_label = self.relable(knn, z1_, z2_, is_syn.numpy(), target_1_)
            new_label = torch.from_numpy(new_label)
            if self.device is not None:
                new_label = new_label.cuda(self.device)
            contloss = self.SCL(torch.stack((z1, z2), dim=1), target_1)

            mixloss = self.ML(output_mixup, output_cutmix, y_mixup, y_cutmix)
            loss = mixloss + contloss
            #loss = mixloss
            losses.update(loss.item(), batch_size)
            mix_loss.update(mixloss.item(), batch_size)
            cont_loss.update(contloss.item(), batch_size)
            #cont_loss.update(0, batch_size)

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % self.print_freq == 0) or (i == len(self.train_loader) - 1):
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'MixLoss {mix_loss.val:.4f} ({mix_loss.avg:.4f})\t'
                    'ContrastiveLoss {cont_loss.val:.4f} ({cont_loss.avg:.4f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch, i, len(self.train_loader), batch_time=batch_time,
                        mix_loss=mix_loss, cont_loss=cont_loss, loss=losses))

    def validate(self, epoch):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('CELoss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        eps = np.finfo(np.float64).eps

        # switch to evaluate mode
        self.model.eval()
        all_preds = []
        all_targets = []

        end = time.time()
        with torch.no_grad():
            for i, (input, target, is_syn) in enumerate(self.val_loader):
                if self.device is not None:
                    input = input.cuda(self.device)
                    target = target.cuda(self.device)

                # compute output
                z, output = self.model(input)

                loss_ce = self.CE(output, target)
                loss = loss_ce

                output = output.float()
                loss = loss.float()

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1.update(acc1[0], input.size(0))
                top5.update(acc5[0], input.size(0))
                losses.update(loss.item(), input.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                _, pred = torch.max(output, 1)
                all_preds.extend(pred.detach().cpu().numpy())
                all_targets.extend(target.detach().cpu().numpy())
            
            cf = confusion_matrix(all_targets, all_preds).astype(float)
            cls_cnt = cf.sum(axis=1)
            cls_hit = np.diag(cf)
            cls_acc = cls_hit / cls_cnt
            output = ('EPOCH: {epoch} {flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'\
                      .format(epoch=epoch + 1 , flag='val', top1=top1, top5=top5, loss=losses))
            self.log.info(output)
            many_shot = self.real_per_class_num > 100
            medium_shot = (self.real_per_class_num <= 100) & (self.real_per_class_num > 20)
            few_shot = self.real_per_class_num <= 20
            print("many avg, med avg, few avg",
                  float(sum(cls_acc[many_shot]) * 100 / (sum(many_shot) + eps)),
                  float(sum(cls_acc[medium_shot]) * 100 / (sum(medium_shot) + eps)),
                  float(sum(cls_acc[few_shot]) * 100 / (sum(few_shot) + eps))
                  )
            
        return top1.avg

    
    def get_real_features(self):
        self.model.eval()

        features = None
        labels = None

        with torch.no_grad():
            for i, (input, target, is_syn) in enumerate(self.train_loader):
                input = torch.cat([input[0], input[1]], dim=0)
                target = target.repeat(2)
                is_syn = is_syn.repeat(2)
                if self.device is not None:
                    input = input.cuda(self.device)
                
                # compute output
                z, output = self.model(input)

                is_syn = is_syn.numpy()
                z = z.cpu().numpy()
                target = target.numpy()
                z = z[is_syn == 0]
                target = target[is_syn == 0]

                if z.shape[0] > 0:
                    if features is None:
                        features = z
                        labels = target
                    else:
                        features = np.concatenate([features, z], axis=0)
                        labels = np.concatenate([labels, target], axis=0)

            return features, labels

    
    def relable(self, knn, z1, z2, is_syn, label):
        is_syn = is_syn.astype(bool)
        output1 = knn.predict(z1)
        output2 = knn.predict(z2)
        index = ((output1 == label) & (output2 == label) | (~is_syn))
        label[~index] = -1
        return label
        
