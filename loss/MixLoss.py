import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixLoss(nn.Module):
    def __init__(self, alpha=(1, 1)):
        super(MixLoss, self).__init__()
        self.alpha = alpha

    def forward(self, output_mixup, output_cutmix, y_mixup, y_cutmix):
        mixup = -torch.mean(torch.sum(F.log_softmax(output_mixup, dim=1) * y_mixup, dim=1))
        cutmix = -torch.mean(torch.sum(F.log_softmax(output_cutmix, dim=1) * y_cutmix, dim=1))
        return self.alpha[0] * mixup + self.alpha[1] * cutmix