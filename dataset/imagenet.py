import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from PIL import Image

from torchvision import transforms


class ImageNet_LT(Dataset):
    def __init__(self,
                 real_path='/mnt/ssd_1/data/imagenet/',
                 syb_path='',
                 txt_path='./data/imagenet/',
                 train=True,
                 transform=None):
        
        self.img_path = []
        self.labels = []
        self.transform = transform
        if train:
            with open(txt_path + 'ImageNet_LT_train.txt', 'r') as f:
                for line in f:
                    self.img_path.append(os.path.join(real_path, line.split()[0]))
                    self.labels.append(int(line.split()[1]))
        
        num_classes = len(np.unique(self.labels))
        self.real_per_class_num = [0] * num_classes
        for label in self.labels:
            self.real_per_class_num[label] += 1
        
        

        
    def __len__(self):
        return len(self.labels)

    def 
