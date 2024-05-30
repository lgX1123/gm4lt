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
                 syn_path='/mnt/ssd_1/gxli/datasets/imagenet/test_imgs',
                 txt_path='./data/imagenet/',
                 train=True,
                 transform=None,
                 real_only=False):
        
        self.transform = transform
        self.train = train
        self.real_path = real_path
        self.syn_path = syn_path
        self.txt_path = txt_path

        if train:
            if real_only:
                self.data = self.produce_real_train_data()
            else:
                self.data = self.produce_train_data()
        else:
            self.data = self.produce_test_data()

        self.x = self.data['x']
        self.y = self.data['y']
        self.is_syn = self.data['is_syn']
        self.targets = self.data['y']

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, item):
        x, y, is_syn = self.x[item], self.y[item], self.is_syn[item]

        with open(x, 'rb') as f:
            img = Image.open(f).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        return img, y, is_syn

    
    def get_real_per_class_num(self):
        return self.real_per_class_num

    def get_syn_per_class_num(self):
        return self.syn_per_class_num

    def produce_test_data(self):
        img_path = []
        labels = []
        data_is_syn = []
        with open(self.txt_path + 'ImageNet_LT_test.txt', 'r') as f:
            for line in f:
                img_path.append(os.path.join(self.real_path, line.split()[0]))
                labels.append(int(line.split()[1]))
                data_is_syn.append(0)
        
        dataset = {
            "x": img_path,
            "y": labels,
            "is_syn": data_is_syn
        }

        return dataset

    def produce_train_data(self):
        img_path = []
        labels = []
        data_is_syn = []
        with open(self.txt_path + 'ImageNet_LT_train.txt', 'r') as f:
            for line in f:
                img_path.append(os.path.join(self.real_path, line.split()[0]))
                labels.append(int(line.split()[1]))
                data_is_syn.append(0)
        
        num_classes = len(np.unique(labels))
        self.real_per_class_num = [0] * num_classes
        for label in labels:
            self.real_per_class_num[label] += 1

        data_num_per_class = max(self.real_per_class_num)
        self.syn_per_class_num = [data_num_per_class - i for i in self.real_per_class_num]
        
        for cls, syn_num in enumerate(self.syn_per_class_num):
            random_syn_index = np.random.choice(data_num_per_class, syn_num, replace=False)
            for i in random_syn_index:
                img_path.append(os.path.join(self.syn_path, str(cls), str(i) + '.JPEG'))
                labels.append(cls)
                data_is_syn.append(1)
        
        dataset = {
            "x": img_path,
            "y": labels,
            "is_syn": data_is_syn
        }

        return dataset
        
    def produce_real_train_data(self):
        img_path = []
        labels = []
        data_is_syn = []
        with open(self.txt_path + 'ImageNet_LT_train.txt', 'r') as f:
            for line in f:
                img_path.append(os.path.join(self.real_path, line.split()[0]))
                labels.append(int(line.split()[1]))
                data_is_syn.append(0)
        
        dataset = {
            "x": img_path,
            "y": labels,
            "is_syn": data_is_syn
        }

        return dataset
