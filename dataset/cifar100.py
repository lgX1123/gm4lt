import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from PIL import Image
import random

class Cifar100(Dataset):
    def __init__(self, imbanlance_rate=0.1, file_path="./data/cifar100/cifar-100-python/", 
                 syn_file_path='./data/cifar100/synthetic/images/cifar100_syn_basic.pkl',
                 num_cls=100, transform=None, train=True):
        self.transform = transform
        assert 0.0 < imbanlance_rate <= 1, "imbanlance_rate must 0.0 < p < 1"
        self.num_cls = num_cls
        self.file_path = file_path
        self.syn_file_path = syn_file_path
        self.imbanlance_rate = imbanlance_rate

        if train is True:
            self.data = self.produce_train_data(self.imbanlance_rate)
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
        x = Image.fromarray(x)
        if self.transform is not None:
            x = self.transform(x)
        return x, y, is_syn

    def get_real_per_class_num(self):
        return self.real_per_class_num

    def get_syn_per_class_num(self):
        return self.syn_per_class_num

    def produce_test_data(self):
        with open(os.path.join(self.file_path, "test"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_test = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_test = dict[b'fine_labels']
        dataset = {
            "x": x_test,
            "y": y_test,
            "is_syn": [0] * x_test.shape[0]
        }

        return dataset

    def produce_train_data(self, imbanlance_rate):

        with open(os.path.join(self.file_path, "train"), 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
            x_train = dict[b'data'].reshape([-1, 3, 32, 32]).transpose(0, 2, 3, 1)
            y_train = dict[b'fine_labels']
        
        with open(self.syn_file_path, 'rb') as f:
            syn_data = pickle.load(f)
        
        y_train = np.array(y_train)
        data_x = None
        data_y = None
        data_is_syn = None

        real_data_percent = []
        syn_data_percent = []
        data_num = int(x_train.shape[0] / self.num_cls)
        total_data_percent = [data_num] * self.num_cls

        for cls_idx in range(self.num_cls):
            num = data_num * (imbanlance_rate ** (cls_idx / (self.num_cls - 1)))
            real_data_percent.append(int(num))
            syn_data_percent.append(data_num - int(num))
        # random.shuffle(real_data_percent)

        self.real_per_class_num = real_data_percent
        self.syn_per_class_num = syn_data_percent
        print("Imbanlance ration is {}".format(real_data_percent[0] / real_data_percent[-1]))
        print("Real images num per class: {}".format(real_data_percent))
        print("Synthetic images num per class: {}".format(syn_data_percent))
        print("Total images num per class: {}".format(total_data_percent))

        for i in range(1, self.num_cls + 1):
            a1 = y_train >= i - 1
            a2 = y_train < i
            index = a1 & a2

            task_train_x = x_train[index]
            label = y_train[index]
            data_num = task_train_x.shape[0]
            index = np.random.choice(data_num, real_data_percent[i - 1], replace=False)
            tem_data = task_train_x[index]

            syn_index = np.random.choice(data_num, syn_data_percent[i - 1], replace=False)
            tem_syn_data = syn_data[i - 1][syn_index]

            tem_data = np.concatenate([tem_data, tem_syn_data], axis=0)

            if data_x is None:
                data_x = tem_data
                data_y = label
                data_is_syn = [0] * len(index) + [1] * len(syn_index)
            else:
                data_x = np.concatenate([data_x, tem_data], axis=0)
                data_y = np.concatenate([data_y, label], axis=0)
                data_is_syn += [0] * len(index) + [1] * len(syn_index)

        dataset = {
            "x": data_x,
            "y": data_y.tolist(),
            "is_syn": data_is_syn,
        }

        return dataset