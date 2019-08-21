import os
from collections import OrderedDict
import torch.nn as nn
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from test_tube import HyperOptArgumentParser
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
import tensorpack.dataflow as df
import pandas as pd
import numpy as np

from torchvision.models import vgg16_bn

train_max_index = 162770
val_max_index = 182637
test_max_index = 202599


class CelebADataFlow(df.DataFlow):
    def __init__(self, data_dir, params):
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, 'images')

        # Only classify blond hair or not
        self.classified_attr = params['attr'][0]

        # Number of training samples
        self.max_train_samples = params['train_samples']

        # Get attributes list
        split_hint_path = os.path.join(data_dir, 'list_eval_partition.txt')
        attr_path = os.path.join(data_dir, 'list_attr_celeba.txt')
        self.df_partition = pd.read_csv(split_hint_path, delim_whitespace=True, names=['image_id', 'partition'])
        self.df_attr = pd.read_csv(attr_path, delim_whitespace=True, skiprows=1)
        # self.df_attr.set_index('')
        self.__get_train_list()

        assert len(self.train_list) == len(self.train_lb_list)

        self.size = len(self.train_lb_list)

    
    def __len__(self):
        return len(self.train_list)


    def __iter__(self):
        for _ in range(len(self.train_list)):
            # Each step, randomize an index
            rdix = np.random.randint(self.size)

            # Get image/read image and label
            img_name = self.train_list[rdix]
            img = cv2.imread(os.path.join(self.img_dir))
            label = self.train_lb_list[rdix]

            yield [img, label]


    def __get_train_list(self):
        self.train_list = []

        # Get list of images which belongs to training partition
        for _, row in self.df_partition.iterrows():
            if row['partition'] == 0:
                self.train_list.append(row['image_id']) 

        # Get specified attr corresponding with training list
        self.train_lb_list = self.df_attr[self.classified_attr]
        self.train_lb_list = self.train_lb_list.loc[self.train_list]







class BlondHairClassifier(LightningModule):
    '''
    Model to classify faces with long hair in CelebA dataset
    '''

    def __init__(self, features, params, num_classes=2, init_weights=True):
        super(BlondHairClassifier, self).__init__()
        self.params = params

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    
    def __init_weights(self):
        pass


    def __build_mode(self):
        pass

if __name__ == '__main__':
    params = {'train_samples': 50, 'attr': ['Blond_Hair']}
    a = CelebADataFlow('data/celeba', params)