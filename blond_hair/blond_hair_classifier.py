import os
from collections import OrderedDict
import torch.nn as nn
import torchvision.transforms as transforms
import torch

torch.cuda.current_device()

import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import pytorch_lightning as pl
from pytorch_lightning.root_module.root_module import LightningModule
import tensorpack.dataflow as df
from tensorpack import imgaug
import pandas as pd
import numpy as np
import glob
import cv2
import albumentations as AB

from torchvision.models import vgg16_bn
from test_tube import HyperOptArgumentParser, Experiment
from pytorch_lightning.models.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

train_max_index = 162770
val_max_index = 182637
test_max_index = 202599

TRAIN = 'TRAIN'
VAL = 'VAL'
TEST = 'TEST'


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


class CelebABlondHairDataFlow(df.DataFlow):
    def __init__(self, params):
        # Setup image dir contains blond hair and not-blond hair
        self.data_dir = params['data_dir']
        self.is_training = params['training']
        self.max_samples = params['max_samples']

        # expected_dir = 'train' if self.is_training else 'val'

        self.img_dir = os.path.join(self.data_dir, '')
        self.blond_dir = os.path.join(self.img_dir, 'blond')
        self.not_blond_dir = os.path.join(self.img_dir, 'not_blond')

        # List of blond hair and not-blond hair images  
        self.train_list = {'blond': [], 'not_blond': []}
        self.train_list['blond'] = (glob.glob(self.blond_dir + '/*.jpg'))
        self.train_list['not_blond'] = (glob.glob(self.not_blond_dir + '/*.jpg'))

        # Size of each hair type
        self.blond_size = len(self.train_list['blond'])
        self.not_blond_size = len(self.train_list['not_blond'])

        # Dataset size is contrained by max_samples
        max_size = min(self.max_samples, self.blond_size + self.not_blond_size)
        self.size = {'total': max_size,\
            'blond': self.blond_size, 'not_blond':self.not_blond_size}

    
    def __len__(self):
        return self.size['total']


    def __iter__(self):
        for _ in range(self.size['total']):
            # Each step, randomize an index
            rdix = np.random.randint(self.size['total'])
            chosen_type = 'blond' if (rdix % 2 == 0) else 'not_blond'

            # Get/read image and label
            # Randomly chooses blond/not-blond dir and image index
            img_name = self.train_list[chosen_type][int(rdix / (self.size[chosen_type]))]
            img = cv2.imread(img_name)

            # Blond hair is 1, other is 0
            label = 1 if chosen_type == 'blond' else 0

            yield [img, label]


class BlondHairClassifier(LightningModule):
    '''
    Model to classify faces with long hair in CelebA dataset
    '''

    def __init__(self, params):
        super(BlondHairClassifier, self).__init__()
        
        # Model hyperparams setup
        self.batch_size = params['batch_size']
        # self.input_dims = params['input_dims']
        self.num_classes = params['num_classes']
        self.learning_rate = params['learning_rate']

        self.ds_train = None
        self.ds_valid = None
        self.ds_test = None

        # Build model
        self.__build_model(params['pretrained'])


    def forward(self, x):
        x = x.transpose(1, 3)
        return self.vgg16.forward(x)

    
    def __init_weights(self):
        pass


    def __build_model(self, is_pretrained):
        self.vgg16 = vgg16_bn(is_pretrained)
        print(self.vgg16.classifier[6].out_features)

        # Freeze training for all layers
        for m_param in self.vgg16.features.parameters():
            m_param.require_grad = False

        # Finetuning step
        num_features = self.vgg16.classifier[6].in_features

        # Remove last layer
        features = list(self.vgg16.classifier.children())[:-1]

        # Add layer with 2 outputs - blond, not-blond
        features.extend([nn.Linear(num_features, self.num_classes)])
        self.vgg16.classifier = nn.Sequential(*features)

        self.loss_func = nn.CrossEntropyLoss()

        print(self.vgg16)


    def cross_entropy_loss(self, inp, targ):
        return self.loss_func(inp, targ)


    def training_step(self, batch, batch_i):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # Forward pass
        x, y = batch
        y_hat = self.forward(x)

        # Calculate loss
        loss = self.cross_entropy_loss(y_hat, y)

        output = OrderedDict({
            'loss': loss
        })

        return output


    def validation_step(self, batch, batch_i):
        """
        Lightning calls this inside the validation loop
        :param data_batch:
        :return:
        """
        x, y = batch
        y_hat = self.forward(x)
        loss = self.cross_entropy_loss(y_hat, y)
        output = OrderedDict({
            'val_loss': loss
        })

        return output
        

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'avg_val_loss': avg_loss}


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):
        """
        Return optimizer to use
        :return: list of optimizers
        """
        optimizer = optim.SGD(self.vgg16.parameters(), self.learning_rate, momentum=0.9)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

        return [optimizer], [scheduler]

    @pl.data_loader
    def tng_dataloader(self):
        print('tng data loader called')
        return self.ds_train

    
    @pl.data_loader
    def val_dataloader(self):
        print('val data loader called')
        return self.ds_val

    
    @pl.data_loader
    def test_dataloader(self):
        print('test data loader called')
        return self.ds_test

    
    def fetch_dataflow(self, ds_train=None, ds_val=None, ds_test=None):
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test
    


if __name__ == '__main__':
    EPOCHS = 20
    params = {
        'data_dir': 'data/train',
        'max_samples': 10000,
        'training': True,
        'batch_size': 8,
        'num_classes': 2,
        'learning_rate': 0.001,
        'pretrained': True
    }

    # Set of transformation
    data_transform = {
        TRAIN: 
            transforms.Compose([
            # Data augmentation is a good practice for the train set
            # Here, we randomly crop the image to 224x224 and
            # randomly flip it horizontally. 
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]),
        VAL: 
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]),
        TEST:
            transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
    }

    aug_train = [imgaug.Albumentations(data_transform[TRAIN])]
    aug_val = [imgaug.Albumentations(data_transform[VAL])]

    ag_image = [
                    imgaug.Albumentations(AB.RandomBrightness(limit=0.2, p=0.5)), 
                    imgaug.Albumentations(AB.RandomContrast(limit=0.2, p=0.5)), 
                    imgaug.Albumentations(AB.GaussianBlur(blur_limit=7, p=0.5)), 
                    imgaug.Resize((256, 256)),
                    imgaug.RandomCrop((224, 224)),
                    # imgaug.Flip(horiz=True),
                    # imgaug.Flip(vert=True),
                    imgaug.Albumentations(AB.RandomRotate90(p=1))
                ]

    # Define training dataset
    ds_train = CelebABlondHairDataFlow(params)
    # Apply augmentation for image only
    ds_train = df.AugmentImageComponent(ds_train, ag_image, 0)
    ds_train = df.BatchData(ds_train, batch_size=params['batch_size'])
    ds_train = df.PrintData(ds_train)
    ds_train = df.MapData(ds_train, lambda dp: [torch.tensor(dp[0], dtype=torch.float), 
                                                torch.tensor(dp[1], dtype=torch.long), 
                                                ])

    params['training'] = False
    params['data_dir'] = 'data/val'

    ag_val = [imgaug.Resize((224, 224))]
    ds_val = CelebABlondHairDataFlow(params)
    ds_val = df.AugmentImageComponent(ds_val, ag_val, 0)
    ds_val = df.BatchData(ds_val, batch_size=params['batch_size'])
    ds_val = df.MapData(ds_val, lambda dp: [torch.tensor(dp[0], dtype=torch.float), 
                                            torch.tensor(dp[1], dtype=torch.long), 
                                            ])

    ds_test = ds_val

    model = BlondHairClassifier(params)
    model.fetch_dataflow(ds_train, ds_val, ds_test)

    #-----------------------------------------------------------------------
    # 2 INIT TEST TUBE EXP
    #-----------------------------------------------------------------------

    # init experiment
    exp = Experiment(
        name='blond_hair', #hyperparams.experiment_name,
        save_dir='runs', #hyperparams.test_tube_save_path,
        # autosave=False,
        # description='experiment'
    )

    exp.save()

    #-----------------------------------------------------------------------
    # 3 DEFINE CALLBACKS
    #-----------------------------------------------------------------------
    model_save_path = 'vgg16_blond_hair' #'{}/{}/{}'.format(hparams.model_save_path, exp.name, exp.version)
    early_stop = EarlyStopping(
        monitor='avg_val_loss',
        patience=5,
        verbose=True,
        mode='auto'
    )

    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        # save_best_only=True,
        # save_weights_only=True,
        verbose=True,
        monitor='val_loss',
        mode='auto',
        period=100,
    )

    #-----------------------------------------------------------------------
    # 4 INIT TRAINER
    #-----------------------------------------------------------------------
    trainer = Trainer(
        experiment=exp,
        checkpoint_callback=checkpoint,
        # early_stop_callback=early_stop,
        max_nb_epochs=EPOCHS, 
        gpus='0' #map(int, args.gpu.split(',')), #hparams.gpus,
        # distributed_backend='ddp'
    )

    #-----------------------------------------------------------------------
    # 5 START TRAINING
    #-----------------------------------------------------------------------
    trainer.fit(model)
