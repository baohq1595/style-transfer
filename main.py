from __future__ import print_function
#%matplotlib inline
import argparse
import os
import datetime
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.models import vgg16_bn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time import gmtime, strftime
from gan.dcgan.dcgan import DCGAN

if __name__ == "__main__":
    # Set random seem for reproducibility
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    is_show_samples = False

    params = {
        'dataroot': "data/wheel_data/images_fuji", # Root directory for dataset
        'loss': 'l2', # Name of used loss function, support l2 and bce
        'workers': 4, # Number of workers for dataloader
        'batch_size': 64, # Batch size during training
        'image_size': 64, # Spatial size of training images. All images will be resized to this
                        # size using a transformer.
        'nc': 3, # Number of channels in the training images. For color images this is 3
        'nz': 40, # Size of z latent vector (i.e. size of generator input)
        'ngf': 64, # Size of feature maps in generator
        'ndf': 64, # Size of feature maps in discriminator
        'num_epochs': 300, # Number of training epochs
        'lr': 0.0002, # Learning rate for optimizers
        'beta1': 0.5, # Beta1 hyperparam for Adam optimizers
        'gpu': '0', # Number of GPUs available. Use 0 for CPU mode.
        'debug': 1, # Set if using summarywriter, capturing generator progress
        'save_path': 'outputs' # path folder to save model and other stuffs
    }

    # We can use an image folder dataset the way we have it setup.
    # Create the dataset
    dataset = dset.ImageFolder(root=params['dataroot'],
                                transform=transforms.Compose([
                                    transforms.Resize(params['image_size']),
                                    transforms.CenterCrop(params['image_size']),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
                            )
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params['batch_size'],
                                            shuffle=True, num_workers=params['workers'])

    model = DCGAN(params, dataloader)
    model.train()