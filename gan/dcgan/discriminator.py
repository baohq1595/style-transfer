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

class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()

        # Initialize model parameters
        self.dis_feature_dims = params['ndf']
        self.num_channels = params['nc']

        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.dis_feature_dims, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.dis_feature_dims, self.dis_feature_dims * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_feature_dims * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.dis_feature_dims * 2, self.dis_feature_dims * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_feature_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.dis_feature_dims * 4, self.dis_feature_dims * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_feature_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.dis_feature_dims * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        # self.logits = model
        # self.model = self.logits.add_module('sigmoid', nn.Sigmoid())

    
    def forward(self, x):
        return self.model(x)


class VGG16_Discriminator(nn.Module):
    def __init__(self, params):
        super(VGG16_Discriminator, self).__init__()

        # Initialize model parameters
        self.dis_feature_dims = params['ndf']
        self.num_channels = params['nc']

        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(self.num_channels, self.dis_feature_dims, 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.dis_feature_dims),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 64 x 64
            nn.Conv2d(self.dis_feature_dims, self.dis_feature_dims, 3, 2, 1, bias=False),
            nn.BatchNorm2d(self.dis_feature_dims),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.dis_feature_dims, self.dis_feature_dims * 2, 3, 1, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 2) x 32 x 32
            nn.Conv2d(self.dis_feature_dims * 2, self.dis_feature_dims * 2, 3, 2, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 2) x 16 x 16
            nn.Conv2d(self.dis_feature_dims * 2, self.dis_feature_dims * 4, 3, 1, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 4) x 16 x 16
            nn.Conv2d(self.dis_feature_dims * 4, self.dis_feature_dims * 4, 3, 1, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 4) x 16 x 16
            nn.Conv2d(self.dis_feature_dims * 4, self.dis_feature_dims * 4, 3, 2, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 4) x 8 x 8
            nn.Conv2d(self.dis_feature_dims * 4, self.dis_feature_dims * 8, 3, 1, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 8) x 8 x 8
            nn.Conv2d(self.dis_feature_dims * 8, self.dis_feature_dims * 8, 3, 1, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 8) x 8 x 8
            nn.Conv2d(self.dis_feature_dims * 8, self.dis_feature_dims * 8, 3, 2, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 8) x 4 x 4
            nn.Conv2d(self.dis_feature_dims * 8, self.dis_feature_dims * 8, 3, 1, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 8) x 4 x 4
            nn.Conv2d(self.dis_feature_dims * 8, self.dis_feature_dims * 8, 3, 2, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 8) x 2 x 2
            nn.Conv2d(self.dis_feature_dims * 8, self.dis_feature_dims * 8, 3, 1, 1),
            nn.BatchNorm2d(self.dis_feature_dims * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 8) x 2 x 2
            nn.Conv2d(self.dis_feature_dims * 8, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        return self.model(x)
