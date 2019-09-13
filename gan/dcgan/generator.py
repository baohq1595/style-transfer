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

class Generator(nn.Module):
    def __init__(self, params, is_sub_pixel=False):
        super(Generator, self).__init__()

        # Initialize model parameters
        self.z_dims = params['nz']
        self.gen_feature_dims = params['ngf']
        self.num_channels = params['nc']

        if not is_sub_pixel:
            self.model = nn.Sequential(
                # input is z, going into a conv layer
                nn.ConvTranspose2d(self.z_dims, self.gen_feature_dims * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(self.gen_feature_dims * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(self.gen_feature_dims * 8, self.gen_feature_dims * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.gen_feature_dims * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8,
                nn.ConvTranspose2d(self.gen_feature_dims * 4, self.gen_feature_dims * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.gen_feature_dims * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(self.gen_feature_dims * 2, self.gen_feature_dims, 4, 2, 1, bias=False),
                nn.BatchNorm2d(self.gen_feature_dims),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(self.gen_feature_dims, self.num_channels, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        else:
            self.model = nn.Sequential(
                # input is z, going into a conv layer
                # nn.ConvTranspose2d(self.z_dims, self.gen_feature_dims * 8, 4, 1, 0, bias=False),
                nn.Conv2d(self.z_dims, self.gen_feature_dims * 16 * 4, 1, 1, 0, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(self.gen_feature_dims * 16),
                nn.ReLU(True),
                # state size. (ngf*8) x 2 x 2
                # nn.ConvTranspose2d(self.gen_feature_dims * 8, self.gen_feature_dims * 4, 4, 2, 1, bias=False),
                nn.Conv2d(self.gen_feature_dims * 16, self.gen_feature_dims * 8 * 4, 2, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(self.gen_feature_dims * 8),
                nn.ReLU(True),
                # state size. (ngf*4) x 4 x 4,
                # nn.ConvTranspose2d(self.gen_feature_dims * 4, self.gen_feature_dims * 2, 4, 2, 1, bias=False),
                nn.Conv2d(self.gen_feature_dims * 8, self.gen_feature_dims * 4 * 4, 4, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(self.gen_feature_dims * 4),
                nn.ReLU(True),
                # state size. (ngf*2) x 8 x 8
                # nn.ConvTranspose2d(self.gen_feature_dims * 2, self.gen_feature_dims, 4, 2, 1, bias=False),
                nn.Conv2d(self.gen_feature_dims * 4, self.gen_feature_dims * 2 * 4, 4, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(self.gen_feature_dims * 2),
                nn.ReLU(True),
                # state size. (ngf) x 16 x 16
                nn.Conv2d(self.gen_feature_dims * 2, self.gen_feature_dims * 1 * 4, 4, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(self.gen_feature_dims),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.Conv2d(self.gen_feature_dims, self.num_channels * 4, 4, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
    
    
    def forward(self, x):
        return self.model(x)