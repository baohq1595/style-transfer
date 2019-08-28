import torch.nn as nn


def weights_init(m):
    '''
    Custom weights initialization called on netG and netD
    From the DCGAN paper, the authors specify that all model weights shall be randomly 
    initialized from a Normal distribution with mean=0, stdev=0.02
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
