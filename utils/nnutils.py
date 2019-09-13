import torch
import torch.nn as nn
import numpy as np


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


def latent_to_image(latent, hparams):
        size = latent.size()
        img = torch.zeros(size[0], 1, hparams['image_size'], hparams['image_size'])#size[1]
        for i in range(hparams['image_size']):
            if i < size[1]:
                img[:,:,:,i] = latent[:,i]
            else:
                img[:,:,:,i] = torch.ones_like(img[:,:,:,0])

        return img


def fake_noise_img_creator(latent_batch, fake_img_batch, hparams):
    l_size = latent_batch.size()
    f_size = fake_img_batch.size()
    assert l_size[0] == f_size[0]

    latent_img = latent_to_image(latent_batch, hparams)
    latent_img = torch.from_numpy(np.stack((latent_img[:,0,:],)*3, axis=1))

    img = torch.cat([latent_img, fake_img_batch], axis=0)

    return img