from __future__ import print_function
#%matplotlib inline
import argparse
import os
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

from torch.utils.tensorboard import SummaryWriter

from nnutils import *


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
                nn.ConvTranspose2d(self.z_dims, self.gen_feature_dims * 8, 4, 1, 0, bias=False),
                nn.Conv2d(self.gen_feature_dims * 8, self.gen_feature_dims * 8 * 4, 3, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(self.gen_feature_dims * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(self.gen_feature_dims * 8, self.gen_feature_dims * 4, 4, 2, 1, bias=False),
                nn.Conv2d(self.gen_feature_dims * 4, self.gen_feature_dims * 4 * 4, 3, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(self.gen_feature_dims * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8,
                nn.ConvTranspose2d(self.gen_feature_dims * 4, self.gen_feature_dims * 2, 4, 2, 1, bias=False),
                nn.Conv2d(self.gen_feature_dims * 2, self.gen_feature_dims * 2 * 4, 3, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(self.gen_feature_dims * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(self.gen_feature_dims * 2, self.gen_feature_dims, 4, 2, 1, bias=False),
                nn.Conv2d(self.gen_feature_dims * 1, self.gen_feature_dims * 1 * 4, 3, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.BatchNorm2d(self.gen_feature_dims),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                # nn.Conv2d(self.gen_feature_dims, self.num_channels * 4, 4, 2, 1, bias=False),
                # nn.PixelShuffle(2),
                nn.ConvTranspose2d(self.gen_feature_dims, self.num_channels, 4, 2, 1, bias=False),
                nn.Conv2d(self.num_channels, self.num_channels * 4, 3, 2, 1, bias=False),
                nn.PixelShuffle(2),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
    
    
    def forward(self, x):
        return self.model(x)


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
            nn.Sigmoid()
        )

    
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


class DCGAN():
    def __init__(self, hparams, dataloader):
        super(DCGAN, self).__init__()

        # Model hyperparams setup
        self.hparams = hparams

        # dataloader
        self.dataloader = dataloader

        device_gpu = 'cuda:{}'.format(hparams['gpu'])
        self.device = torch.device(device_gpu if (torch.cuda.is_available()) else "cpu")

        # Create generator
        self.netG = Generator(params, is_sub_pixel=True).to(self.device)

        # Apply the weights_init function to randomly initialize all weights
        # to mean=0, stdev=0.2.
        self.netG.apply(weights_init)

        print('\n==============Generator=============\n', self.netG)

        # Create disciminator
        self.netD = Discriminator(params).to(self.device)
        # Apply the weights_init function to randomly initialize all weights
        # to mean=0, stdev=0.2.
        self.netD.apply(weights_init)
        
        print('\n==============Discriminator=============\n', self.netD)

        # ================================================
        # Define loss funciton
        # ================================================
        # Initialize BCELoss function
        self.criterion = nn.BCELoss()


    def __latent_to_image(self, latent):
        size = latent.size()
        img = torch.zeros(size[0], 1, self.hparams['image_size'], self.hparams['image_size'])#size[1]
        for i in range(self.hparams['image_size']):
            if i < size[1]:
                img[:,:,:,i] = latent[:,i]
            else:
                img[:,:,:,i] = torch.ones_like(img[:,:,:,0])

        return img


    def __fake_noise_img_creator(self, latent_batch, fake_img_batch):
        l_size = latent_batch.size()
        f_size = fake_img_batch.size()
        assert l_size[0] == f_size[0]

        latent_img = self.__latent_to_image(latent_batch)
        latent_img = torch.from_numpy(np.stack((latent_img[:,0,:],)*3, axis=1))

        # img = torch.zeros(f_size[0], f_size[1], self.hparams['image_size'], self.hparams['image_size'] + l_size[1])
        img = torch.cat([latent_img, fake_img_batch], axis=0)

        return img

    
    def train(self):
        summwriter = SummaryWriter('logs/face_gen/')
        # Establish convention for real and fake labels during training
        # Create batch of latent vectors that we will use to visulize
        # the progression of the generator
        fixed_noise = torch.randn(64, self.hparams['nz'], 1, 1, device=self.device)

        for i, data in enumerate(self.dataloader, 0):
            # summwriter.add_graph(model=Generator(self.hparams), input_to_model=torch.zeros(1, self.hparams['nz'], 1, 1))
            # summwriter.add_graph(model=Discriminator(self.hparams), input_to_model=torch.zeros(
            #     1, self.hparams['nc'], self.hparams['image_size'], self.hparams['image_size']
            # ))
            break

        real_label = 1
        fake_label = 0

        # Setup Adam optim for both G and D
        optimizerD = optim.Adam(self.netD.parameters(),\
            lr=self.hparams['lr'], betas=(self.hparams['beta1'], 0.99))
        
        optimizerG = optim.Adam(self.netG.parameters(),\
            lr=self.hparams['lr'], betas=(self.hparams['beta1'], 0.99))

        # ================================================
        # Training
        # ================================================
        # Lists to keep track of progress
        img_list = []
        latent_img_list = []
        G_losses = []
        D_losses = []
        iters = 0

        print('Start training...')

        for epoch in range(self.hparams['num_epochs']):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                # Step 1: Update D-network
                # Maximize log(D(x)) + log(1 - D(G(z)))
                # Train with all-real batch
                self.netD.zero_grad()
                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=self.device)
                # Forward pass real batch through D
                output = self.netD(real_cpu).view(-1)
                # Calculate loss on all-real batch
                errD_real = self.criterion(output, label)
                # Calculate gradients for D in backward pass
                errD_real.backward()
                D_x = output.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.hparams['nz'], 1, 1, device=self.device)
                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(fake_label)
                # Classify all fake batch with D
                output = self.netD(fake.detach()).view(-1)
                # Calculate D's loss on the all-fake batch
                errD_fake = self.criterion(output, label)
                # Calculate the gradients for this batch
                errD_fake.backward()
                D_G_z1 = output.mean().item()
                # Add the gradients from the all-real and all-fake batches
                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()


                # Step 2: Update G-network
                # Maximize log(D(G(z)))
                self.netG.zero_grad()
                label.fill_(real_label) # Fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake
                # batch thru D
                output = self.netD(fake).view(-1)
                # Calculate G's loss based on this output
                errG = self.criterion(output, label)
                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output.mean().item()
                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'\
                        % (epoch, self.hparams['num_epochs'], i, len(self.dataloader),\
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                # summwriter.add_scalar('Loss_D', errD.item(), i)
                # summwriter.add_scalar('Loss_G', errG.item(), i)
                # summwriter.add_scalar('D(x)', D_x, i)
                # summwriter.add_scalar('D(x)', D_G_z1, i)
                # summwriter.add_histogram('Noise Distribution', noise, i)

                # Save losses for plotting later
                G_losses.append(errG.item())
                D_losses.append(errD.item())

                # Check how the generator is doing by saving G's output on fixed_noise
                if (iters % 500 == 0) or ((epoch == self.hparams['num_epochs']-1) and (i == len(dataloader)-1)):
                    with torch.no_grad():
                        fake = self.netG(fixed_noise).detach().cpu()
                    # img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
                    # latent_img_list.append(vutils.make_grid(self.__latent_to_image(fixed_noise), padding=2, normalize=True))

                    img = self.__fake_noise_img_creator(fixed_noise[:8], fake[:8])
                    img_list.append(vutils.make_grid(img, padding=2, normalize=True))
                    # summwriter.add_image('Images', vutils.make_grid(img, padding=2, normalize=True), iters)
                    # summwriter.add_image('Generated Images', img_list[-1])
                    # summwriter.add_image('Noise Images', latent_img_list[-1])

                iters += 1

        # Result
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig('loss.png')

        #%%capture
        fig = plt.figure(figsize=(8,8))
        plt.axis("off")
        ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
        ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

        # HTML(ani.to_jshtml())
        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

        ani.save('sub_pix_res.mp4', writer=writer)
        # summwriter.close()




if __name__ == "__main__":
    # Set random seem for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    is_show_samples = False

    params = {
        'dataroot': "data", # Root directory for dataset
        'workers': 4, # Number of workers for dataloader
        'batch_size': 256, # Batch size during training
        'image_size': 64, # Spatial size of training images. All images will be resized to this
                        # size using a transformer.
        'nc': 3, # Number of channels in the training images. For color images this is 3
        'nz': 40, # Size of z latent vector (i.e. size of generator input)
        'ngf': 64, # Size of feature maps in generator
        'ndf': 64, # Size of feature maps in discriminator
        'num_epochs': 10, # Number of training epochs
        'lr': 0.0002, # Learning rate for optimizers
        'beta1': 0.5, # Beta1 hyperparam for Adam optimizers
        'gpu': '0' # Number of GPUs available. Use 0 for CPU mode.
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

    