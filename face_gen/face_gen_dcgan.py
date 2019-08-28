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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
sys.path.append('D:\\workspace\\python\\style-transfer')

from nnutils import *


class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()

        # Initialize model parameters
        self.z_dims = params['nz']
        self.gen_feature_dims = params['ngf']
        self.num_channels = params['nc']

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
        'workers': 0, # Number of workers for dataloader
        'batch_size': 128, # Batch size during training
        'image_size': 64, # Spatial size of training images. All images will be resized to this
                        # size using a transformer.
        'nc': 3, # Number of channels in the training images. For color images this is 3
        'nz': 100, # Size of z latent vector (i.e. size of generator input)
        'ngf': 64, # Size of feature maps in generator
        'ndf': 64, # Size of feature maps in discriminator
        'num_epochs': 5, # Number of training epochs
        'lr': 0.0002, # Learning rate for optimizers
        'beta1': 0.5, # Beta1 hyperparam for Adam optimizers
        'ngpu': 1 # Number of GPUs available. Use 0 for CPU mode.
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

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and params['ngpu'] > 0) else "cpu")

    # Plot some training images
    real_batch = next(iter(dataloader))
    if is_show_samples:
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
        plt.show()


    # Create generator
    netG = Generator(params).to(device)

    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netG.apply(weights_init)

    print('\n==============Generator=============\n', netG)

    # Create disciminator
    netD = Discriminator(params).to(device)
    # Apply the weights_init function to randomly initialize all weights
    # to mean=0, stdev=0.2.
    netD.apply(weights_init)
    
    print('\n==============Discriminator=============\n', netD)

    # ================================================
    # Define loss funciton
    # ================================================
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visulize
    # the progression of the generator
    fixed_noise = torch.randn(64, params['nz'], 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1
    fake_label = 0

    # Setup Adam optim for both G and D
    optimizerD = optim.Adam(netD.parameters(),\
        lr=params['lr'], betas=(params['beta1'], 0.99))
    
    optimizerG = optim.Adam(netG.parameters(),\
        lr=params['lr'], betas=(params['beta1'], 0.99))

    # ================================================
    # Training
    # ================================================
    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print('Start training...')

    for epoch in range(params['num_epochs']):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # Step 1: Update D-network
            # Maximize log(D(x)) + log(1 - D(G(z)))
            # Train with all-real batch
            netD.zero_grad()
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            # Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, params['nz'], 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            optimizerD.step()


            # Step 2: Update G-network
            # Maximize log(D(G(z)))
            netG.zero_grad()
            label.fill_(real_label) # Fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake
            # batch thru D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()

            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'\
                    % (epoch, params['num_epochs'], i, len(dataloader),\
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Save losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == params['num_epochs']-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

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
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    ani.save('im.mp4', writer=writer)