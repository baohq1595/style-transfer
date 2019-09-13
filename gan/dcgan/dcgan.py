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
from torch.utils.tensorboard import SummaryWriter

from utils.nnutils import fake_noise_img_creator, weights_init
from gan.dcgan.generator import Generator
from gan.dcgan.discriminator import Discriminator

class DCGAN():
    def __init__(self, hparams, dataloader):
        super(DCGAN, self).__init__()

        # Model hyperparams setup
        self.hparams = hparams

        # dataloader
        self.dataloader = dataloader
        self.save_path = hparams.get('save_path', 'outputs')
        self.__create_dirs(self.save_path)

        device_gpu = 'cuda:{}'.format(hparams['gpu'])
        self.device = torch.device(device_gpu if (torch.cuda.is_available()) else "cpu")

        # Create generator
        self.netG = Generator(self.hparams, is_sub_pixel=False).to(self.device)

        # Apply the weights_init function to randomly initialize all weights
        # to mean=0, stdev=0.2.
        self.netG.apply(weights_init)

        print('\n==============Generator=============\n', self.netG)

        # Create disciminator
        self.netD = Discriminator(self.hparams).to(self.device)
        # Apply the weights_init function to randomly initialize all weights
        # to mean=0, stdev=0.2.
        self.netD.apply(weights_init)
        
        print('\n==============Discriminator=============\n', self.netD)

        # ================================================
        # Define loss funciton
        # ================================================
        # Initialize BCELoss function
        self.criterion = nn.BCELoss()

        self.loss_type = self.hparams['loss']
        self.is_debug = self.hparams['debug']
    
    def g_criterion(self, d_fake_logits):
        '''
        Generator criterion for LSE loss (L2 norm)
        '''
        return torch.sum((d_fake_logits - 1)**2) / 2

    def d_criterion(self, d_real_logits, d_fake_logits):
        '''
        Discriminator criterion for LSE loss (L2 norm)
        '''
        return torch.sum((d_real_logits - 1)**2 + (d_fake_logits**2)) / 2

    def __create_dirs(self, path):
        log_dir = os.path.join(path, 'logs')
        result_dir = os.path.join(path, 'result')
        model_dir = os.path.join(path, 'model')
        
        dirs = [log_dir, result_dir, model_dir]
        for sub_dir in dirs:
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)


    
    def train(self):
        summwriter = None

        # Establish convention for real and fake labels during training
        # Create batch of latent vectors that we will use to visulize
        # the progression of the generator
        fixed_noise = torch.randn(64, self.hparams['nz'], 1, 1, device=self.device)

        if self.is_debug:
            summwriter = SummaryWriter(os.path.join(self.save_path, 'logs/wheel_gen_{}'.format(
                strftime("%Y-%m-%d-%H-%M-%S", gmtime()))))
            real_batch = next(iter(self.dataloader))

            summwriter.add_graph(model=Generator(self.hparams),\
                input_to_model=fixed_noise.detach().cpu())
            summwriter.add_graph(model=Discriminator(self.hparams), input_to_model=real_batch[0])

            summwriter.flush()

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
        G_losses = []
        D_losses = []
        iters = 0

        print('Start training...')

        for epoch in range(self.hparams['num_epochs']):
            # For each batch in the dataloader
            for i, data in enumerate(self.dataloader, 0):
                errD_real = 0.0
                errD_fake = 0.0
                errG = 0.0

                # Step 1: Update D-network
                # Maximize log(D(x)) + log(1 - D(G(z)))
                # Train with all-real batch
                self.netD.zero_grad()

                # Format batch
                real_cpu = data[0].to(self.device)
                b_size = real_cpu.size(0)
                label = torch.full((b_size,), real_label, device=self.device)

                # Forward pass real batch through D
                output_real = self.netD(real_cpu).view(-1)

                # Calculate loss on all-real batch
                if self.loss_type == 'bce':
                    errD_real = self.criterion(output_real, label)  
                    # Calculate gradients for D in backward pass
                    errD_real.backward()

                # Calculate gradients for D in backward pass
                D_x = output_real.mean().item()

                # Train with all-fake batch
                # Generate batch of latent vectors
                noise = torch.randn(b_size, self.hparams['nz'], 1, 1, device=self.device)

                # Generate fake image batch with G
                fake = self.netG(noise)
                label.fill_(fake_label)

                # Classify all fake batch with D
                output_fake = self.netD(fake.detach()).view(-1)

                # Calculate D's loss on the all-fake batch
                D_G_z1 = output_fake.mean().item()

                # Calculate the gradients for this batch
                if self.loss_type == 'bce':
                    errD_fake = self.criterion(output_fake, label)
                    errD_fake.backward()
                else:
                    errD_logits = self.d_criterion(output_real, output_fake) # L2-norm
                    errD_logits.backward()

                # Add the gradients from the all-real and all-fake batches
                errD = 0.0
                if self.loss_type == 'bce':
                    errD = errD_real + errD_fake
                else:
                    errD = errD_logits 
                # Update D
                optimizerD.step()
                
                # Step 2: Update G-network
                # Maximize log(D(G(z)))
                self.netG.zero_grad()
                label.fill_(real_label) # Fake labels are real for generator cost
                # Since we just updated D, perform another forward pass of all-fake
                # batch thru D
                output_gen = self.netD(fake).view(-1)

                # Calculate G's loss based on this output
                if self.loss_type == 'bce':
                    errG = self.criterion(output_gen, label)
                else:
                    errG = self.g_criterion(output_gen) # L2 norm

                # Calculate gradients for G
                errG.backward()
                D_G_z2 = output_gen.mean().item()

                # Update G
                optimizerG.step()

                # Output training stats
                if i % 50 == 0:
                    print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'\
                        % (epoch, self.hparams['num_epochs'], i, len(self.dataloader),\
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

                if self.is_debug:
                    summwriter.add_scalar('Loss_D', errD.item(), iters)
                    summwriter.add_scalar('Loss_G', errG.item(), iters)
                    summwriter.add_histogram('Noise Distribution', noise, iters)

                    # Save losses for plotting later
                    G_losses.append(errG.item())
                    D_losses.append(errD.item())

                    # Check how the generator is doing by saving G's output on fixed_noise
                    if (iters % 41 * 10 == 0) or ((epoch == self.hparams['num_epochs']-1) and (i == len(self.dataloader)-1)):
                        with torch.no_grad():
                            fake = self.netG(fixed_noise).detach().cpu()
                        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

                        img = fake_noise_img_creator(fixed_noise[:8], fake[:8], self.hparams)
                        summwriter.add_image('Images', vutils.make_grid(img, padding=2, normalize=True), iters)

                iters += 1

            if epoch % 80 == 0:
                torch.save(self.netG.state_dict(), os.path.join(self.save_path, 'model', 'generator{}-epoch-{}'.format(
                    strftime("%Y-%m-%d-%H-%M", gmtime()), epoch))
                )

                torch.save(self.netD.state_dict(), os.path.join(self.save_path, 'model', 'discriminator{}-epoch-{}'.format(
                    strftime("%Y-%m-%d-%H-%M", gmtime()), epoch))
                )

        # Result
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(self.save_path, 'result', 'loss-{}.png'.format(strftime("%Y-%m-%d-%H-%M", gmtime()))))

        # Save model
        torch.save(self.netG.state_dict(), os.path.join(self.save_path, 'model', 'generator{}'.format(
            strftime("%Y-%m-%d-%H-%M", gmtime())
        )))

        torch.save(self.netD.state_dict(), os.path.join(self.save_path, 'model', 'discriminator{}'.format(
            strftime("%Y-%m-%d-%H-%M", gmtime())
        )))

        if self.is_debug:
            #%%capture
            fig = plt.figure(figsize=(8,8))
            plt.axis("off")
            ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
            ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

            # Set up formatting for the movie files
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)

            ani.save(os.path.join(self.save_path, 'result', 'generating-{}.mp4'.format(strftime("%Y-%m-%d-%H-%M", gmtime()))), writer=writer)
            summwriter.close()