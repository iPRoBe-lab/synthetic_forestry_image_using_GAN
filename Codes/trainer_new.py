import os
import time
import torch
import datetime
from datetime import date
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
from torch.autograd import Variable
#from torchsummary import summary
from utils import *

class Trainer():
    def __init__(self, train_loader, test_data, roi, args, device):
        
        self.train_loader = train_loader
        self.test_data = test_data
        self.roi = roi
        self.image_height = args.image_height
        self.image_width = args.image_width
        self.max_value = self.test_data.dataset["gcc_rounded"].max()
        self.min_value = self.test_data.dataset["gcc_rounded"].min()
        self.n_channels = args.n_channels
        self.latent_dim = args.latent_dim
        self.embedding_dim = args.embedding_dim
        self.batch_size = args.batch_size
        self.num_samples = args.num_samples
        self.num_epoch = args.num_epoch
        self.device = device
        self.args = args

        folder_name = args.model + "_" + args.adv_loss + "_" + args.version
        self.sample_path = os.path.join(args.sample_path, folder_name)
        self.model_save_path = os.path.join(args.model_save_path, folder_name)
        self.log_path = os.path.join(args.log_path, folder_name)
        self.exec_summary_file_name = "Execution_summary_" + str(date.today().strftime("%m-%d-%Y"))+ ".txt"
        self.load_model = args.load_model
        self.build_model()
        
    def build_model(self):
        if self.args.model == "DCGAN":
            from DCGAN_adjusted_v13 import Generator, Discriminator
        elif self.args.model == "DCGAN_1":
            from DCGAN_1_adjusted_v13 import Generator, Discriminator
        # Initialize weights
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                #m.weight.data.normal_(0.0, 0.02)
                #torch.nn.init.xavier_uniform_(m.weight)
                #torch.nn.init.kaiming_normal_(m.weight)
                torch.nn.init.orthogonal_(m.weight)
            elif classname.find('BatchNorm') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)
            elif classname.find('Linear') != -1:
                #m.weight.data.normal_(0.0, 0.02)
                #torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.zero_()

        self.G = Generator(self.image_height, self.image_width, self.latent_dim, self.embedding_dim,
                           self.args.spectral_G, self.args.batch_norm_G, self.args.bias_G).to(self.device)
        self.D = Discriminator(self.image_height, self.image_width, self.embedding_dim,
                               self.args.spectral_D, self.args.batch_norm_D, self.args.bias_D, sigmoid=False).to(self.device)
        self.G = nn.DataParallel(self.G)
        self.D = nn.DataParallel(self.D)
        
        
        # Criterion 
        self.criterion = torch.nn.BCELoss()
        self.BCE_stable = torch.nn.BCEWithLogitsLoss()
        self.BCE_stable_noreduce = torch.nn.BCEWithLogitsLoss(reduction=False)
        # Loss and optimizer
        self.d_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.D.parameters()), lr=self.args.lr_D, 
                                            betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)
        self.g_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.G.parameters()), lr=self.args.lr_G, 
                                            betas=(self.args.beta1, self.args.beta2), weight_decay=self.args.weight_decay)
        # exponential weight decay on lr
        self.d_decay = torch.optim.lr_scheduler.ExponentialLR(self.d_optimizer, gamma=1-self.args.decay_D)
        self.g_decay = torch.optim.lr_scheduler.ExponentialLR(self.g_optimizer, gamma=1-self.args.decay_G)
    
        if self.load_model is None:
            self.G.apply(weights_init)
            self.D.apply(weights_init)
        elif self.args.load_model_path is None:
            self.G.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_G.pth'.format(self.load_model))))
            self.D.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_D.pth'.format(self.load_model))))
            self.g_optimizer.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_g_optimizer.pth'.format(self.load_model))))
            self.d_optimizer.load_state_dict(torch.load(os.path.join(self.model_save_path, '{}_d_optimizer.pth'.format(self.load_model))))
            print('Loaded trained models (Epoch: {})..!'.format(self.load_model))

        else:
            self.G.load_state_dict(torch.load(os.path.join(self.args.load_model_path, '{}_G.pth'.format(self.load_model))))
            self.D.load_state_dict(torch.load(os.path.join(self.args.load_model_path, '{}_D.pth'.format(self.load_model))))
            self.g_optimizer.load_state_dict(torch.load(os.path.join(self.args.load_model_path, '{}_g_optimizer.pth'.format(self.load_model))))
            self.d_optimizer.load_state_dict(torch.load(os.path.join(self.args.load_model_path, '{}_d_optimizer.pth'.format(self.load_model))))
            print('Loaded trained models (Epoch: {})..!'.format(self.load_model))

    
    def train(self):
        steps_per_epoch = len(self.train_loader)
        self.roi = Variable(self.roi.to(self.device))
        
        # Start with trained model
        if self.args.load_model:
            start = self.args.load_model
        else:
            start = 0
        
        # Fixed latent space input to validate during testing
        z_test = Variable(torch.randn(self.num_samples, self.latent_dim).to(self.device), requires_grad = False)
        #z_test_values = Variable(((self.max_value-self.min_value)*torch.rand((self.num_samples,1))+self.min_value).to(self.device), requires_grad = False)
        self.test_data.dataset["gcc_bin"] = pd.cut(self.test_data.dataset["gcc"], bins=self.args.gcc_bins) 
        sampled_test_df = self.test_data.dataset.groupby("gcc_bin").sample(n=int(self.num_samples/(len(self.args.gcc_bins)-1)), 
                                                                           replace=False, random_state=999).reset_index()

        #sampled_test_df = self.test_data.dataset.sample(self.num_samples, ignore_index=False).reset_index()
        z_test_values = Variable(torch.FloatTensor(sampled_test_df["gcc_rounded"].values).view(self.num_samples,1).to(self.device), requires_grad = False)
        
        # Save the test images for comparison
        fig, axs = plt.subplots(self.args.n_rows, self.args.n_cols, figsize=(22,16), squeeze=True)
        fig.suptitle("Sample Test Images", fontsize=26)
        axs = axs.flatten()
        for i in range(0, self.num_samples):
            test_img = tensor_to_PIL(denorm(self.test_data.__getitem__(sampled_test_df["index"][i])[0]))
            axs[i].imshow(test_img)
            axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            axs[i].set_title("GCC: %.4f ; RCC: %.4f"%(sampled_test_df["gcc"][i], sampled_test_df["rcc"][i]), fontsize=16)
        fig.tight_layout()
        plt.savefig(os.path.join(self.sample_path, "test_image_start_epoch_%d.png"%(start + 1)))
        plt.close() 
            
        start_time = time.time()
        for epoch in range(start, self.num_epoch): 
            data_itr = iter(self.train_loader)
            d_loss_real_history = []
            d_loss_fake_history = []
            g_loss_fake_history = []
            for step in range(0, steps_per_epoch):
                self.D.train()
                self.G.train()            
                self.reset_grad()
                # ================== Train D ================== #
                real_images, values, _ = next(data_itr)
                current_batch_size = real_images.size(0)
                # Compute loss with real images
                real_images = Variable(real_images.to(self.device))
                values = Variable(values.view(current_batch_size,1).type(torch.FloatTensor).to(self.device))
                d_out_real = self.D(real_images, self.roi.unsqueeze(0).repeat(current_batch_size,1,1,1), values)
                if self.args.adv_loss == 'HingeGAN':
                    d_loss_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
                
                z = Variable(torch.randn(current_batch_size, self.latent_dim).to(self.device))
                #fake_values = Variable(((self.max_value-self.min_value)*torch.rand((current_batch_size,1))+self.min_value).to(self.device))
                fake_images = self.G(z, self.roi.unsqueeze(0).repeat(current_batch_size,1,1,1), values)
                d_out_fake = self.D(fake_images.detach(), self.roi.unsqueeze(0).repeat(current_batch_size,1,1,1), values)
                
                if self.args.adv_loss == 'HingeGAN':
                    d_loss_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
                
                # Backward + Optimize
                d_loss_real_history.append(d_loss_real.item())
                d_loss_fake_history.append(d_loss_fake.item())
                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # ================== Train G ================== #
                # Create random noise
                z = Variable(torch.randn(current_batch_size, self.latent_dim).to(self.device))
                fake_values = Variable(((self.max_value-self.min_value)*torch.rand((current_batch_size,1))+self.min_value).to(self.device))
                fake_images = self.G(z, self.roi.unsqueeze(0).repeat(current_batch_size,1,1,1), fake_values)

                # Compute loss with fake images
                g_out_fake = self.D(fake_images, self.roi.unsqueeze(0).repeat(current_batch_size,1,1,1), fake_values)  # batch x n
                if self.args.adv_loss == 'HingeGAN':
                    g_loss_fake = - g_out_fake.mean()
                
                g_loss_fake_history.append(g_loss_fake.item())
                g_loss_fake.backward()
                self.g_optimizer.step()
            self.d_decay.step()
            self.g_decay.step()

            # Print out log info
            if (epoch + 1) % self.args.log_step == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))
                print("Elapsed [{}], Epoch [{}/{}], d_loss_real: {:.4f}, d_loss_fake: {:.4f}, g_loss_fake: {:.4f} ".
                      format(elapsed, epoch + 1, self.num_epoch,
                             np.mean(d_loss_real_history), np.mean(d_loss_fake_history), np.mean(g_loss_fake_history)))
                with open(os.path.join(self.log_path, self.exec_summary_file_name), 'a') as txt_file_object:
                    txt_file_object.write("\n>Elapsed [{}], Epoch [{}/{}], d_loss_real: {:.4f}, d_loss_fake: {:.4f}, g_loss_fake: {:.4f} ".
                      format(elapsed, epoch + 1, self.num_epoch,
                             np.mean(d_loss_real_history), np.mean(d_loss_fake_history), np.mean(g_loss_fake_history)))
                    txt_file_object.close()
        

            if (epoch+1) % self.args.model_save_step==0:
                torch.save(self.G.state_dict(), os.path.join(self.model_save_path, '{}_G.pth'.format(epoch + 1)))
                torch.save(self.D.state_dict(), os.path.join(self.model_save_path, '{}_D.pth'.format(epoch + 1)))
                torch.save(self.g_optimizer.state_dict(), os.path.join(self.model_save_path, '{}_g_optimizer.pth'.format(epoch + 1)))
                torch.save(self.d_optimizer.state_dict(), os.path.join(self.model_save_path, '{}_d_optimizer.pth'.format(epoch + 1)))



            # Sample images
            if (epoch + 1) % self.args.sample_step == 0:
                self.G.eval()
                fake_images = self.G(z_test, self.roi.unsqueeze(0).repeat(self.num_samples,1,1,1), z_test_values).detach()
                fig, axs = plt.subplots(self.args.n_rows, self.args.n_cols, figsize=self.args.plot_figsize, squeeze=True)
                fig.suptitle("After epoch %d"%(epoch + 1), fontsize=26)
                axs = axs.flatten()
                for i in range(0, self.num_samples):
                    fake_img = denorm(fake_images.data[i].to("cpu"))
                    gcc = calculate_gcc(fake_img, self.roi.to("cpu"))
                    rcc = calculate_rcc(fake_img, self.roi.to("cpu"))
                    test_img = np.array(tensor_to_PIL(denorm(self.test_data.__getitem__(sampled_test_df["index"][i])[0])))                
                    fake_img = tensor_to_PIL(fake_img)
                    ssim_score = calculate_ssim_score(test_img, np.array(fake_img))
                    axs[i].imshow(fake_img)
                    axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
                    axs[i].set_title("GCC (Input): %.4f; RCC: %.4f\nSSIM: %.2f"%(sampled_test_df["gcc"][i], sampled_test_df["rcc"][i], ssim_score), fontsize=16)
                    axs[i].text(0.5,-0.1, "Computed GCC: %.4f"%(gcc), size=16, ha="center", transform=axs[i].transAxes)
                    axs[i].text(0.5,-0.2, "Predicted RCC: %.4f"%(rcc), size=16, ha="center", transform=axs[i].transAxes)
                fig.tight_layout()
                plt.savefig(os.path.join(self.sample_path, '{}_synthetic_image.png'.format(epoch + 1)))
                plt.close() 
                
                    #save_image(denorm(fake_images.data),
                    #           os.path.join(self.sample_path, 'Epoch_{}_fake.png'.format(epoch + 1)))
    def reset_grad(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
    