import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import numpy as np
import sys
import tqdm
import os

import math
import matplotlib.pyplot as plt
from .networks import skip, fcn
#from .networks_Fconv import skip, fcn
from .SSIM import SSIM
import scipy.io as sio
import xlwt
from torch import optim
import torch.nn.utils as nutils
from scipy.signal import convolve2d

sys.path.append('../')
from .util import evaluation_image, get_noise, move2cpu, calculate_psnr, save_final_kernel_png, tensor2im01, calculate_parameters
from .kernel_generate import gen_kernel_random, gen_kernel_random_motion, make_gradient_filter, ekp_kernel_generator

sys.path.append('../../')

from torch.utils.tensorboard import SummaryWriter

'''
# ------------------------------------------
# models of DIPDKP, etc.
# ------------------------------------------
'''



class VAEKernelEstimator(nn.Module):
    def __init__(self, kernel_size):
        super(VAEKernelEstimator, self).__init__()
        self.kernel_size = kernel_size
        self.fc1 = nn.Linear(kernel_size * kernel_size, 128)
        self.fc2_mu = nn.Linear(128, kernel_size * kernel_size)
        self.fc2_logvar = nn.Linear(128, kernel_size * kernel_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc2_mu(h1), self.fc2_logvar(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class DIPDKP:
    def __init__(self, conf, lr, hr, device=torch.device('cuda')):
        self.conf = conf
        self.lr = lr
        self.sf = conf.sf
        self.hr = hr
        self.kernel_size = min(conf.sf * 4 + 3, 21)

        _, C, H, W = self.lr.size()
        self.input_dip = get_noise(C, 'noise', (H * self.sf, W * self.sf)).to(device).detach()
        self.lr_scaled = F.interpolate(self.lr, size=[H * self.sf, W * self.sf], mode='bicubic', align_corners=False)
        self.input_dip.requires_grad = False
        self.net_dip = skip(C, 3,
                            num_channels_down=[128, 128, 128, 128, 128],
                            num_channels_up=[128, 128, 128, 128, 128],
                            num_channels_skip=[16, 16, 16, 16, 16],
                            upsample_mode='bilinear',
                            need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')
        self.net_dip = self.net_dip.to(device)
        self.optimizer_dip = torch.optim.Adam([{'params': self.net_dip.parameters()}], lr=conf.dip_lr)

        self.kernel_estimator = VAEKernelEstimator(self.kernel_size).to(device)
        self.optimizer_kernel = torch.optim.Adam(self.kernel_estimator.parameters(), lr=5e-3)

        self.ssimloss = SSIM().to(device)
        self.mse = torch.nn.MSELoss().to(device)

    def estimate_kernel(self, x):
        return self.kernel_estimator(x)

    def train(self):
        try:
            self.print_and_output_setting()
            _, C, H, W = self.lr.size()
            path = os.path.join(self.conf.input_dir, self.conf.filename).replace('lr_x', 'gt_k_x').replace('.png', '.mat')
            if not self.conf.real:
                kernel_gt = sio.loadmat(path)['Kernel']
            else:
                kernel_gt = np.zeros([self.kernel_size, self.kernel_size])

            self.MC_warm_up()

            for self.iteration in tqdm.tqdm(range(self.conf.max_iters), ncols=60):
                if self.conf.model == 'DIPDKP':
                    self.kernel_code.requires_grad = False
                    self.optimizer_kp.zero_grad()
                    sr = self.net_dip(self.input_dip)
                    sr_pad = F.pad(sr, mode='circular', pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2))
                    k_losses = torch.zeros(self.conf.D_loop)
                    k_loss_probability = torch.zeros(self.conf.D_loop)
                    k_loss_weights = torch.zeros(self.conf.D_loop)
                    x_losses = torch.zeros(self.conf.D_loop)

                    for k_p in range(self.conf.D_loop):
                        kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                        self.MCMC_sampling()
                        k_losses[k_p] = self.mse(self.kernel_random, kernel)
                        out_x = F.conv2d(sr_pad, self.kernel_random.expand(3, -1, -1, -1).clone().detach(), groups=3)
                        out_x = out_x[:, :, 0::self.sf, 0::self.sf]
                        x_losses[k_p] = self.mse(out_x, self.lr)

                    sum_exp_x_losses = 1e-5
                    lossk = 0
                    for i in range(self.conf.D_loop):
                        sum_exp_x_losses += (x_losses[i] - min(x_losses))

                    for i in range(self.conf.D_loop):
                        k_loss_probability[i] = (x_losses[i] - min(x_losses)) / sum_exp_x_losses
                        k_loss_weights[i] = (-(1 - k_loss_probability[i])**2) * torch.log(k_loss_probability[i] + 1e-3)
                        lossk += k_loss_weights[i].clone().detach() * k_losses[i]

                    if self.conf.D_loop != 0:
                        lossk.backward(retain_graph=True)
                        lossk.detach()
                        self.optimizer_kp.step()

                    ac_loss_k = 0
                    for i_p in range(self.conf.I_loop_x):
                        self.optimizer_dip.zero_grad()
                        self.optimizer_kp.zero_grad()
                        kernel = self.net_kp(self.kernel_code).view(1, 1, self.kernel_size, self.kernel_size)
                        sr = self.net_dip(self.input_dip)
                        sr_pad = F.pad(sr, mode='circular', pad=(self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2, self.kernel_size // 2))
                        out_x = F.conv2d(sr_pad, kernel.expand(3, -1, -1, -1).clone().detach(), groups=3)
                        out_x = out_x[:, :, 0::self.sf, 0::self.sf]
                        disturb = np.random.normal(0, np.random.uniform(0, self.conf.Image_disturbance), out_x.shape)
                        disturb_tc = torch.from_numpy(disturb).type(torch.FloatTensor).to(torch.device('cuda'))

                        if self.iteration <= 80:
                            loss_x = 1 - self.ssimloss(out_x, self.lr + disturb_tc)
                        else:
                            loss_x = self.mse(out_x, self.lr + disturb_tc)

                        self.im_HR_est = sr
                        grad_loss = self.conf.grad_loss_lr * self.noise2_mean * 0.20 * torch.pow(self.calculate_grad_abs() + 1e-8, 0.67).sum() / self.num_pixels
                        loss_x_update = loss_x + grad_loss
                        loss_x_update.backward(retain_graph=True)
                        loss_x_update.detach()
                        self.optimizer_dip.step()

                        out_k = F.conv2d(sr_pad.clone().detach(), kernel.expand(3, -1, -1, -1), groups=3)
                        out_k = out_k[:, :, 0::self.sf, 0::self.sf]

                        if self.iteration <= 80:
                            loss_k = 1 - self.ssimloss(out_k, self.lr)
                        else:
                            loss_k = self.mse(out_k, self.lr)

                        ac_loss_k = ac_loss_k + loss_k

                        if (self.iteration * self.conf.I_loop_x + i_p + 1) % (self.conf.I_loop_k) == 0:
                            ac_loss_k.backward(retain_graph=True)
                            self.optimizer_kp.step()
                            ac_loss_k = 0

                        if (((self.iteration * self.conf.I_loop_x + i_p) + 1) % self.conf.Print_iteration == 0 or
                                ((self.iteration * self.conf.I_loop_x + i_p) + 1) == 1):
                            self.print_and_output(sr, kernel, kernel_gt, loss_x, i_p)

            kernel = move2cpu(kernel.squeeze())
            save_final_kernel_png(kernel, self.conf, self.conf.kernel_gt)

            if self.conf.verbose:
                print('{} estimation complete! (see --{}-- folder)\n'.format(self.conf.model, self.conf.output_dir_path) + '*' * 60 + '\n\n')

            return kernel, sr
        except Exception as e:
            print(f"An error occurred: {e}")
            return None, None