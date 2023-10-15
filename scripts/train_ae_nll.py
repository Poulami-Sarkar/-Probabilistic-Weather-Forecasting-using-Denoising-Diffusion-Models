# Import libraries
import sys
import os
import math
from pathlib import Path

sys.path.insert(0,os.getcwd()+"/..")

from datasets.WeatherBenchData import WeatherBenchData

from torch.utils.data import DataLoader,BatchSampler,SequentialSampler
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss, MSELoss


from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers import UNet3DConditionModel, UNet2DModel, AutoencoderKL, VQModel
from diffusers.models.vae import DiagonalGaussianDistribution

import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import re

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import ASHAScheduler

import optuna
from optuna.trial import TrialState

# Change these
channels = 110
r = 7

# Define Model
vae = AutoencoderKL(in_channels= channels, out_channels=channels,latent_channels=64,down_block_types= ("DownEncoderBlock2D",),
        up_block_types = ("UpDecoderBlock2D",),
        block_out_channels= (1024,),
        )
#vae.load_state_dict(torch.load("/home/sarkar/Documents/code/weather_forecasting_using_denoising_diffusion_models/checkpoints/eval_mse_weighted_wider.pth")["state_dict"])

# Load Dataset
base_path = "/home/sarkar/Documents/WeatherBench_12/"

train_data = WeatherBenchData(Path(base_path),"train",subset=2016)
val_data = WeatherBenchData(Path(base_path),"val")
test_data = WeatherBenchData(Path(base_path),"test")
print(len(train_data),len(val_data),len(test_data))

# Set up dataloaders

batch_size = 64
train_dataloader = DataLoader(train_data, sampler=BatchSampler(
        SequentialSampler(train_data), batch_size=batch_size, drop_last=False
    ))

val_dataloader = DataLoader(val_data, sampler=BatchSampler(
        SequentialSampler(train_data), batch_size=batch_size, drop_last=False
    ))

test_dataloader = DataLoader(test_data, sampler=BatchSampler(
        SequentialSampler(test_data), batch_size=batch_size, drop_last=False
    ))


# Set up optimizers
kl_weight = 0.3

vae = vae.to("cuda")   
vae.train()
    
# Process tqdm bar
#batch_bar = tqdm(total=len(train_dataloader), leave=False, position=0, desc="Train")

opt = torch.optim.Adam(list(vae.encoder.parameters())+
    list(vae.decoder.parameters())+
    list(vae.quant_conv.parameters())+
    list(vae.post_quant_conv.parameters()),
    lr=0.001,betas=(0.6,0.999))
mse_loss = MSELoss()
scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.2)

epochs = 100

# Gaussian Likelihood
def gaussian_likelihood(x_hat, logscale, x):
    scale = torch.exp(logscale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1, 2, 3))


# Training code
run = "run"+str(r)
def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.sum(torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi),dim=[1,2,3])

for e in range(0,10):
    train_loss = 0.0
    count = 1 
    for i, batch in enumerate(train_dataloader):
        
        # load batch
        if channels == 1:
            _,x = batch
            x = x.reshape(x.shape[1], 1,  32,64)
        else:
            x,output = batch
            x = x.squeeze(0)
        
        x = x.to("cuda")
        #tgt = tgt.to("cuda")

        opt.zero_grad()

        # get reconstruct image
        x_hat = vae(x) 
        # MSE loss between original image and reconstructed one
        loss_mse = mse_loss(x,x_hat[0][0])
        
        # NLL loss
        nll_loss = gaussian_nll(x_hat[0][0], x_hat[2], x).sum()/(batch_size*110*32*64)
        
        #log_var = x_hat[1].logvar
        #rec_tensor = ((x.squeeze(0) - x_hat[0][0]) ** 2)
        #nll_loss = 0.5 * torch.mean(torch.sum(rec_tensor, dim=1)
        #                    + torch.sum(log_var, dim=1)
        #                    + torch.log(2 * torch.tensor(np.pi)))

        # KL divergence between encoder distrib. and N(0,1) distrib. 
        loss_kl = x_hat[1].kl().sum()/(batch_size*110*32*64)
        # Get total loss
        #loss = kl_weight * loss_mse +  (1 - kl_weight) * loss_kl #(loss_mse * (1 - config["kl_weight"]) + loss_kl * config["kl_weight"])
        loss  = kl_weight * loss_kl + (1-kl_weight) * nll_loss
        train_loss += loss.item()
        
        batch_loss = train_loss/count
        
        loss.backward()
        opt.step()

        #log_file.writelines
        print('[%d/%d][%d/%d] Loss KL: %.4f Loss NLL: %.4f Total Loss: %.4f'
                % (e, epochs, i,len(train_dataloader), loss_kl, 
                    nll_loss, batch_loss))

        count +=1
    scheduler.step()
    #if e%5 == 0:
    #    torch.save({'epoch': e, 'state_dict':vae.state_dict()},'../checkpoints/'+run+'/+str(e)+'.pth')

    """
    batch_bar.set_postfix(
        loss = f"{train_loss/(i+1):.4f}",
        mse_loss = f"{loss_mse:.4f}",
        kl_loss = f"{loss_kl:.4f}",
        lr = f"{optimizer.param_groups[0]['lr']:.4f}"
    )
    """

#batch_bar.close()
#train_loss /= len(dataloader)

torch.save({'epoch': e,
                    'state_dict':vae.state_dict()},
                    '/home/sarkar/Documents/code/weather_forecasting_using_denoising_diffusion_models/checkpoints/eval_mse_weighted_wider.pth' )

