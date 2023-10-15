import sys
import os
import math
from pathlib import Path

sys.path.insert(0,os.getcwd()+"/..")

from dataset.WeatherBenchData import WeatherBenchData

from torch.utils.data import DataLoader,BatchSampler,SequentialSampler
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss, MSELoss


from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline

# Imports for discriminator model 
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss

from collections import namedtuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import re

#from ldm.models.diffusion.ddpm_weather import DDPM
from ldm.modules.diffusionmodules.openaimodel import UNetModel
from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from diffusers import UNet3DConditionModel, UNet2DConditionModel, AutoencoderKL, VQModel,UNet2DModel
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from model.w_o_time_emb import DDPM

ver = 'diffusion_v2'
validation = True
# Path to dataset
base_path = "/media/sarkar/81fbe66f-ecc3-4c4b-857f-74f158bfdc61/WeatherBench_12_unnorm/"
# Path to conditioning model (110 channel ae)
cond_path = "/home/sarkar/Documents/code/weather_forecasting_using_denoising_diffusion_models/outputs/checkpoints/vqgan_v7_110ch_ep380.pth"
# Path to target encoder model (1 channel ae)
tgt_path = "/home/sarkar/Documents/code/weather_forecasting_using_denoising_diffusion_models/outputs/checkpoints/vqgan_v14_1ch_best_ep126.pth"


# Load dataset
train_data = WeatherBenchData(Path(base_path),"train")
test_data = WeatherBenchData(Path(base_path),"test")
val_data = WeatherBenchData(Path(base_path),"val")

print(len(train_data),len(test_data))

# %%

# Data Looaders
batch_size = 28
train_dataloader = DataLoader(train_data, sampler=BatchSampler(
        SequentialSampler(train_data), batch_size=batch_size, drop_last=False
    ))

test_dataloader = DataLoader(test_data, sampler=BatchSampler(
        SequentialSampler(test_data), batch_size=batch_size, drop_last=False
    ))
val_dataloader = DataLoader(val_data, sampler=BatchSampler(
        SequentialSampler(val_data), batch_size=batch_size, drop_last=False
    ))

# Define the Diffusion model
diffusion = DDPM( 
    cond_model_pth=cond_path,
    first_stage_model_pth= tgt_path
)


diffusion.load_state_dict(torch.load("../checkpoints/diffusion_v2.pth")["state_dict"])


total_params = sum(p.numel() for p in diffusion.parameters())
print(f"Number of parameters: {total_params}")


lr = 1e-4
params = list(diffusion.model.parameters())
params = params #+ [self.logvar]
opt = torch.optim.AdamW(params, lr=lr)
mse_loss = MSELoss()
epochs = 75
best_val_loss =999999

for e in range(50,epochs):
    train_loss = 0.0
    count = 1 
    total_loss = 0
    for i, batch in enumerate(train_dataloader):
        
        opt.zero_grad()
        
        cond,tgt = batch
        tgt = tgt.reshape(tgt.shape[1], 1,  32,64)
        cond = cond.squeeze(0)

        # Compute latents on the 1ch imputs
        latents = diffusion.first_stage_model.encode(tgt).latents

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        t = torch.randint(0, diffusion.num_timesteps, (bsz,), device="cpu").long()
        tgt_noisy = diffusion.q_sample(x_start=latents, t=t, noise=noise)

        # Compute latents on the conditioning data
        cond = diffusion.cond_stage_model.encode(cond).latents
        cond = cond.view(cond.size(0), 64, -1)

        target = noise

        tgt_noisy = tgt_noisy.to("cuda")
        cond = cond.to("cuda")
        t = t.to("cuda")
        target = target.to("cuda")
        # Return prediction
        pred_noise = diffusion(tgt_noisy,cond,t)
        # Compute loss
        loss = mse_loss(target, pred_noise)

        loss.backward()
        opt.step()

        train_loss = loss.item()
        total_loss += train_loss

        print('[%d/%d][%d/%d] Loss: %.4f Total Loss: %.4f '
                % (e, epochs, i,len(train_dataloader), train_loss, 
                    total_loss/(i+1)))
        
    if e%5 == 0:
        torch.save({'epoch': e,
                'state_dict':diffusion.state_dict()},
                '../checkpoints/'+ver+'_'+str(e)+'.pth')

    ###############
    #   Validation
    ###############
    print("Starting validation")
    if validation:
        diffusion.eval()
        tot_val_loss = 0 
        for j, batch in enumerate(val_dataloader):
    
            opt.zero_grad()
        
            cond_,tgt_ = batch
            hour = tgt_["hour"].squeeze(0)* 2 * math.pi / 24
            day = (tgt_["day"].squeeze(0) - 1 )* 2 * math.pi / 365

            tgt = tgt_["data"]
            cond = cond_["data"]
            tgt = tgt.reshape(tgt.shape[1], 1,  32,64)
            cond = cond.squeeze(0)
            with torch.no_grad():
                # Compute latents on the 1ch imputs
                latents = diffusion.first_stage_model.encode(tgt).latents

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                t = torch.randint(0, diffusion.num_timesteps, (bsz,), device="cpu").long()
                tgt_noisy = diffusion.q_sample(x_start=latents, t=t, noise=noise)

                # Compute latents on the conditioning data
                cond = diffusion.cond_stage_model.encode(cond).latents
                cond = cond.view(cond.size(0), 64, -1)

                target = noise

                tgt_noisy = tgt_noisy.to("cuda")
                cond = cond.to("cuda")
                t = t.to("cuda")
                target = target.to("cuda")
                day = day.to("cuda")
                hour = hour.to("cuda")
                # Return prediction
                pred_noise = diffusion(tgt_noisy,t,day,hour,cond)
                
                # Compute loss
                loss = mse_loss(target, pred_noise)
                tot_val_loss += loss
                    
        tot_val_loss /= len(val_dataloader)   
        print("Validation score:", tot_val_loss) 

        if best_val_loss >= tot_val_loss:
            best_val_loss = tot_val_loss
            print("Best validation score:", best_val_loss)
            torch.save({'epoch': e,
                    'state_dict':diffusion.state_dict()},
                    "../outputs/checkpoints/"+ver+"_best_ep"+str(e)+".pth")

    

    if e%1 == 0:
        torch.save({'epoch': e,
                'state_dict':diffusion.state_dict()},
                '../checkpoints/'+ver+'_'+str(e)+'.pth')

torch.save({'epoch': e,
    'state_dict':diffusion.state_dict()},
    '../checkpoints/'+ver+'.pth')

torch.save({'epoch': e,
    'state_dict':diffusion.state_dict()},
    '../checkpoints/new_'+ver+'.pth')
