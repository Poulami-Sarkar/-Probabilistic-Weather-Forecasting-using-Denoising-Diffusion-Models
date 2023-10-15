# Imports
# TODO: Add validation loss to this. 

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


############################
# Set parameters
############################
channels = 1

############################
# Load Dataset
############################
base_path = "/home/sarkar/Documents/WeatherBench_12/"

train_data = WeatherBenchData(Path(base_path),"train",subset=2016)
val_data = WeatherBenchData(Path(base_path),"val")
test_data = WeatherBenchData(Path(base_path),"test")
print(len(train_data),len(val_data),len(test_data))

# Set to maximum batch size accepted by the GPU
batch_size = 128

# Initialize dataloaders
train_dataloader = DataLoader(train_data, sampler=BatchSampler(
        SequentialSampler(train_data), batch_size=batch_size, drop_last=False
    ))

val_dataloader = DataLoader(val_data, sampler=BatchSampler(
        SequentialSampler(train_data), batch_size=batch_size, drop_last=False
    ))

test_dataloader = DataLoader(test_data, sampler=BatchSampler(
        SequentialSampler(test_data), batch_size=batch_size, drop_last=False
    ))


############################
# Initialize optimization 
############################

r = 1
log_file = open("optimization_v6.log","w")
def objective(trial):
    global r

    # Define Model
    latent_channels = 4 
    vae = AutoencoderKL(in_channels= channels, out_channels=channels,latent_channels=latent_channels)
    vae = vae.to("cuda")
    # Set up Optimizer
    lr = 0.004671610280498531 #trial.suggest_float("lr", 1e-5, 0.5e-3, log=True)
    step_size = 30 #trial.suggest_int("step_size", 30, 40, log=True)
    beta1  = 0.7186239401027583 #trial.suggest_float("beta1", 0.5, 0.9, log=True)
    beta2 = 0.835640610388541 #trial.suggest_float("beta2", 0.8, 1.0, log=True)
    kl_weight = trial.suggest_float("kl_weight", 0.1, 0.5, log=True)
    #latent_channels = trial.suggest_int("latent_channels", 2, 16, log=True)
    # Rest of your code
    
    opt = torch.optim.Adam(list(vae.encoder.parameters())+
        list(vae.decoder.parameters())+
        list(vae.quant_conv.parameters())+
        list(vae.post_quant_conv.parameters()),
        lr=lr,betas=(beta1,beta2))
    
    log_file.writelines("\nNew iteration: Learning rate: " +str(lr)+" step_size: "+str(step_size)+" beta1: "+str(beta1)+" beta2: "+ str(beta2))
    
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=step_size, gamma=0.2)

    mse_loss = MSELoss()

    loss = 0 
    epoch =50

    # Train Loop
    #run = "run_"+str(r)
    #os.mkdir("checkpoints/"+run)
    #log_file = open("checkpoints/"+run+"/train_log.log","w")


    for e in range(0,epoch):
        train_loss = 0.0
        count = 1 

        vae.train()
        for i, batch in enumerate(train_dataloader):
            
            if channels == 1:
                _,x = batch
                x = x.reshape(x.shape[1], 1,  32,64)
            else:
                x,tgt = batch
                x = x.squeeze(0)

            x = x.to("cuda")
            
            opt.zero_grad()

            # Get reconstruct image
            x_hat = vae(x) 
            
            # NLL Loss
            log_var = x_hat[1].logvar
            rec_tensor = ((x.squeeze(0) - x_hat[0][0]) ** 2)
            nll_loss = 0.5 * torch.mean(torch.sum(rec_tensor, dim=1)
                                + torch.sum(log_var, dim=1)
                                + torch.log(2 * torch.tensor(np.pi)))

            # KL divergence between encoder distrib. and N(0,1) distrib. 
            loss_kl = x_hat[1].kl().sum()/batch_size
            
            # MSE loss
            mse =  mse_loss(x,x_hat[0][0])
            # Get total loss
            loss  = loss_kl * kl_weight + mse * (1- kl_weight)#nll_loss
            train_loss += loss.item()
            batch_loss = train_loss/count
            
            count +=1
            loss.backward()
            opt.step()

            log_file.writelines('\n[%d/%d][%d/%d] Loss KL: %.4f Loss NLL: %.4f Total Loss: %.4f'
                    % (e, epoch, i,len(train_dataloader), loss_kl, 
                        mse, batch_loss))

        validation_loss = batch_loss
        # Start Validation  
        #if (e % 10) == 0:
        """val_loss = 0
        vae.eval()
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader): 
                x,tgt = batch
                x = x.squeeze(0)
                x = x.to("cuda")
                tgt = tgt.to("cuda")
                x_hat = vae(x) 
                # NLL Loss
                log_var = x_hat[1].logvar
                rec_tensor = ((x.squeeze(0) - x_hat[0][0]) ** 2)
                nll_loss = 0.5 * torch.mean(torch.sum(rec_tensor, dim=1)
                                    + torch.sum(log_var, dim=1)
                                    + torch.log(2 * torch.tensor(np.pi)))

                # KL divergence between encoder distrib. and N(0,1) distrib. 
                loss_kl = x_hat[1].kl().sum()/batch_size
                
                # Get total loss
                loss  = loss_kl + nll_loss
                val_loss += loss.item()
                validation_loss = val_loss/count
            log_file.writelines('[%d/%d][%d/%d] Loss KL: %.4f Loss NLL: %.4f Total Loss: %.4f'
                    % (e, epochs, i,len(val_dataloader), loss_kl, 
                        nll_loss, validation_loss))"""

            
        trial.report(validation_loss,e)
        scheduler.step()
        
    
    return validation_loss   
    


############################
# Start hyperParameter search
############################

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)