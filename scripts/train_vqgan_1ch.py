import sys
import os
import math
from pathlib import Path

sys.path.insert(0,os.getcwd()+"/..")

from dataset.WeatherBenchData import WeatherBenchData

from torch.utils.data import DataLoader,BatchSampler,SequentialSampler
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import KLDivLoss, MSELoss, BCELoss


from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers import UNet3DConditionModel, UNet2DModel, AutoencoderKL, VQModel

# Imports for discriminator model 
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import re


base_path = "/home/sarkar/Documents/WeatherBench_12_unnorm/"

train_data = WeatherBenchData(Path(base_path),"train")
val_data = WeatherBenchData(Path(base_path),"val",subset=2010)
test_data = WeatherBenchData(Path(base_path),"test")


print(len(train_data),len(val_data),len(test_data))

batch_size = 128 * 2
train_dataloader = DataLoader(train_data, sampler=BatchSampler(
        SequentialSampler(train_data), batch_size=batch_size, drop_last=False
    ))

val_dataloader = DataLoader(val_data, sampler=BatchSampler(
        SequentialSampler(val_data), batch_size=batch_size, drop_last=False
    ))


test_dataloader = DataLoader(test_data, sampler=BatchSampler(
        SequentialSampler(test_data), batch_size=batch_size, drop_last=False
    ))




def calculate_adaptive_weight( nll_loss, g_loss, last_layer=None):
        discriminator_weight=1.0
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

# VQ-VAE Model for the generator
channels = 1
ndf =64
#vq_model = VQModel(in_channels= channels, out_channels=channels,latent_channels=4,vq_embed_dim=4)
vq_model = VQModel(in_channels= channels, out_channels=channels,latent_channels=4,vq_embed_dim=4,block_out_channels=(ndf,ndf))

# Dicriminator model from taming-transformer package
discriminator = NLayerDiscriminator(input_nc=channels,
                                                 n_layers=3,
                                                 use_actnorm=True, # Try with this set to false 
                                                 ndf=ndf
                                                 ).apply(weights_init)


"""file  = "../outputs/checkpoints/vqgan_v12_1ch_ep200.pth" 
vq_model.load_state_dict(torch.load(file)["state_dict"])
file  = "../outputs/checkpoints/vqgan_disc_v12_1ch_ep200.pth" 
discriminator.load_state_dict(torch.load(file)["state_dict"])"""
validation = True


vq_model = vq_model.to("cuda") 
discriminator = discriminator.to("cuda")



kl_weight = 0.7

mse_loss = MSELoss()
    
# Process tqdm bar
#batch_bar = tqdm(total=len(train_dataloader), leave=False, position=0, desc="Train")

opt_g = torch.optim.Adam(list(vq_model.encoder.parameters()) +
    list(vq_model.decoder.parameters()) +
    list(vq_model.quant_conv.parameters())+
    list(vq_model.post_quant_conv.parameters())+
    list(vq_model.quantize.parameters()), lr = 0.0001, betas=(0.8,0.999))

scheduler_g = torch.optim.lr_scheduler.MultiStepLR(opt_g, milestones=[50,100], gamma=0.1)
#torch.optim.lr_scheduler.StepLR(opt_g, step_size=10, gamma=0.2)

opt_d = torch.optim.Adam(discriminator.parameters(), lr = 0.0001, betas=(0.9,0.999))

scheduler_d = torch.optim.lr_scheduler.MultiStepLR(opt_d, milestones=[50,100], gamma=0.1)
#torch.optim.lr_scheduler.StepLR(opt_d, step_size=10, gamma=0.2)

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight



epochs = 200
global_step = 0 
d_factor =1.0
discriminator_iter_start = 1.0

rec_mode ="L2"
best_val_loss = 9999

for e in range(epochs):

    train_gloss = 0.0
    train_dloss =0.0
    count = 1 
    for i, batch in enumerate(train_dataloader):
        
        # load batch
        if channels == 1:
            _,x = batch
            x = x["data"]
            x = x.reshape(x.shape[1], 1,  32,64)
        else:
            x,output = batch
            x = x["data"]
            x = x.squeeze(0)
        x = x.to("cuda")
        

        vq_model.train()
        discriminator.train()

        #########################################
        #   Discriminator Loss
        #########################################

        opt_d.zero_grad()
        
        x_hat = vq_model(x) 
        
        # Update disciminator weights
        logits_real = discriminator(x.squeeze(0).contiguous())        
        logits_fake = discriminator(x_hat[0][0].contiguous().detach())

        d_loss = hinge_d_loss(logits_real, logits_fake)
        d_loss.backward()

        train_dloss += d_loss.item()
        batch_dloss = train_dloss/count 

        opt_d.step()                


        #########################################
        #   Genetator Loss
        #########################################
        
        opt_g.zero_grad()

        # Emnbedding Loss
        quant_loss = x_hat[2].mean()

        # Reconstruction Loss
        if rec_mode == "L1":
            rec_tensor = torch.abs((x.squeeze(0) - x_hat[0][0]))
        else:
            rec_tensor = ((x.squeeze(0) - x_hat[0][0]) ** 2)
        rec_loss = torch.mean(rec_tensor)

        # Generator loss
        logits_fake = discriminator(x_hat[0][0].contiguous()) 
        # Comes from the disciminator model
        gen_loss = -torch.mean(logits_fake)#torch.mean(F.relu(1.0 - logits_fake))

        last_layer = vq_model.decoder.conv_out.weight
        weight = calculate_adaptive_weight(rec_loss,gen_loss,last_layer)
        d_factor = adopt_weight(1.0, global_step, threshold=5000, value=0.) # Try with also 10000. Print the weight values in the output

        #weighted_gen = max(0,gen_loss * weight * d_factor) # For v5 and v7
        weighted_gen = d_factor*gen_loss * weight # For v6 v8

        # Generator loss
        g_loss = rec_loss + quant_loss + weighted_gen

        train_gloss += g_loss.item()
        batch_gloss = train_gloss/count
        
        g_loss.backward()
        opt_g.step()
        global_step+=1


        print('[%d/%d][%d/%d] Loss G: %f Loss D: %f Total GLoss: %f Total DLoss: %f  Rec Loss: %f Quant Loss: %f Gen Loss: %f' 
                % (e, epochs, i,len(train_dataloader), g_loss, 
                    d_loss, batch_gloss,batch_dloss, rec_loss,quant_loss, weighted_gen))
                
        count +=1

    scheduler_g.step()
    scheduler_d.step()

    ###############
    #   Validation
    ###############
    print("Starting validation")
    if validation:
        vq_model.eval()
        tot_val_loss = 0 
        for i, batch in enumerate(val_dataloader):
    
            # load batch
            if channels == 1:
                _,x = batch
                x = x["data"]
                x = x.reshape(x.shape[1], 1,  32,64)
            else:
                x,output = batch
                x = x["data"]
                x = x.squeeze(0)
            x = x.to("cuda")
            
            x = x.to("cuda")

            with torch.no_grad():
                # get reconstruct image
                x_hat = vq_model(x) 

                # Emnbedding Loss
                quant_loss = x_hat[2].mean()

                # Reconstruction Loss
                rec_tensor = ((x.squeeze(0) - x_hat[0][0]) ** 2)
                rec_loss = torch.mean(rec_tensor)

                val_loss = rec_loss + quant_loss #+ disc_factor * gen_loss
                #print(rec_loss,quant_loss,gen_loss)

                tot_val_loss += val_loss
                
        tot_val_loss /= len(val_dataloader)   
        print("Validation score:", tot_val_loss) 

        if best_val_loss >= tot_val_loss:
            best_val_loss = tot_val_loss
            print("Best validation score:", best_val_loss)
            torch.save({'epoch': e,
                    'state_dict':vq_model.state_dict()},
                    "../outputs/checkpoints/vqgan_v14_1ch_best_ep"+str(e)+".pth")
            torch.save({'epoch': e,
                    'state_dict':discriminator.state_dict()},
                    "../outputs/checkpoints/vqgan_disc_v14_best_1ch_ep"+str(e)+".pth")   
            
    else: 
        if (e+1) % 50 == 0:
            torch.save({'epoch': e,
                        'state_dict':vq_model.state_dict()},
                        "../outputs/checkpoints/vqgan_v14_1ch_ep"+str(e)+".pth")

            torch.save({'epoch': e,
                    'state_dict':discriminator.state_dict()},
                    "../outputs/checkpoints/vqgan_disc_v14_1ch_ep"+str(e)+".pth")        


# Save final model 
torch.save({'epoch': e,
                    'state_dict':vq_model.state_dict()},
                    "../outputs/checkpoints/vqgan_v14_1ch_ep200.pth")

torch.save({'epoch': e,
                    'state_dict':discriminator.state_dict()},
                    "../outputs/checkpoints/vqgan_disc_v14_1ch_ep200.pth")