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

# Imports for discriminator model 
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import re


base_path = "/home/sarkar/Documents/WeatherBench_12/"

train_data = WeatherBenchData(Path(base_path),"train",subset=2016)
test_data = WeatherBenchData(Path(base_path),"test")
print(len(train_data))


batch_size = 128
train_dataloader = DataLoader(train_data, sampler=BatchSampler(
        SequentialSampler(train_data), batch_size=batch_size, drop_last=False
    ))
test_dataloader = DataLoader(test_data, sampler=BatchSampler(
        SequentialSampler(test_data), batch_size=batch_size, drop_last=False
    ))


# VQ-VAE Model for the generator
channels = 110
vq_model = VQModel(in_channels= channels, out_channels=channels,latent_channels=4)
vq_model.load_state_dict(torch.load("../outputs/checkpoints/vq_c110_ep49.pth")["state_dict"])

# Dicriminator model from taming-transformer package
discriminator = NLayerDiscriminator(input_nc=channels,
    n_layers=3,
    use_actnorm=True, # Try with this set to false 
    ndf=64
    ).apply(weights_init)


kl_weight = 0.7

mse_loss = MSELoss()


vq_model = vq_model.to("cuda") 
discriminator = discriminator.to("cuda")

    
# Process tqdm bar
#batch_bar = tqdm(total=len(train_dataloader), leave=False, position=0, desc="Train")

opt_g = torch.optim.Adam(list(vq_model.encoder.parameters()) +
    list(vq_model.decoder.parameters()) +
    list(vq_model.quant_conv.parameters())+
    list(vq_model.post_quant_conv.parameters())+
    list(vq_model.quantize.parameters()), lr = 0.001, betas=(0.6,0.98))

scheduler_g = torch.optim.lr_scheduler.StepLR(opt_g, step_size=30, gamma=0.2)

opt_d = torch.optim.Adam(discriminator.parameters(), lr = 0.0001, betas=(0.6,0.9))

scheduler_d = torch.optim.lr_scheduler.StepLR(opt_d, step_size=30, gamma=0.2)

def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight

epochs = 100
global_step = 0 
disc_factor =1.0
discriminator_iter_start = disc_factor

for e in range(50,epochs):

    train_gloss = 0.0
    train_dloss =0.0
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
        

        vq_model.train()
        discriminator.train()

        opt_g.zero_grad()
        opt_d.zero_grad()


        #########################################
        #   Genetator Loss
        #########################################

        x_hat = vq_model(x) 
        posterior = x_hat[1]
        log_var = posterior.logvar

        # Emnbedding Loss
        quant_loss = x_hat[2].mean()

        # Reconstruction Loss
        rec_tensor = ((x.squeeze(0) - x_hat[0][0]) ** 2)
        rec_loss = torch.mean(rec_tensor)

        # NLL loss
        nll_loss = 0.5 * torch.mean(torch.sum(rec_tensor, dim=1)
                                    + torch.sum(log_var, dim=1)
                                    + torch.log(2 * torch.tensor(np.pi)))

        #  kl divergence
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # Generator loss
        logits_fake = discriminator(x_hat[0][0].contiguous()) # Comes from the disciminator model
        gen_loss = -torch.mean(logits_fake)
        disc_factor = adopt_weight(disc_factor, global_step, threshold=discriminator_iter_start)

        g_loss = rec_loss + quant_loss #+ disc_factor * gen_loss
        #print(rec_loss,quant_loss,gen_loss)

        train_gloss += g_loss.item()
        batch_gloss = train_gloss/count
        
        g_loss.backward()
        opt_g.step()
        global_step+=1
        #########################################
        #   Discriminator Loss
        #########################################

        # Update disciminator weights
        logits_real = discriminator(x.squeeze(0).contiguous())
        logits_fake = discriminator(x_hat[0][0].contiguous())
        d_loss = hinge_d_loss(logits_real, logits_fake)

        train_dloss += d_loss.item()
        batch_dloss = train_dloss/count

        #d_loss.backward(retain_graph=True)
        #opt_d.step()

        

        print('[%d/%d][%d/%d] Loss G: %.4f Loss D: %.4f Total GLoss: %.4f Total DLoss: %.4f' 
                % (e, epochs, i,len(train_dataloader), g_loss, 
                    d_loss, batch_gloss,batch_dloss))

        count +=1

    scheduler_g.step()
    scheduler_d.step()

    """
    batch_bar.set_postfix(
        loss = f"{train_loss/(i+1):.4f}",
        mse_loss = f"{loss_mse:.4f}",
        kl_loss = f"{loss_kl:.4f}",
        lr = f"{optimizer.param_groups[0]['lr']:.4f}"
    )
    """
torch.save({'epoch': e,
                    'state_dict':vq_model.state_dict()},
                    '../outputs/checkpoints/vq_c110_ep'+str(e)+'.pth' )
#batch_bar.close()
#train_loss /= len(dataloader)

