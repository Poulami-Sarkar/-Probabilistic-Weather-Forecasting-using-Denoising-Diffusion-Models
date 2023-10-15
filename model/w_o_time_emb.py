# %%
from typing import Any, Dict, List, Optional, Tuple, Union




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

#from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from diffusers import UNet3DConditionModel, UNet2DConditionModel, AutoencoderKL, VQModel,UNet2DModel
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from diffusers.models.embeddings import TimestepEmbedding 

import sys
import os
import math
from pathlib import Path

sys.path.insert(0,os.getcwd()+"/..")

from torch.utils.data import DataLoader,BatchSampler,SequentialSampler,RandomSampler
import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss, MSELoss
import torch.nn as nn


from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
from diffusers import UNet3DConditionModel, UNet2DModel, AutoencoderKL, VQModel,UNet2DConditionModel
from diffusers.models.embeddings import TimestepEmbedding 
from diffusers.models.unet_2d_condition import UNet2DConditionOutput


# Imports for discriminator model 
from taming.modules.discriminator.model import NLayerDiscriminator, weights_init
from taming.modules.losses.vqperceptual import hinge_d_loss, vanilla_d_loss

from collections import namedtuple

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tqdm
import re
from ldm.models.diffusion.ddim import DDIMSampler

from ldm.modules.diffusionmodules.openaimodel import UNetModel
import properscoring as ps

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from datetime import datetime
from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like

__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device="cuda") + r2

class DDPM(nn.Module):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config=None,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 cond_model_pth="../",
                 first_stage_model_pth="../",
                 channels=110,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 learning_rate=1e-4,
                 learn_logvar=False,
                 logvar_init=0.,
                 v_posterior=0.,
                 conditioning_key = "crossattn",
                 device="cuda",
                 latent_ch=4
    ):
        super().__init__()
        
        # Initialize parameters
        self.channels = channels
        self.latent_ch=latent_ch
        self.device = device
        self.num_timesteps = int(timesteps)
        self.learning_rate = learning_rate
        self.v_posterior = v_posterior
        self.register_schedule()
        self.cond_model_pth = cond_model_pth
        self.first_stage_model_pth = first_stage_model_pth


        # U-Net model that performs the denoising 
        self.model = DiffusionWrapper(latent_ch,unet_config, conditioning_key)
        #self.model = self.model.to("cuda")

        # Model that encodes the latents of the 1channel geo-potentials and 110 channel conditionings
        self.initialise_models()

        
        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)
    

    def forward(self, x_noisy,cond,t):

        # Apply model
        x_recon = self.apply_model(x_noisy, t, cond)

        return x_recon

    def apply_model(self,x_noisy, t, cond):
        # Apply model
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}
            
        x_recon = self.model(x_noisy, t, **cond)

        return x_recon.sample

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt
    

    def initialise_models(self,):
        self.cond_stage_model = VQModel(in_channels= self.channels, out_channels=self.channels,latent_channels=64)
        self.first_stage_model = VQModel(in_channels= 1, out_channels=1,latent_channels=4,vq_embed_dim=4,block_out_channels=(64,64))
        #load weights
        self.first_stage_model.load_state_dict(torch.load(self.first_stage_model_pth)["state_dict"])
        self.cond_stage_model.load_state_dict(torch.load(self.cond_model_pth)["state_dict"])
        
        self.first_stage_model = self.first_stage_model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

        self.cond_stage_model = self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if given_betas != None :
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()


    def q_sample(self, x_start, t, noise):
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

class DiffusionWrapper(nn.Module):
    def __init__(self, latent_ch,diff_model_config=None, conditioning_key=None):
        super().__init__()
        self.latent_ch = latent_ch
        if diff_model_config:
            self.diffusion_model = self.instantiate_from_config(diff_model_config)
        else:
            self.diffusion_model = UNet2DConditionModel(
                in_channels=self.latent_ch,
                out_channels=self.latent_ch,
                down_block_types = (
                    "CrossAttnDownBlock2D",
                    #"CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types = ("UpBlock2D", "CrossAttnUpBlock2D", "CrossAttnUpBlock2D"),
                block_out_channels = (320, 640, 640),
                #encoder_hid_dim_type="image_proj",
                #encoder_hid_dim= 32*64,
                cross_attention_dim = 32*64,
                )
        self.conditioning_key = conditioning_key
        self.diffusion_model = self.diffusion_model.to("cuda")
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def instantiate_from_config(self,diff_model_config):
        model = None
        #TODO: Instantiate the U-Net model
        return model

    def forward(self, x, t,c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, cc) #,context=cc
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out
    
