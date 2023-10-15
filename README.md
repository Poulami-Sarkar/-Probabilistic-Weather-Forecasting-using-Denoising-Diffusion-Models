# -Probabilistic-Weather-Forecasting-using-Denoising-Diffusion-Models

This 

# Dataset

The dataset can be downloadedd from [Weatherbench](https://dataserv.ub.tum.de/index.php/s/m1524895?path=%2F5.625deg)

# Training 

## Autoencoders

### Variational Autoencoder 
`python train_ae_nll.py`<br/ >


### VQ- Autoencoder 
`python train_vq.py` <br/ >

### VQ-GAN  
For 1 channel geopotential 500 <br/ >
`python train_vqgan_1ch.py` <br/ >
For 1 channel 110-channel conditioning <br/ >
`python train_vqgan_110ch.py` <br/ >


## Diffusion model
