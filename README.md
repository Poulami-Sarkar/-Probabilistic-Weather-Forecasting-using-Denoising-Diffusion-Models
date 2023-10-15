# -Probabilistic-Weather-Forecasting-using-Denoising-Diffusion-Models


We test the applicability of diffusion models for generating probabilistic weather forecasts. The study builds upon existing literature on diffusion models, which generally focuses on conditional and unconditional image generation and extends it to the generation of weather maps. In particular, this work focuses on leveraging latent diffusion models to generate an ensemble forecast.
This work uses the WeatherBench dataset as the source of meteorological data for model training. The goal is to predict geopotential fields at the atmospheric level of 500Pha. Given the initial atmospheric conditions, the diffusion model aims to generate a three-day forecast. We investigates the use of high-dimensional atmospheric feature variables as conditioning variables, a venture relatively unexplored in existing studies on diffusion models.
Moreover, it offers quantitative evaluations of diffusion models in short-medium range forecasts and compares the performance of these models with other methods previously trained on the WeatherBench dataset using metrics specific to ensemble predictions like CRPS. This work additionally presents a critical analysis of the results and proposes solutions for further experiments.

# Dataset

The dataset can be downloadedd from [Weatherbench](https://dataserv.ub.tum.de/index.php/s/m1524895?path=%2F5.625deg)

# Training 
`cd scripts` <br />
## Autoencoders

### Variational Autoencoder 
`python train_ae_nll.py`<br />


### VQ- Autoencoder 
`python train_vq.py` <br />

### VQ-GAN  
For 1 channel geopotential 500 <br/ >
`python train_vqgan_1ch.py` <br/ >
For 1 channel 110-channel conditioning <br/ >
`python train_vqgan_110ch.py` <br/ >


## Diffusion model
Train model without time embedding<br/ >
`python train_diffusion.py`<br/ >


Train model with time embedding<br/ >
`python train_diffusion_time_emb.py` <br/ >

# Inference
The inference code is provided in `notebooks/solver_analysis.ipynb`

