from model import Unet3D
from diffusion import GaussianDiffusion
from trainer import Trainer
import torch

model = Unet3D(
    dim = 64,
    dim_mults = (1, 2, 4,8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 64,
    num_frames = 8,
    timesteps = 1000,   
    loss_type = 'l1'    
).cuda()

trainer = Trainer(
    diffusion,
    './flow_dataset',      
    train_batch_size =4,
    train_lr = 1e-4,
    save_and_sample_every = 500,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    results_folder = './results',               
    num_sample_rows=2
)

trainer.train()