
import torch
from torchvision import transforms as T
from PIL import Image
import torch.nn.functional as F
from torch import nn
import matplotlib.pyplot as plt
import numpy as np  

def noop(*args, **kwargs):
    pass

def exists(x):
    return x is not None

def is_odd(n):
    return (n % 2) == 1

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def is_list_str(x):

    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def cycle(dl):
    while True:
        for data in dl:
            yield data

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}


def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif
def video_tensor_to_gif(tensor, path, duration = 120, loop = 0, optimize = True):
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(path, save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor
def gif_to_tensor(path, channels = 1, transform = T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    tensors = torch.stack(tensors, dim = 1)
    return tensors

def identity(t, *args, **kwargs):
    return t


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))

def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))