import torch
import numpy as np

from compressai.ops.bound_ops import LowerBound

def get_latent_sampling_fn(latent_inference_model: str, device: str):
    """
    Return the latent sampling function
    """
    if latent_inference_model == 'uniform':
        latent_distrib = torch.distributions.uniform.Uniform(torch.tensor([0.]).to(device),torch.tensor([1.]).to(device))
        def sampling_func(size, a):
            u = latent_distrib.sample(size)[...,0]
            return a*(u - 0.5)
    elif latent_inference_model == 'gaussian':
        latent_distrib = torch.distributions.normal.Normal(torch.tensor([0.]).to(device),torch.tensor([1.]).to(device))
        def sampling_func(size, a):
            return a*latent_distrib.sample(size)[...,0]
    elif latent_inference_model == 'dirac':
        def sampling_func(size, a):
            return 0
    else:
        raise ValueError(f'Unknown latent inference model {latent_inference_model}')
    return sampling_func


class AverageSolution:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def reinit(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def getval(self):
        return self.avg


def init_lower_bound_z(vmin=1.5e-5, vmax=20, device='cuda'):
    """
    Initialize the lower bound for the latent variable z
    """
    
    lb_min = LowerBound(np.log(vmin)).to(device)
    lb_max = LowerBound(-np.log(vmax)).to(device)

    def lower_bound_fn(z):
        return lb_min(-lb_max(-z))

    return lower_bound_fn
