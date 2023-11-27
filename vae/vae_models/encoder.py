import numpy as np
import torch
import torch.nn as nn


class LightEncoder(nn.Module):
    """
    Encoder (inference model) of a VAE
    """

    def __init__(self, in_channels: int, N: int = 32, M: int = 64):
        """
        Parameters
        ----------
        in_channels: number of input channels
        N: number of channels of the latent variable
        M: number of hidden channels in the encoder and decoder
        """
        super(LightEncoder, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, N, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(N),
            # nn.LeakyReLU(),
            # nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.Conv2d(N, 2*M, kernel_size=5, stride=2, padding=2)
        )

    def forward(self, x):
        x = self.layers(x)
        mu_z, logsd_z = x.chunk(2, dim=1)
        return {
            "mu_z": mu_z,
            "logsd_z": logsd_z
        }


class LightEncoderWithFixedVariance(nn.Module):
    """
    Encoder (inference model) of a VAE with a fixed posterior variance q(z|x)
    """

    def __init__(self, in_channels: int, N:int = 32, M:int = 64, sigma: float = 1.):
        """
        Parameters
        ----------
        in_channels: number of input channels
        N: number of channels of the latent variable
        M: number of hidden channels in the encoder and decoder
        sigma: standard deviation of the approximated posterior q(z|x)
        """
        super(LightEncoderWithFixedVariance, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, N, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            # nn.BatchNorm2d(N),
            # nn.LeakyReLU(),
            # nn.Conv2d(N, N, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.Conv2d(N, M, kernel_size=5, stride=2, padding=2)
        )

        self.logsd_z = nn.Parameter(np.log(sigma) * torch.tensor([1.]), requires_grad=False)

    def forward(self, x):
        mu_z = self.layers(x)
        return {
            "mu_z": mu_z,
            "logsd_z": self.logsd_z
        }

