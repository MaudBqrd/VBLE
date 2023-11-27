import torch
import torch.nn as nn


class LightDecoder(nn.Module):
    """
    Decoder (generative model) of a VAE
    """

    def __init__(self, in_channels: int, N:int = 32, M:int = 64, gamma: float = 1.):
        """
        Parameters
        ----------
        in_channels: number of input channels
        N: number of channels of the latent variable
        M: number of hidden channels in the encoder and decoder
        gamma: standard deviation of the decoder (p(x|z) = N(mu_\theta(z), \gamma^2 I)), to fix the trade off between
               the KL and the data fidelity term. Put gamma='variable' to learn it as a nn parameters
        """
        super(LightDecoder, self).__init__()

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            # nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            # nn.BatchNorm2d(N),
            # nn.LeakyReLU(),
            nn.ConvTranspose2d(N, N, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm2d(N),
            nn.LeakyReLU(),
            nn.Conv2d(N, in_channels, kernel_size=5, stride=1, padding=2),
            # nn.ConvTranspose2d(N, in_channels, kernel_size=5, stride=2, padding=2, output_padding=1)
        )

        self.learn_gamma = gamma == "variable"
        if self.learn_gamma:
            self.gamma_x = nn.Parameter(torch.ones(1))
        else:
            self.gamma_x = nn.Parameter(float(gamma) * torch.ones(1), requires_grad=False)

    def forward(self, z):
        x = self.layers(z)
        return {
            "x_rec": x,
            "gamma_x": self.gamma_x
        }
