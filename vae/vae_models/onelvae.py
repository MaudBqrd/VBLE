import math
from abc import abstractmethod

import torch
import torch.nn as nn

from .encoder import LightEncoder, LightEncoderWithFixedVariance
from .decoder import LightDecoder


class OneLatentVAE(nn.Module):
    """
    Base class for VAEs with one latent variable
    """

    def __init__(self, in_channels: int, N: int, M: int, gamma=1., learn_top_prior=False):
        """
        Parameters
        ----------
        in_channels: number of input channels
        N: number of channels of the latent variable
        M: number of hidden channels in the encoder and decoder
        gamma: standard deviation of the decoder (p(x|z) = N(mu_\theta(z), \gamma^2 I)), to fix the trade off between
               the KL and the data fidelity term. Put gamma='variable' to learn it as a nn parameters
        learn_top_prior: True to learn standard deviations of the prior p_\theta(z) as nn parameters
        """
        super(OneLatentVAE, self).__init__()

        self.gamma = gamma
        self.learn_gamma = gamma == "variable"
        self.in_channels = in_channels
        self.learn_top_prior = learn_top_prior

        if self.learn_top_prior:
            self.prior_scale = nn.Parameter(torch.ones(M))
        else:
            self.prior_scale = nn.Parameter(torch.ones(M), requires_grad=False)

    def forward(self, x):
        out_encoder = self.encoder(x)
        z = self.posterior_sample(out_encoder)
        out_encoder.update({"z": z})
        out_decoder = self.decoder(z)
        out_decoder.update(out_encoder)
        return out_decoder

    def generator_sample(self, img_shape):
        """
        Image generation

        Parameters
        ----------
        img_shape: (W,H) spatial dimensions of input images
        """
        z = self.z_sample(img_shape)
        x = self.decoder(z)
        return x

    def get_prior_scale(self, top_var_shape, batch_size: int):
        """
        Get p_\theta(z) parameters (p_\theta(z) = N(0,1) if learn_top_prior=False,
        else p_\theta(z) = N(0,\sigma_\theta^2I))

        Parameters
        ----------
        top_var_shape: Spatial shape (W',H') of the latent variable (C,H',W')
        batch_size: wanted batch size
        """
        prior_scale = torch.clip(
            self.prior_scale.view((1, -1, 1, 1)).expand((batch_size, -1, *top_var_shape)),
            min=1e-5)
        return prior_scale

    @classmethod
    def instantiate(cls, in_channels, args, vae_config):
        model = cls(in_channels=in_channels, gamma=args["gamma"], learn_top_prior=args["learn_top_prior"],
                    **vae_config[args["model"]])
        return model

    @abstractmethod
    def posterior_sample(self, out_encoder: dict) -> torch.Tensor:
        pass

    @abstractmethod
    def z_sample(self, img_shape) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_losses(self, x: torch.Tensor, out_dict: dict) -> dict:
        pass


class OneLatentVAEVanilla(OneLatentVAE):
    """
    Classical VAE
    """

    def __init__(self, in_channels, N, M, gamma=1., learn_top_prior=False):
        super().__init__(in_channels, N=N, M=M, gamma=gamma, learn_top_prior=learn_top_prior)

        self.encoder = LightEncoder(in_channels, N=N, M=M)
        self.decoder = LightDecoder(in_channels, N=N, M=M, gamma=gamma)

    def posterior_sample(self, out_encoder: dict) -> torch.Tensor:
        """
        Posterior sampling z ~ q_\phi (z|x)

        Parameters
        ----------
        out_encoder: {"mu_z": posterior mean, "logsd_z": posterior log std}
        """
        sd_z = torch.exp(out_encoder["logsd_z"])
        eps = torch.randn_like(out_encoder["mu_z"])
        return out_encoder["mu_z"] + eps * sd_z

    def z_sample(self, img_shape):
        """
        Prior sampling z ~ p_\theta(z)

        Parameters
        ----------
        img_shape: (H,W) spatial dimension of images
        """
        top_var_shape = (img_shape[0] // 4, img_shape[1] // 4)
        prior_scale = self.get_prior_scale(top_var_shape, 1)
        top_prior_distribution = torch.distributions.normal.Normal(0, prior_scale)
        z = top_prior_distribution.sample()
        return z

    def compute_losses(self, x: torch.Tensor, out_dict: dict) -> dict:
        """
        Parameters
        ----------
        x: (bs,C,H,W) input image
        out_dict: output of the forward function {"x_rec": output image, "gamma_x": gamma param, "mu_z": posterior mean,
                                                  "logsd_z": posterior std}

        Returns
        -------
        loss_dict: {"gen_loss": data fid loss, "mse_loss": MSE, "latent_loss": KL loss}.
                    VAE loss = gen_loss + latent_loss
        """
        HALF_LOG_TWO_PI = 0.91894
        BS, _, H, W = out_dict["x_rec"].shape
        num_pixels = BS * H * W

        loss_dict = {}
        if x is not None:
            if self.learn_gamma:
                loss_dict["gen_loss"] = torch.sum(
                    0.5 * ((x - out_dict["x_rec"]) / out_dict["gamma_x"]).pow(2) + out_dict["gamma_x"].log() + HALF_LOG_TWO_PI) / num_pixels
            else:
                loss_dict["gen_loss"] = torch.sum(0.5 * ((x - out_dict["x_rec"]) / out_dict["gamma_x"]).pow(2)) / num_pixels

            loss_dict["mse_loss"] = torch.mean((x - out_dict["x_rec"]).pow(2))

        if "logsd_z" in out_dict and out_dict["logsd_z"] is not None:
            sd_z = torch.exp(out_dict["logsd_z"])

            # prior_scale = torch.clip(self.prior_scale.view((1, -1, 1, 1)).expand((out_dict["mu_z"].shape[0], -1, *out_dict["mu_z"].shape[2:])), min=1e-5)
            prior_scale = self.get_prior_scale(out_dict["mu_z"].shape[2:], x.shape[0])
            loss_dict["latent_loss"] = 0.5 * torch.sum((out_dict["mu_z"].pow(2) + sd_z.pow(2))/(prior_scale.pow(2)) - 1 + 2 * prior_scale.log() - 2 * out_dict["logsd_z"]) / num_pixels
            # loss_dict["latent_loss"] = 0.5 * torch.sum(out_dict["mu_z"].pow(2) + sd_z.pow(2) - 1 - 2 * out_dict["logsd_z"]) / num_pixels

        return loss_dict


class OneLatentVAEFixedVariance(OneLatentVAEVanilla):
    """
    VAE with a Gaussian approximated posterior having a fixed variance, i.e. q_phi(z|x) = N(mu(x), 0.1 I)
    """

    def __init__(self, in_channels, N, M, gamma=1., sigma=0.1, learn_top_prior=False):
        super().__init__(in_channels, N=N, M=M, gamma=gamma, learn_top_prior=learn_top_prior)
        self.encoder = LightEncoderWithFixedVariance(in_channels, N=N, M=M, sigma=sigma)


class OneLatentVAEUniform(OneLatentVAE):
    """
    Compressive VAE, that is a VAE with a uniform approximated posterior (with fixed variance)
    i.e.  q_phi(z|x) = U(z_bar - 0.5, z_bar + 0.5)
          p_theta(z) = [N(0,1)*U(-1/2,1/2)]
    """

    def __init__(self, in_channels, N, M, gamma=1., learn_top_prior=True):
        """
        To get good results, assert learn_top_prior is equal to True
        """
        super().__init__(in_channels, N=N, M=M, gamma=gamma, learn_top_prior=learn_top_prior)

        self.encoder = LightEncoderWithFixedVariance(in_channels, N=N, M=M)
        self.decoder = LightDecoder(in_channels, N=N, M=M, gamma=gamma)

        self.pz_func = torch.distributions.normal.Normal
        self.quant_step = 1.

    def posterior_sample(self, out_encoder: dict) -> torch.Tensor:
        """
        Posterior sampling z ~ q_\phi (z|x)

        Parameters
        ----------
        out_encoder: {"mu_z": z_bar posterior mean, "quant_step": = 1 in training but left as a parameter}
        """
        if 'quant_step' in out_encoder:
            quant_step = out_encoder['quant_step']
        else:
            quant_step = self.quant_step
        quant_noise_z = torch.zeros(out_encoder["mu_z"].shape)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -1/2, 1/2).to(out_encoder["mu_z"].device)
        return out_encoder["mu_z"] + quant_step * quant_noise_z

    def z_sample(self, img_shape):
        """
        Prior sampling z ~ p_\theta(z)

        Parameters
        ----------
        img_shape: (H,W) spatial dimension of images
        """
        top_var_shape = (img_shape[0] // 4, img_shape[1] // 4)
        prior_scale = self.get_prior_scale(top_var_shape, 1)
        top_prior_distribution = torch.distributions.normal.Normal(0, prior_scale)
        z = top_prior_distribution.sample()
        z = z + torch.rand(z.shape) - 0.5
        return z

    def compute_losses(self, x, out_dict):
        """
        Parameters
        ----------
        x: (bs,C,H,W) input image
        out_dict: output of the forward function {"x_rec": output image, "gamma_x": gamma param,
                                                  "z": sampled latent variable}

        Returns
        -------
        loss_dict: {"gen_loss": data fid loss, "mse_loss": MSE, "latent_loss": KL loss}.
                    VAE loss = gen_loss + latent_loss
        """
        HALF_LOG_TWO_PI = 0.91894
        BS, _, H, W = out_dict["x_rec"].shape
        num_pixels = BS * H * W

        loss_dict = {}
        if x is not None:
            if self.learn_gamma:
                loss_dict["gen_loss"] = torch.sum(
                    0.5 * ((x - out_dict["x_rec"]) / out_dict["gamma_x"]).pow(2) + out_dict["gamma_x"].log() + HALF_LOG_TWO_PI) / num_pixels
            else:
                loss_dict["gen_loss"] = torch.sum(0.5 * ((x - out_dict["x_rec"]) / out_dict["gamma_x"]).pow(2)) / num_pixels
            loss_dict["mse_loss"] = torch.mean((x - out_dict["x_rec"]).pow(2))

        loss_dict["latent_loss"] = math.log(2.0) * self.feature_probs_based_sigma(out_dict["z"])[0] / num_pixels

        return loss_dict

    def feature_probs_based_sigma(self, z):
        """
        Auxilliary function for computing the rate loss of compressive VAEs
        """
        # prior_scale = self.prior_scale.view((1,-1,1,1)).expand((z.shape[0], -1, *z.shape[2:]))
        prior_scale = self.get_prior_scale(z.shape[2:], z.shape[0])

        probs = self.pz_func(0, prior_scale).cdf(z + self.quant_step/2) - self.pz_func(0, prior_scale).cdf(z - self.quant_step/2)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
        return total_bits, probs


