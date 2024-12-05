import torch
from abc import ABC
import torch.nn as nn
import numpy as np

from cae.cae_models import CompressAICAE
from vae.vae_models import vae_models, vae_config

from.prior_nn_utils import get_latent_sampling_fn

def get_priornn_module(model_path: str, model_type: str, device: str, in_channels: int, latent_inference_model: str, optimize_h: bool = False):
    """
    Return the PriorNN module
    """
    if "mbt" in model_type or "cheng" in model_type:
        return CAEPriorNN(model_path, model_type, device, in_channels, latent_inference_model, optimize_h)
    elif "vae" in model_type:
        return VAEPriorNN(model_path, model_type, device, in_channels, latent_inference_model)
    else:
        raise ValueError(f'Unknown model type {model_type}')


class PriorNN(ABC):
    """
    Class wrapping the neural network regularizer
    """

    def __init__(self, model_path: str, model_type: str, device: str = 'cuda', in_channels: int = 1,
                 latent_inference_model: str = 'uniform', optimize_h: bool = False):
        """
        Parameters
        ----------
        model_path: path to the nn checkpoint
        model_type: model type, available = cheng, mbt
        device: cuda or cpu
        in_channels: number of channels
        latent_inference_model: uniform, gaussian, dirac
        optimize_h: True if (z,h) is optimized instead of z for CAE models with hyperprior
        """

        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.in_channels = in_channels
        self.latent_inference_model = latent_inference_model
        self.optimize_h = optimize_h

        self.latent_sampling_fn = get_latent_sampling_fn(latent_inference_model, device)

    def sample_vble_model(self, inference_params: nn.Module, sampling: bool = False, n_samples: int = 1):
        """
        Sample x \sim q(x|z)q(z). If sampling is False, return the center of the distribution
        """
        pass

    def net_encoder(self, x: torch.Tensor):
        """
        Encoder network, sample z \sim q_\phi(z|x)
        """
        pass

    def compute_kl_z(self, inference_params: nn.Module, out_vble_sampling: dict):
        """
        Compute KL(q(z)||p_theta(z))
        """
        pass

    @classmethod
    def load_prior_model(cls, model_path: str, model_type: str, device: str, n_channels: int):
        """
        Load the model used for regularization.

        Parameters
        ----------
        model_path: path to the nn checkpoint
        model_type: model type
        device: cuda or cpu
        n_channels: number of channels
        """
        if model_type in ["mbt", "cheng"]:
            net = CompressAICAE(model_path, model_type, device, n_channels)
        elif '1lvae' in model_type:
            ckpt = torch.load(model_path, map_location=device)
            net = vae_models[model_type].instantiate(in_channels=n_channels, gamma=ckpt['config']['gamma'], learn_top_prior=False, default_params=vae_config[model_type])
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
        else:
            raise NotImplementedError

        # freeze nn parameters
        for p in net.parameters():
            p.requires_grad = False

        net.to(device)
        return net
        
    
    def get_latent_dim(self, x_shape):
        """
        Get the dimension of the latent space

        Parameters
        ----------
        x_shape: (C,H,W)
        optimize_h: True if (z,h) is optimized instead of z for CAE models with hyperprior
        """
        with torch.no_grad():
            x0 = torch.zeros(x_shape).to(self.device)
            out_dict = self.net_encoder(x0[None, :])
            # out_dict = self.net_func((x0[None, :], torch.tensor([1.]).float().to('cuda')))
            if self.optimize_h:
                z = (out_dict['z'], out_dict['h'])
            else:
                z = out_dict['z']
        if len(z) > 1:
            zdim = []
            for zi in z:
                zdim.append(zi.shape)
        else:
            zdim = [z.shape]
        return zdim


class CAEPriorNN(PriorNN):
    """
    Class wrapping the CAE regularizer
    """

    def __init__(self, model_path: str, model_type: str, device: str = 'cuda', in_channels: int = 1,
                 latent_inference_model: str = 'uniform', optimize_h: bool = False):
        super().__init__(model_path, model_type, device, in_channels, latent_inference_model, optimize_h)

        self.net = PriorNN.load_prior_model(model_path, model_type, device, in_channels)

        # assert latent_inference_model in ["uniform", "dirac"], f"Latent inference model {latent_inference_model} not supported for CAE"
    

    def sample_vble_model(self, inference_params: nn.Module, sampling: bool = False, n_samples: int = 1):
        """
        Sample x \sim q(x|z)q(z). If sampling is False, return the center of the distribution
        """

        # Sampling of z
        z_bar = inference_params.get_zbar()
        if sampling:
            z = z_bar.expand((n_samples, -1, -1, -1))
            z = z + self.latent_sampling_fn(z.size(), inference_params.get_az())
        else:
            z = z_bar

        # Sampling of the second latent variable h if --optimize_h
        if self.optimize_h:
            h_bar = inference_params.get_hbar()
            if sampling:
                h = h_bar.expand((n_samples, -1, -1, -1))
                h = h + self.latent_sampling_fn(h.size(), inference_params.get_ah())
            else:
                h = h_bar
            out_dict = self.net.decoder({"z_bar": inference_params.get_zbar().expand((z.shape[0],-1,-1,-1)), "z": z, "h_bar": inference_params.get_hbar().expand((h.shape[0],-1,-1,-1)), "h": h})
            out_dict.update({"z_bar": inference_params.get_zbar(), "z": z, "h_bar": inference_params.get_hbar(), "h": h})
        else:
            out_dict = self.net.decoder({"z_bar": inference_params.get_zbar().expand((z.shape[0],-1,-1,-1)), "z": z})
            out_dict.update({"z_bar": z_bar, "z": z})

        return out_dict


    def net_encoder(self, x: torch.Tensor, posterior_sampling: bool = False):
        """
        Encoder network
        """
        if self.in_channels == 1:
            x = x.expand((-1, 3, -1, -1))
        out_encoder = self.net.encoder(x)

        if posterior_sampling:
            out_encoder["h"] = out_encoder['h_bar'] + torch.rand_like(out_encoder['h_bar']).to(self.device) - 0.5
            out_encoder["z"] = out_encoder['z_bar'] + torch.rand_like(out_encoder['z_bar']).to(self.device) - 0.5
        else:
            out_encoder["h"] = out_encoder['h_bar']
            out_encoder["z"] = out_encoder['z_bar']
        return out_encoder
    

    def net_decoder(self, out_encoder: dict):
        """
        Decoder network
        """
        out_decoder = self.net.decoder(out_encoder)
        out_decoder.update(out_encoder)
        if self.in_channels == 1:
            out_decoder["x_rec"] = torch.mean(out_decoder["x_rec"], dim=1, keepdim=True)
            if "x_rec_std" in out_decoder:
                out_decoder["x_rec_std"] = torch.mean(out_decoder["x_rec_std"], dim=1, keepdim=True)
        if "sigma_decoder" not in out_decoder:
            out_decoder["x_rec_std"] = None
        return out_decoder


    def compute_kl_z(self, inference_params: nn.Module, out_vble_sampling: dict):
        """
        Compute KL(q(z)||p_\theta(z)): Monte Carlo approximation E[log(q(z))-log(p_\theta(z))]
        Accepted latent_inference_model: uniform, dirac
        """
        _, _, H, W = out_vble_sampling["x_rec"].size()
        num_pixels = H * W
        n_latent_sample = out_vble_sampling["z"].size(0)

        # compute log(p(z))
        bpp_loss = self.net.compute_bpp_loss(out_vble_sampling["likelihoods"], num_pixels)
        minus_log_pz = bpp_loss * np.log(2) / n_latent_sample

        if self.latent_inference_model == "dirac":
            return minus_log_pz
        elif self.latent_inference_model in ["uniform", "gaussian"]:
            log_az = torch.sum(torch.log(inference_params.get_az())) / num_pixels
            if self.optimize_h:
                log_ah = torch.sum(torch.log(inference_params.get_ah())) / num_pixels
            else:
                log_ah = 0
            kl_z = - log_az - log_ah + minus_log_pz
            return kl_z
        else:
            raise NotImplementedError


class VAEPriorNN(PriorNN):
    """
    Class wrapping the VAE regularizer
    """

    def __init__(self, model_path: str, model_type: str, device: str = 'cuda', in_channels: int = 1,
                 latent_inference_model: str = 'uniform'):
        super().__init__(model_path, model_type, device, in_channels, latent_inference_model, optimize_h=False)

        self.net = PriorNN.load_prior_model(model_path, model_type, device, in_channels)
        assert latent_inference_model in ["gaussian", "dirac"], f"Latent inference model {latent_inference_model} not supported for VAE"



    def sample_vble_model(self, inference_params: nn.Module, sampling: bool = False, n_samples: int = 1):
        """
        Sample x \sim q(x|z)q(z). If sampling is False, return the center of the distribution
        """
        # Sampling of z
        z_bar = inference_params.get_zbar()
        if sampling:
            z = z_bar.expand((n_samples,*([-1]*(len(z_bar.shape)-1))))
            z = z + self.latent_sampling_fn(z.size(), inference_params.get_az())
        else:
            z = z_bar
        
        # Decoding z and computing p(x|z)
        out_dict = self.net.decoder(z)
        out_dict.update({"z": z, "z_bar": z_bar})
        return out_dict

    def net_encoder(self, x: torch.Tensor, posterior_sampling: bool = False):
        """
        Encoder network
        """
        out_encoder = self.net.encoder(x)
        if posterior_sampling:
            z = self.net.posterior_sample(out_encoder)
        else:
            z = out_encoder["mu_z"]
        out_encoder.update({"z": z})
        return out_encoder
    
    def net_decoder(self, out_encoder: dict):
        """
        Decoder network
        """
        out_decoder = self.net.decoder(out_encoder["z"])
        return out_decoder


    def compute_kl_z(self, inference_params: nn.Module, out_vble_sampling: dict):
        """
        Compute KL(q(z)||p_\theta(z)): 
        Accepted latent_inference_model: dirac : approximate KL with Monte Carlo
                                         gaussian : KL between two gaussians
        """
        _, _, H, W = out_vble_sampling["x_rec"].size()
        num_pixels = H * W
        n_latent_sample = out_vble_sampling["z"].size(0)

        prior_scale = self.net.get_prior_scale().to(out_vble_sampling["z"].device).expand(inference_params.get_zbar().shape)
        if len(out_vble_sampling["z"].shape) > 2:  # fully convolutionnal bottleneck
            out_vble_sampling["z"] = out_vble_sampling["z"].expand((out_vble_sampling["z"].shape[0], -1, *out_vble_sampling["z"].shape[2:]))
        if self.latent_inference_model == "dirac":  # 0.5||z||^2
            kl_z = 0.5 * torch.sum((out_vble_sampling["z"] / prior_scale.to(out_vble_sampling["z"].device)).pow(2)) / (num_pixels * n_latent_sample)
        elif self.latent_inference_model == "gaussian": # KL(q(z)||p(z)) between two gaussians
            a = inference_params.get_az()
            kl_z = 0.5 * torch.sum((a.pow(2) + inference_params.get_zbar().pow(2)) / prior_scale.pow(2) + 2 * torch.log(prior_scale) - 2 * torch.log(a) -1) / num_pixels
        else:
            raise NotImplementedError
        return kl_z