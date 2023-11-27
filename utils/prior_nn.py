import logging
import math
import torch

from vae.vae_models import vae_models, vae_config
from compressai_models import compressai_models, load_pretrained


class PriorNN:
    """
    Class wrapping the VAE/CAE regularizer
    """

    def __init__(self, model_path: str, model_type: str, device: str = 'cuda', algo: str = 'vble', in_channels: int = 1,
                 optimize_h: bool = False):
        """
        Parameters
        ----------
        model_path: path to the nn checkpoint
        model_type: model type, available = cheng, mbt, mbt-mean (CAE),
                                            1lvae-vanilla, 1lvae-fixedvar, 1lvae-uniform (VAE)
        device: cuda or cpu
        algo: optimization algorithm vble or mapz (deterministic gradient descent)
        in_channels: number of input channels
        optimize_h: True to optimize (z,h) instead of z for CAE models with hyperprior
        """

        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.in_channels = in_channels

        self.decoder, self.net_func, self.compute_losses = self.get_encoder_decoder(in_channels, optimize_h)

        if algo is not None:
            self.reg_func = self.get_reg_func(algo, optimize_h)
            self.algo = algo
        else:
            self.algo = None

    def get_reg_func(self, algo: str, optimize_h: bool):
        """
        Get the regularization function depending on the algorithm

        Parameters
        ----------
        algo: optimization algorithm vble or mapz (deterministic gradient descent)
        optimize_h: True to optimize (z,h) instead of z for CAE models with hyperprior
        """

        if algo == 'mapz':
            reg_func = lambda out_dict: self.compute_losses(out_dict)["latent_loss"]
        elif algo == 'vble':
            if optimize_h:
                def custom_reg_func(out_dict):
                    N, _, H, W = out_dict["x_rec"].size()
                    num_pixels = N * H * W
                    bpp_loss = self.compute_losses(out_dict)["latent_loss"]
                    return bpp_loss - torch.sum(torch.log(out_dict["a"][0])) / (num_pixels * math.log(2)) - torch.sum(torch.log(out_dict["a"][1])) / (num_pixels * math.log(2))
            else:
                def custom_reg_func(out_dict):
                    N, _, H, W = out_dict["x_rec"].size()
                    num_pixels = N * H * W
                    bpp_loss = self.compute_losses(out_dict)["latent_loss"]
                    return bpp_loss - torch.sum(torch.log(out_dict["a"])) / (num_pixels*math.log(2))
            reg_func = custom_reg_func
        elif algo == "custom":
            # Craft your custom regularization here
            def custom_reg_func(out_dict):
                pass
            reg_func = custom_reg_func
        else:
            raise NotImplementedError()
        return reg_func

    def get_encoder_decoder(self, in_channels: int, optimize_h=False):
        """
        Function to wrap the neural network used for regularization into a decoder, net_func and compute_losses
        functions.

        net_func: x [BS, C, H, W] nn.Tensor, latent_sampling (bool), noise_factor -> dict (keys=("x_rec") at least))
        decoder : z [latent dim] or (z,h) nn.Tensor/list nn.Tensor, latent_sampling (bool), noise_factor -> dict (keys=("x_rec") at least))
        compute_losses : fct dict -> dict (keys=("latent_loss"))
        """

        net = PriorNN.load_prior_model(self.model_path, self.model_type, self.device, in_channels)

        if '1lvae' in self.model_type:
            decoder, net_func, compute_losses = get_nn_func_1lvae(net)

        elif self.model_type in ["mbt", "mbt-mean", "cheng"]:
            bu_values = not optimize_h
            decoder, net_func, compute_losses = get_nn_func_compressai(net, self.device, bu_values=bu_values)

        else:
            raise NotImplementedError()

        return decoder, net_func, compute_losses

    @classmethod
    def load_prior_model(cls, model_path: str, model_type: str, device: str, n_channels: int):
        """
        Load the model used for regularization.

        Parameters
        ----------
        model_path: path to the nn checkpoint
        model_type: model type, available = cheng, mbt, mbt-mean (CAE),
                                            1lvae-vanilla, 1lvae-fixedvar, 1lvae-uniform (VAE)
        device: cuda or cpu
        n_channels: number of channels
        """
        if '1lvae' in model_type:
            ckpt = torch.load(model_path, map_location='device')
            net = vae_models[model_type].instantiate(in_channels=n_channels, args=ckpt['config'], vae_config=vae_config)
            net.load_state_dict(ckpt["state_dict"])
            net.eval()
        elif model_type in ["mbt", "mbt-mean", "cheng"]:
            checkpoint = torch.load(model_path, map_location=device)
            if "state_dict" in checkpoint.keys():  # retrained model from compressAI
                net = compressai_models[model_type].from_state_dict(checkpoint["state_dict"]).to(device)
            else:  # model directly downloaded from compressAI without finetuning
                net = compressai_models[model_type].from_state_dict(load_pretrained(checkpoint)).to(device)
        else:
            raise NotImplementedError

        # freeze nn parameters
        for p in net.parameters():
            p.requires_grad = False

        net.to(device)
        return net

    def get_latent_dim(self, x_shape, optimize_h: bool):
        """
        Get the dimension of the latent space

        Parameters
        ----------
        x_shape: (C,H,W)
        optimize_h: True if (z,h) is optimized instead of z for CAE models with hyperprior
        """
        with torch.no_grad():
            x0 = torch.zeros(x_shape).to(self.device)
            out_dict = self.net_func(x0[None, :])
            # out_dict = self.net_func((x0[None, :], torch.tensor([1.]).float().to('cuda')))
            if optimize_h:
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


## -------- CUSTOM FUNCTIONS FOR EACH REGULARIZATION NEURAL NETWORK ----------

def get_nn_func_1lvae(net):
    """
    Get decoder, net_func and compute_losses function for VAE models.
    """
    logger = logging.getLogger('logfile')

    def net_decoder(bar_z: list, latent_sampling=False, noise_factor: torch.Tensor = torch.tensor([1.])) -> dict:
        """
        Return D(bar_z) if not latent_sampling=False
               D(bar_z + noise_factor*eps) with eps ~ N(0,1) or eps ~ U(-0.5,0.5) (1lvae-uniform model) if latent_sampling

        Parameters
        ---------
        bar_z: [bar_z]
        latent_sampling: to do sampling before decoding
        noise_factor: tensor, size=z[0].shape

        Returns
        -------
        out_decoder: dict {"x_rec":..., "z": z_sampled, "mu_z": z not sampled}
        """
        if latent_sampling:
            noise_factor = noise_factor.to(bar_z[0].device)
            z_sampled = net.posterior_sample({"mu_z": bar_z[0], "logsd_z": torch.log(noise_factor), "quant_step": noise_factor})
        else:
            z_sampled = bar_z
        out_decoder = net.decoder(z_sampled)
        out_decoder.update({"z": z_sampled, "mu_z": bar_z[0]})
        return out_decoder

    def net_func(x: torch.Tensor, latent_sampling=False) -> dict:
        """
        Forward function of the model, with or without sampling in the latent space

        Parameters
        ----------
        x: shape=(1,C,H,W)
        latent_sampling: True to enable latent sampling with classical approximated posterior q_\phi(z|x)

        Returns
        -------
        out_decoder: {"x_rec":..., "z": z_sampled, "mu_z": z not sampled}
        """
        out_encoder = net.encoder(x)
        if latent_sampling:
            z = net.posterior_sample(out_encoder)
        else:
            z = out_encoder["mu_z"]
        out_encoder.update({"z": z})
        out_decoder = net.decoder([z])
        out_decoder.update(out_encoder)
        return out_decoder

    def compute_losses(out_dict: dict, no_kl_loss=True) -> dict:
        """
        Compute the latent loss L(z) that will be approximately used in image restoration by R(z) = \lambda L(z),
        see the method get_reg_func of PriorNN class

        Parameters
        ----------
        out_dict: output of net_decoder function defined above
        no_kl_loss: auxiliary parameter. If True, L(z) = 0.5 ||z||^2, if False, L(z) = KL(N(mu_z, sigma_z)||N(0,1))

        Returns
        -------
        standard_loss_dict: {"latent_loss": ...}
        """
        loss_dict = net.compute_losses(x=None, out_dict=out_dict)

        standard_loss_dict = {}

        if "latent_loss" not in loss_dict or no_kl_loss:  # 1lvae-vanilla
            BS, _, H, W = out_dict["x_rec"].shape
            num_pixels = BS * H * W
            prior_scale = net.get_prior_scale(out_dict["mu_z"].shape[2:], 1)
            standard_loss_dict["latent_loss"] = 0.5 * torch.sum((out_dict["z"]/prior_scale).pow(2)) / (num_pixels * math.log(2))
        else:
            logger.warning("KL loss taken as latent loss")
            standard_loss_dict["latent_loss"] = loss_dict["latent_loss"] / math.log(2)

        return standard_loss_dict

    return net_decoder, net_func, compute_losses


def get_nn_func_compressai(net, device='cuda', bu_values=False):
    """
    Get decoder, net_func and compute_losses function for compressive AE models.
    """

    def net_decoder_bu_values(z_bar, latent_sampling=False, noise_factor=1.):
        """
        Return D(z_bar) if not latent_sampling=False
               D(z_bar + noise_factor*u) with u ~ U(-0.5,0.5)  if latent_sampling

        Parameters
        ---------
        z_bar: [latent variable]
        latent_sampling: to do sampling before decoding
        noise_factor: tensor, size=z[0].shape

        Returns
        -------
        out_decoder: dict {"x_rec":..., "z": z_sampled, "z_bar": z not sampled}
        """
        z_bar = z_bar[0]
        if latent_sampling:
            z = z_bar + noise_factor * (torch.rand_like(z_bar).to(device) - 0.5)
        else:
            z = z_bar

        out_dict = net.decoder({"z_bar": z_bar, "z": z})
        return out_dict

    def net_decoder(z_bar, latent_sampling=False, noise_factor=[1., 1.]):
        """
        Case when z AND h are optimized, thus z_bar = [z_bar, h_bar]

        Return D(z_bar) if not latent_sampling=False
               D(z_bar + noise_factor*u) with u ~ U(-0.5,0.5)  if latent_sampling

        Parameters
        ---------
        z_bar: [z_bar, h_bar]
        latent_sampling: to do sampling before decoding
        noise_factor: tensor, size=z[0].shape

        Returns
        -------
        out_decoder: dict {"x_rec":..., "z": z_sampled, "z_bar": z not sampled, "h": h_sampled, "h_bar": h not sampled}
        """
        z_bar, h_bar = z_bar[:2]
        if latent_sampling:
            z = z_bar + noise_factor[0] * (torch.rand_like(z_bar).to(device) - 0.5)
            h = h_bar + noise_factor[1] * (torch.rand_like(h_bar).to(device) - 0.5)
        else:
            z = z_bar
            h = h_bar

        out_dict = net.decoder({"z_bar": z_bar, "z": z, "h_bar": h_bar, "h": h})
        return out_dict

    def net_func(x, latent_sampling=False):
        """
        Forward function of the model, with or without sampling in the latent space

        Parameters
        ----------
        x: shape=(1,C,H,W)
        latent_sampling: True to enable latent sampling with classical approximated posterior q_\phi(z|x)

        Returns
        -------
        out_decoder: {"x_rec":..., "z": z_sampled, "z_bar": z not sampled}
        """
        out_encoder = net.encoder(x)

        if latent_sampling:
            out_encoder['z'] = out_encoder['z_bar'] + torch.rand_like(out_encoder['z_bar']).to(device) - 0.5
            out_encoder['h'] = out_encoder['h_bar'] + torch.rand_like(out_encoder['h_bar']).to(device) - 0.5
        else:
            out_encoder['z'] = out_encoder['z_bar']
            out_encoder['h'] = out_encoder['h_bar']

        out_decoder = net.decoder(out_encoder)
        out_decoder.update(out_encoder)
        return out_decoder

    def compute_losses(out_dict):
        """
        Compute the Rate loss L(z) that will be approximately used in image restoration by R(z) = \lambda L(z),
        see the method get_reg_func of PriorNN class

        Parameters
        ----------
        out_dict: output of net_decoder/net_decoder_bu_values function defined above

        Returns
        -------
        standard_loss_dict: {"latent_loss": ...}
        """
        loss_dict = net.compute_losses(None, out_dict)
        standard_loss_dict = {}
        standard_loss_dict["latent_loss"] = loss_dict["bpp_loss"]
        return standard_loss_dict

    if bu_values:
        return net_decoder_bu_values, net_func, compute_losses
    return net_decoder, net_func, compute_losses

