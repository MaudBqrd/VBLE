import torch
import math
import torch.nn as nn

from compressai.models import JointAutoregressiveHierarchicalPriors, Cheng2020Attention

class CompressAICAE(nn.Module):

    def __init__(self, model_path, model_type, device, n_channels):
        super().__init__()

        self.model_type = model_type
        self.in_channels = n_channels

        state_dict = torch.load(model_path, map_location=device)["state_dict"]

        if model_type == "mbt":
            self.model = JointAutoregressiveHierarchicalPriors.from_state_dict(state_dict).to(device)
        elif model_type == "cheng":
            self.model = Cheng2020Attention.from_state_dict(state_dict).to(device)
        else:
            raise ValueError("Model type not recognized")
        
        self.model.eval()
    

    def encoder(self, x):
        
        x = x.expand(-1, 3, -1, -1) if self.in_channels == 1 else x 
        y = self.model.g_a(x)
        z = self.model.h_a(y)
        return {
            "z_bar": y,
            "h_bar": z
        }

    def decoder(self, encoder_dict):
        encoder_dict.setdefault("z", encoder_dict["z_bar"])

        if 'h' not in encoder_dict.keys():
            encoder_dict["h_bar"] = self.model.h_a(encoder_dict["z_bar"])
        encoder_dict.setdefault("h", encoder_dict["h_bar"])

        ctx_params = self.model.context_prediction(encoder_dict["z"])

        z_hat, z_likelihoods = self.entropy_bottleneck_forward_(encoder_dict["h"])
        params = self.model.h_s(z_hat)
        gaussian_params = self.model.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )

        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_likelihoods = self.gaussian_conditional_likelihood_(encoder_dict["z"], scales_hat, means=means_hat)

        x_hat = self.model.g_s(encoder_dict["z"])
        out_dict = {
            "x_rec": torch.mean(x_hat, dim=1, keepdim=True) if self.in_channels == 1 else x_hat,
            "sigma_z": scales_hat,
            "mu_z": means_hat,
            "z_bar": encoder_dict["z_bar"],
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "h_bar": encoder_dict["h_bar"]
        }
        
        return out_dict
    
    def compute_bpp_loss(self, likelihoods_dict, num_pixels):
        bpp_loss = sum(
            (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
            for likelihoods in likelihoods_dict.values()
        )
        return bpp_loss
    
    def gaussian_conditional_likelihood_(self, z, scales, means=None):
        likelihood = self.model.gaussian_conditional._likelihood(z, scales, means)
        if self.model.gaussian_conditional.use_likelihood_bound:
            likelihood = self.model.gaussian_conditional.likelihood_lower_bound(likelihood)
        return likelihood
    
    def entropy_bottleneck_forward_(self, h):
        perm = torch.cat(
                (
                    torch.tensor([1, 0], dtype=torch.long, device=h.device),
                    torch.arange(2, h.ndim, dtype=torch.long, device=h.device),
                )
            )
        inv_perm = perm
        h = h.permute(*perm).contiguous()
        shape = h.size()
        outputs = h.reshape(h.size(0), 1, -1)
        likelihood, _, _ = self.model.entropy_bottleneck._likelihood(outputs)
        if self.model.entropy_bottleneck.use_likelihood_bound:
            likelihood = self.model.entropy_bottleneck.likelihood_lower_bound(likelihood)

        # Convert back to input tensor shape
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood