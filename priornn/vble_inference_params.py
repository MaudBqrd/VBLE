import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

from .prior_nn_utils import AverageSolution, init_lower_bound_z

def get_estimate_name_to_save(**dico_params):
    if dico_params["latent_inference_model"] != "dirac":
        estimate_to_save = ["x_xmmse", "x_zmmse", "x_xmmse_std"]
    else:
        estimate_to_save = ["x_zmmse"]
    return estimate_to_save


### GENERAL INFERENCE PARAMS CLASS ###

class InferenceParams(nn.Module):
    """
    General Inference Parameters class
    """

    def __init__(self, device: str = 'cpu'):
        super().__init__()
        self.device = device
        self.optimized_param_names = []
    
    def get_param_names(self):
        """
        Get the names of the optimized parameters
        """
        return self.optimized_param_names
    
    def get_running_estimates(self):
        """
        Get the running estimates of the parameters
        """
        running_estimates = {}
        with torch.no_grad():
            for name in self.optimized_param_names:
                # running_estimates[name] = getattr(self, f"get_{name}")(getattr(self, f"{name}_running_estimate").getval().to(self.device)).detach().cpu()
                running_estimates[name] = getattr(self, f"{name}_running_estimate").getval()
        return running_estimates
    
    def update_running_estimates(self, iter:int, max_iters:int, estimate_type:str, patience):
        """
        Update the running estimates of the parameters

        Parameters
        ----------
        iter: int, current iteration
        max_iters: int, maximum number of iterations
        estimate_type: str, type of estimate: last (last iteration), last100 (mean over the last 100 iterations), min (minimum loss)
        patience: int, number of iterations since the minimum loss has not been improved
        """
        if (max_iters - iter < 100 and estimate_type == 'last100') or (iter == max_iters and estimate_type == 'last'):
            for name in self.get_param_names():
                running_estimate = getattr(self, f"{name}_running_estimate")
                running_estimate.update(getattr(self, name).data.detach().cpu())
        elif estimate_type == 'min':
            if patience == 0:
                for name in self.get_param_names():
                    running_estimate = getattr(self, f"{name}_running_estimate")
                    running_estimate.reinit()
                    running_estimate.update(getattr(self, name).data.detach().cpu())
    

### LATENT INFERENCE PARAMS ###

class LatentInferenceParams(InferenceParams, ABC):

    """
    General latent inference parameters class
    """

    def __init__(self, latent_inference_model: str, optimize_h: bool, device: str = 'cpu', zdim: tuple = None):

        super().__init__(device)

        self.latent_inference_model = latent_inference_model
        self.optimize_h = optimize_h
        self.zdim = zdim

        assert len(self.zdim) == 1 or len(self.zdim) == 2, "only one and two latent variable models implemented"
        assert len(self.zdim) == 2 if self.optimize_h else True, "optimize_h only implemented for two latent variable models"

        self.zbar = nn.Parameter(torch.zeros(self.zdim[0], device=device), requires_grad=True)
        self.zbar_running_estimate = AverageSolution()
        self.optimized_param_names.append("zbar")

        if self.optimize_h:
            self.hbar = nn.Parameter(torch.zeros(self.zdim[1], device=device), requires_grad=True)
            self.hbar_running_estimate = AverageSolution()
            self.optimized_param_names.append("hbar")
        
    @abstractmethod
    def get_az(self, param=None):
        pass

    @abstractmethod
    def get_ah(self, param=None):
        pass

    def get_zbar(self, param=None):
        if param is None:
            return self.zbar
        else:
            return param
    
    def get_hbar(self, param=None):
        if param is None:
            return self.hbar
        else:
            return param

    @classmethod
    def instantiate_from_params(cls, params_dict, latent_inference_model: str, optimize_h: bool, device: str = 'cpu'):
        """
        Instantiate the class from a dictionary of parameters

        /!\ example_param corresponds to example_param.data
            example_param_value corresponds to self.get_example_param(example_param) 
        """
        zdim = [params_dict["zbar"].shape]
        if "hbar" in params_dict:
            zdim.append(params_dict["hbar"].shape)
        
        latentinferenceparams = LATENT_INFERENCE_MODELS[latent_inference_model](latent_inference_model, optimize_h, device, zdim)
        for k,v in params_dict.items():
            setattr(latentinferenceparams, k, nn.Parameter(v.to(device), requires_grad=True))
        return latentinferenceparams
    
    def init_from_paramvalues(self, init_dict):
        """
        Initialize the class from a dictionary of parameters values

        /!\ example_param corresponds to example_param.data
            example_param_value corresponds to self.get_example_param(example_param) 
        """
        LATENT_INFERENCE_MODELS[self.latent_inference_model].init_from_paramvalues(init_dict)
    
    def construct_from_running_estimates(self):
        """
        Construct the class from the running estimates
        """
        running_estimates = self.get_running_estimates()
        vblexzparams = LATENT_INFERENCE_MODELS[self.latent_inference_model].instantiate_from_params(running_estimates, self.latent_inference_model, self.optimize_h, self.device)
        return vblexzparams


class MeanVarLatentInferenceParams(LatentInferenceParams):

    """
    Latent inference parameters class for uniform and gaussian latent inference models
    """

    def __init__(self, latent_inference_model: str, optimize_h: bool, device: str = 'cpu', zdim: tuple = None):
        super().__init__(latent_inference_model, optimize_h, device, zdim)

        if latent_inference_model == "uniform":
            fact = 0
        elif latent_inference_model == "gaussian":
            fact = -3
        else:
            raise ValueError("latent_inference_model must be uniform or gaussian")
        self.az = nn.Parameter(fact*torch.ones(zdim[0], device=device), requires_grad=True)
        self.az_running_estimate = AverageSolution()
        self.optimized_param_names.append("az")    

        self.bound_z = init_lower_bound_z(vmin=1.5e-5, vmax=20, device=device)

        if self.optimize_h:
            self.ah = nn.Parameter(torch.zeros(zdim[1], device=device), requires_grad=True)
            self.ah_running_estimate = AverageSolution()
            self.optimized_param_names.append("ah")
    
    def get_az(self, param=None):
        if param is None:
            az = self.az
        else:
            az = param
        return torch.exp(self.bound_z(az))
    
    def get_ah(self, param=None):
        if param is None:
            ah = self.ah
        else:
            ah = param
        return torch.exp(self.bound_z(ah))

    def init_from_paramvalues(self, init_dict):
        """
        Initialize the class from a dictionary of parameters values

        /!\ example_param corresponds to example_param.data
            example_param_value corresponds to self.get_example_param(example_param) 
        """
        zdim = [init_dict["z_init"][0].shape]
        if len(zdim) == 2:
            zdim.append(init_dict["z_init"][1].shape)
                
        self.zbar.data = init_dict["z_init"][0].to(self.device)
        if self.optimize_h and len(zdim) == 2:
            self.hbar.data = init_dict["z_init"][1].to(self.device)
        if "a_init" in init_dict:
            self.az.data = torch.log(init_dict["a_init"][0]).to(self.device)
            if self.optimize_h and len(init_dict["a_init"]) == 2:
                self.ah.data = torch.log(init_dict["a_init"][1]).to(self.device)


class DiracLatentInferenceParams(LatentInferenceParams):
    """
    Latent inference parameters class for dirac latent inference model
    """

    def __init__(self, latent_inference_model: str, optimize_h: bool, device: str = 'cpu', zdim: tuple = None):
        super().__init__(latent_inference_model, optimize_h, device, zdim)

    def get_az(self, param=None):
        return None 
    
    def get_ah(self, param=None):
        return None 

    def init_from_paramvalues(self, init_dict):
        """
        Initialize the class from a dictionary of parameters values

        /!\ example_param corresponds to example_param.data
            example_param_value corresponds to self.get_example_param(example_param) 
        """
        zdim = [init_dict["z_init"][0].shape]
        if len(zdim) == 2:
            zdim.append(init_dict["z_init"][1].shape)
        
        self.zbar.data = init_dict["z_init"][0].to(self.device)
        if self.optimize_h and len(zdim) == 2:
            self.hbar.data = init_dict["z_init"][1].to(self.device)


LATENT_INFERENCE_MODELS = {
    "uniform": MeanVarLatentInferenceParams,
    "gaussian": MeanVarLatentInferenceParams,
    "dirac": DiracLatentInferenceParams
}


