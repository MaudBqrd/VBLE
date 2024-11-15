import logging
import torch
import torch.nn as nn
from torch import optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time

from utils.ir_utils import get_metrics
from priornn import LATENT_INFERENCE_MODELS


def vble(y: torch.Tensor, datafit_loss, priornet: nn.Module, zdim, lamb: float = 1, max_iters:int = 300,
         lr: float = 0.1, xtarget: torch.Tensor = None, zinit=None, device='cuda',
         verbose=False, latent_inference_model:str = "dirac", gd_final_value: str = 'last', optimizer_name: str = 'adam',
         clip_grad_norm: float = -1., post_sampling_bs: int = 4, n_samples_sgvb: int = 1,
         use_scheduler: bool = False, n_samples_to_save:int = 2) -> dict:
    """
    VBLE algorithm (optimization of the approximated posterior + posterior sampling) to solve y = Ax + w

    Parameters
    ----------
    y: measurement matrix (1,C,H,W)
    datafit_loss: datafit loss <-> - log p(y|x)
    priornet: object of class PriorNN, contains the generative/compressive autoencoder used for regularization
    zdim: dimension of the latent variable(s)
    lamb: regularization parameter
    max_iters: number of gradient descent iterations
    lr: learning rate of gradient descent
    xtarget: ground truth solution of the inverse problem (to compute running metrics)
    zinit: initial latent vector to start the gradient descent with
    device: cuda or gpu
    verbose: True for detailed prints
    latent_inference_model: 'uniform', 'gaussian' or 'dirac'
    gd_final_value: last/last100/min. last: z final is the last z of gradient descent. last100: z final is an average
                    of the last 100 iterations. min: z final corresponds to the z with the minimum loss during the
                    iterations
    optimizer_name: adam or sgd
    clip_grad_norm: -1 for no gradient clipping, else gradient clipping value
    post_sampling_bs: batch size for posterior sampling
    n_samples_sgvb: number of sample for SGVB estimate computation
    use_scheduler: True to use a scheduler
    n_samples_to_save: number of posterior sampling estimate to save (from 0 to 100
    """

    logger = logging.getLogger('logfile')

    dico_loss = {'loss': [], 'datafit': [], 'kl_z': [], 'psnr': [], 'ssim': [],
                 'loss_diff': [], 'min_loss': [], 'patience': []}


    # Initialization
    optimize_h = len(zdim) == 2
    if zinit is None:
        zinit = [torch.zeros(zdim[i], device=device) for i in range(len(zinit))]
    inference_params = LATENT_INFERENCE_MODELS[latent_inference_model](latent_inference_model, optimize_h, device, zdim)
    inference_params.init_from_paramvalues({"z_init": zinit})

    last_loss = 0
    patience = 0
    min_loss = 1e8

    convergence = False
    k = 0  # current iteration number

    # Optimizer
    if optimizer_name == 'adam':
        optimizer = optim.Adam(inference_params.parameters(), lr=lr)
        scheduler = None
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(inference_params.parameters(), lr=lr, momentum=0.9, nesterov=True)
        if use_scheduler:
            scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True, patience=30, min_lr=0.5 * lr, threshold=0)
        else:
            scheduler = None
    else:
        raise NotImplementedError
    

    while not convergence and k<max_iters:
        k += 1

        optimizer.zero_grad()

        # loss computation + backward
        datafit, kl_z = ir_loss(inference_params, y, datafit_loss, priornet, n_samples_sgvb=n_samples_sgvb)
        loss = datafit + lamb*kl_z
        loss.backward()

        # clip grad norm for better convergence
        if clip_grad_norm != -1:
            _ = torch.nn.utils.clip_grad_norm_(inference_params.parameters(), clip_grad_norm)
        optimizer.step()
        if scheduler is not None and k > 1:
            scheduler.step(loss.item())

        # save running values
        dico_loss['loss'].append(loss.item())
        dico_loss['loss_diff'].append(np.abs(loss.item() - last_loss))
        dico_loss['datafit'].append(datafit.item())
        dico_loss['kl_z'].append(kl_z.item())
        last_loss = loss.item()
        if loss.item() < min_loss and k > 10:
            min_loss = loss.item()
            patience = 0
        else:
            patience += 1
        dico_loss['min_loss'].append(min_loss)
        dico_loss['patience'].append(patience)

        # compute z,a running estimates
        inference_params.update_running_estimates(k, max_iters, gd_final_value, patience)

        # print functions
        if k % 10 == 0 and verbose:
            str_to_plot = ''
            str_to_plot += f'Iter {k} -- datafit {datafit.item():.4f} -- KL_z {kl_z.item():.4f}'

            if xtarget is not None:
                with torch.no_grad():
                    xk = priornet.sample_vble_model(inference_params, sampling=False)["x_rec"]

            if xtarget is not None:
                _, psnr, ssim, _, = get_metrics(xk[0].detach().cpu().numpy(),
                                                 xtarget[0].detach().cpu().numpy(),
                                                 lpips_fn=None, device=device, border=5)
                dico_loss['psnr'].append(psnr)
                dico_loss['ssim'].append(ssim)
                str_to_plot += f'-- PSNR {psnr:.2f} -- SSIM {ssim:.4f}'

            if verbose:
                logger.info(str_to_plot)

    if verbose:
        logger.info('Optim terminated in %d iterations' % k)

    # construct final inference params with averaged running estimates
    final_inference_params = inference_params.construct_from_running_estimates()
    final_inference_params = final_inference_params.to(device)

    # compute MMSE-z estimate
    xopt_zmmse = priornet.sample_vble_model(final_inference_params, sampling=False)["x_rec"].detach().cpu()

    # sampling of 100 posterior samples
    n_loop_post_sampling = 100 // post_sampling_bs
    add_post_sampling = 100 % post_sampling_bs
    xopt_samples = []
    with torch.no_grad():
        for _ in range(n_loop_post_sampling):
            xaux = priornet.sample_vble_model(final_inference_params, sampling=True, n_samples=post_sampling_bs)["x_rec"]
            xopt_samples.append(xaux.cpu())
        if add_post_sampling > 0:
            xaux = priornet.sample_vble_model(final_inference_params, sampling=True, n_samples=add_post_sampling)["x_rec"]
            xopt_samples.append(xaux.cpu())
    xopt_samples = torch.cat(xopt_samples, dim=0)

    # compute MMSE-x estimate and marginal deviations
    xopt_xmmse_std = torch.std(xopt_samples, dim=0, keepdim=True).detach().cpu()
    xopt_xmmse = torch.mean(xopt_samples, dim=0, keepdim=True).detach().cpu()

    xopt_samples = xopt_samples[:n_samples_to_save]
    return {
        "inference_params": inference_params,
        "dico_loss": dico_loss,
        "x_zmmse": xopt_zmmse,
        "x_xmmse_std": xopt_xmmse_std,
        "x_xmmse": xopt_xmmse,
        "x_samples": xopt_samples
    }


def ir_loss(inference_params: nn.Module, y: torch.Tensor, datafit_loss, priornet: nn.Module, n_samples_sgvb: int = 1) -> tuple:
    """
    To compute the loss optimized for the inverse problem : L = ||A D(z) - y||^2 + lambda R(z)

    Parameters
    ----------
    inference_params: the current inference parameters which are being optimized
    y: the measurement matrix
    datafit_loss: loss associated to the degradation operator A
    priornet: instance of PriorNN, the AE used for regularization
    """
    # Run VBLE inference model
    out_decoder = priornet.sample_vble_model(inference_params, sampling=True, n_samples=n_samples_sgvb)

    # Compute losses
    Datafit = datafit_loss(out_decoder["x_rec"], y) / n_samples_sgvb
    kl_z = priornet.compute_kl_z(inference_params, out_decoder)

    return Datafit, kl_z
