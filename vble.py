import logging
import torch
import torch.nn as nn
from torch import optim
import numpy as np

from compressai_models.ops.bound_ops import LowerBound
from utils.ir_utils import get_metrics


def vble(y: torch.Tensor, datafit_op, priornet: nn.Module, zdim, lamb: float = 0.1, max_iters:int = 5000,
         lr: float = 1e-2, xtarget: torch.Tensor = None, zinit=None, device='cuda',
         verbose=False, stochastic=True, gd_final_value: str = 'last', optimizer_name: str = 'adam',
         clip_grad_norm: float = -1.) -> dict:
    """
    VBLE algorithm (optimization of the approximated posterior + posterior sampling) to solve y = Ax + w

    Parameters
    ----------
    y: measurement matrix (1,C,H,W)
    datafit_op: datafit operator A : x -> y
    priornet: object of class PriorNN, contains the generative/compressive autoencoder used for regularization
    zdim: dimension of the latent variable(s)
    lamb: regularization parameter
    max_iters: number of gradient descent iterations
    lr: learning rate of gradient descent
    xtarget: ground truth solution of the inverse problem (to compute running metrics)
    zinit: initial latent vector to start the gradient descent with
    device: cuda or gpu
    verbose: True for detailed prints
    stochastic: True for stochastic (VBLE algo) gradient descent, False for deterministic (Map-z)
    gd_final_value: last/last100/min. last: z final is the last z of gradient descent. last100: z final is an average
                    of the last 100 iterations. min: z final corresponds to the z with the minimum loss during the
                    iterations
    optimizer_name: adam or sgd
    clip_grad_norm: -1 for no gradient clipping, else gradient clipping value
    """

    logger = logging.getLogger('logfile')

    dico_loss = {'loss': [], 'datafit': [], 'reg': [], 'lamb': [], 'mse': [], 'psnr': [], 'ssim': [],
                 'loss_diff': [], 'min_loss': [], 'patience': [], 'min_dist': [], 'min_dist_norm':[]}

    # Initialization
    if zinit is None:
        zinit = [torch.zeros(zdim[i], device=device) for i in range(len(zinit))]
    n_latent_variables = len(zinit)
    zk = [zinit[i].requires_grad_(True) for i in range(n_latent_variables)]

    get_a_from_params, zk = get_a_from_params_func(stochastic, zk, zinit, zdim, device)  # new zk = [zk, ak] if stochastic

    last_loss = 0
    patience = 0
    min_loss = 1e8

    convergence = False
    k = 0  # current iteration number

    # Optimizer

    if optimizer_name == 'adam':
        optimizer = optim.Adam(zk, lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = optim.SGD(zk, lr=lr, momentum=0.95, nesterov=True)
    else:
        raise NotImplementedError

    zk_avg = AverageSolution()
    if stochastic:
        ak_avg = AverageSolution()
    else:
        ak_avg = None
    if n_latent_variables == 2:  # optimize on (z,h)
        hk_avg = AverageSolution()
        if stochastic:
            ahk_avg = AverageSolution()
        else:
            ahk_avg = None
    else:
        hk_avg = None
        ahk_avg = None

    while not convergence and k<max_iters:

        k += 1
        optimizer.zero_grad()

        # loss computation + backward
        datafit, reg = ir_loss(zk, y, datafit_op, priornet, n_latent_variables, stochastic, get_a_from_params)
        loss = datafit + lamb*reg
        loss.backward()

        # clip grad norm for better convergence
        if clip_grad_norm != -1:
            _ = torch.nn.utils.clip_grad_norm_(zk, clip_grad_norm)
        optimizer.step()

        # save running values
        dico_loss['loss'].append(loss.item())
        dico_loss['loss_diff'].append(np.abs(loss.item() - last_loss))
        dico_loss['datafit'].append(datafit.item())
        dico_loss['reg'].append((lamb*reg).item())
        dico_loss['lamb'].append(lamb)
        last_loss = loss.item()
        if loss.item() < min_loss and k > 10:
            min_loss = loss.item()
            patience = 0
        else:
            patience += 1
        dico_loss['min_loss'].append(min_loss)
        dico_loss['patience'].append(patience)

        # compute z,a running estimates
        update_z_and_h_estimate(gd_final_value, max_iters, k, stochastic, n_latent_variables, patience, get_a_from_params,
                                zk, zk_avg, hk_avg, ak_avg, ahk_avg)

        # print functions
        if k % 10 == 0:
            if verbose:
                if stochastic:
                    logger.info(f'Iter {k} -- datafit {datafit.item()} -- reg {reg.item()} -- '
                                f'min c {torch.min(zk[1]).item():.3f} max c {torch.max(zk[1]).item():.3f}')
                else:
                    logger.info(f'Iter {k} -- datafit {datafit.item()} -- reg {reg.item()}')

                if xtarget is not None:
                    with torch.no_grad():
                        xk = priornet.decoder(zk[:n_latent_variables])["x_rec"]
                        mse, psnr, ssim, _ = get_metrics(xk[0].detach().cpu().numpy(),
                                                         xtarget[0].detach().cpu().numpy(),
                                                         lpips_fn=None, device=device)
                        dico_loss['mse'].append(mse)
                        dico_loss['psnr'].append(psnr)
                        dico_loss['ssim'].append(ssim)
                    logger.info(f'Iter {k} -- PSNR {psnr} -- SSIM {ssim}')

    if verbose:
        logger.info('Adam terminated in %d iterations' % k)

    # z mmse estimate for VBLE, mapz estimate for MAP-z
    xk_zmmse = priornet.decoder(zk[:n_latent_variables])["x_rec"].detach().cpu()

    if stochastic:  # compute MMSE-x estimate for VBLE

        a = ak_avg.getval().to(device)
        zk = [zk_avg.getval().to(device)]
        if n_latent_variables == 2:
            zk = zk + [hk_avg.getval().to(device)]
            a = [a, ahk_avg.getval().to(device)]

        xks = []
        post_sampling_bs = 16
        n_loop_post_sampling = 100 // post_sampling_bs
        with torch.no_grad():
            zk_expanded = [zk[i].expand((post_sampling_bs,-1,-1,-1)) for i in range (len(zk))]
            if n_latent_variables == 2:
                a_expanded = [a[i].expand((post_sampling_bs, -1, -1, -1)) for i in range(len(a))]
            else:
                a_expanded = a.expand((post_sampling_bs,-1,-1,-1))
            for i in range(n_loop_post_sampling):
                xaux = priornet.decoder(zk_expanded[:n_latent_variables], latent_sampling=True, noise_factor=a_expanded)["x_rec"]
                xks.append(xaux.cpu())
        xks = torch.cat(xks, dim=0)
        xk_xmmse_std = torch.std(xks, dim=0, keepdim=True).detach().cpu()
        xk_xmmse = torch.mean(xks, dim=0, keepdim=True).detach().cpu()
        x_samples = xks[:2]
        zk = [zk[i].detach().cpu() for i in range(len(zk))]

        if n_latent_variables == 1:
            a = a.detach().cpu()
        elif n_latent_variables == 2:
            a = [a[0].detach().cpu(), a[1].detach().cpu()]

    else:
        xk_xmmse_std = None
        xk_xmmse = None
        a = None
        x_samples = None

    return {
        "zk": zk,
        "dico_loss": dico_loss,
        "x_zmmse": xk_zmmse,
        "x_xmmse_std": xk_xmmse_std,
        "x_xmmse": xk_xmmse,
        "c": a,
        "x_samples": x_samples
    }


def ir_loss(z: torch.Tensor, y: torch.Tensor, datafit_op, priornet: nn.Module, n_latent_variables: int = 1,
            latent_sampling=False, get_a_from_params=None) -> (float, float):
    """
    To compute the loss optimized for the inverse problem : L = ||A D(z) - y||^2 + lambda R(z)

    Parameters
    ----------
    z: the current z estimate (1,C,H,W)
    y: the measurement matrix
    datafit_op: degradation operator A
    priornet: instance of PriorNN, the AE used for regularization
    n_latent_variables: int, number of latent variables optimized
    latent_sampling: True if VBLE, False for deterministic gradient descent
    get_a_from_params: function zk -> ak the current variance parameters estimate
    """

    a = get_a_from_params(z)
    z = z[:n_latent_variables]

    ## Datafit = ||A*G(z) - y||^2
    out_decoder = priornet.decoder(z, latent_sampling=latent_sampling, noise_factor=a)
    out_decoder.update({"a": a})

    Datafit_vec = (datafit_op(out_decoder["x_rec"]) - y).pow(2)
    Datafit = torch.sum(Datafit_vec)

    Reg = priornet.reg_func(out_decoder)

    return Datafit, Reg


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


def update_z_and_h_estimate(zk_type, max_iters, k, stochastic, n_latent_variables, patience, get_c_from_params,
                            zk, zk_avg, hk_avg=None, ak_avg=None, ahk_avg=None):
    """
    Auxiliary function to update zk, ak (hk if optimize_h) estimates according to zk_type
    """
    if (max_iters - k < 100 and zk_type == 'last100') or (k == max_iters and zk_type == 'last'):
        zk_avg.update(zk[0].detach().cpu())
        if n_latent_variables == 2:
            hk_avg.update(zk[1].detach().cpu())
        if stochastic:
            if n_latent_variables == 1:
                ak_avg.update(get_c_from_params(zk).detach().cpu())
            elif n_latent_variables == 2:
                c_cur = get_c_from_params(zk)
                ak_avg.update(c_cur[0])
                ahk_avg.update(c_cur[1])
    elif zk_type == 'min':
        if patience == 0:
            zk_avg.reinit()
            zk_avg.update(zk[0].detach().cpu())
            if n_latent_variables == 2:
                hk_avg.reinit()
                hk_avg.update(zk[1].detach().cpu())

            if stochastic:
                if n_latent_variables == 1:
                    ak_avg.reinit()
                    ak_avg.update(get_c_from_params(zk).detach().cpu())
                elif n_latent_variables == 2:
                    c_cur = get_c_from_params(zk)
                    ak_avg.reinit()
                    ahk_avg.reinit()
                    ak_avg.update(c_cur[0])
                    ahk_avg.update(c_cur[1])

    return zk_avg, hk_avg, ak_avg, ahk_avg


def get_a_from_params_func(stochastic, zk, zinit, zdim, device):
    lb = LowerBound(1e-4).to(device)
    if stochastic:
        if len(zk) == 1:  # optimization only on z
            zk = zinit + [torch.ones(zdim[0]).to(device).requires_grad_(True)]
            get_a_from_params = lambda zk: lb(zk[-1])
        elif len(zk) == 2:  # optimization on (z,h)
            zk = zinit + [torch.ones(zdim[i]).to(device).requires_grad_(True) for i in range(2)]
            get_a_from_params = lambda zk: [lb(zk[i]) for i in [-2, -1]]
        else:
            raise NotImplementedError
    else:
        def get_a_from_params(zk):
            return None
    return get_a_from_params, zk