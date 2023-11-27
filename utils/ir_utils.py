import numpy as np
import torch
from scipy import signal
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize


def gaussian_kernel(kernel_size: int, std: float, normalised=False) -> np.array:
    """
    Create a Gaussian kernel

    Parameters
    ----------
    kernel_size: size of the kernel
    std: standard deviation of the kernel
    normalised: True to normalise the final kernel
    """
    gaussian1D = signal.gaussian(kernel_size, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    if normalised:
        gaussian2D /= (2 * np.pi * (std ** 2))
    return gaussian2D


def get_datafit_operator(problem: str, device: str, dico_params: dict, x_size: torch.Size):
    """
    Get the operator A in the forward model y = A(x) + w.

    Parameters
    ----------
    problem: inverse problem type (inpainting, denoising, deblur, sisr supported)
    device: cpu or cuda
    dico_params: dictionary of parameters
    x_size: size CHW of the image
    """
    if problem == 'inpainting':
        if dico_params["mask"] is not None:
            mask = np.load(dico_params["mask"])
            mask = torch.tensor(mask).view((1, *mask.shape))
        else:
            mask = torch.rand((1, *x_size[1:])) > dico_params["proba_missing"]
        mask = mask.to(device)
        datafit_op = lambda x: mask * x

    elif problem == 'denoising':
        datafit_op = lambda x: x

    elif problem == 'deblur':
        if dico_params["kernel"] is not None:
            k = np.load(dico_params['kernel'])
            kernel = torch.tensor(k).view((1, 1, *k.shape)).to(device)

            ftm = p2o(kernel, x_size[1:])

            def tmp(ftm, x):
                if len(x.shape) == 3:
                    x = torch.unsqueeze(x, dim=0)
                Fx = torch.fft.fftn(x, dim=(-2, -1))
                return torch.real(torch.fft.ifftn(ftm * Fx, dim=(-2, -1)))

            datafit_op = lambda x: tmp(ftm, x)

        else:
            kernel = torch.tensor(gaussian_kernel(dico_params['kernel_size'], dico_params['kernel_std'], True)).to(
                device)
            kernel = kernel.view((1, 1, *kernel.shape)).float()
            kernel = kernel.expand((x_size[0], -1, -1, -1))

            def tmp(kernel, x):
                datafit = torch.nn.functional.conv2d(x, kernel, padding=(kernel.shape[-2] // 2, kernel.shape[-1] // 2),
                                                     groups=x_size[0])
                return datafit

            datafit_op = lambda x: tmp(kernel, x)
    elif problem == 'sisr':

        from utils.cubic_downsampling import imresize
        datafit_op = lambda x: imresize(x, scale=1 / dico_params['scale_factor'], antialiasing=True)

    else:
        raise NotImplementedError
    return datafit_op


def compute_measurement(x_target: torch.Tensor, datafit_op, problem: str, sigma: float) -> torch.Tensor:
    """
    Compute y in the forward model y = A(x) + w

    Parameters
    ----------
    x_target: tensor shape=(1,C,H,W), x in y = A(x) + w
    datafit_op: function A
    problem: inverse problem considered (denoising, deblur, sisr, inpainting)
    sigma: noise w in [0,1]
    """
    y_size = datafit_op(x_target[:1]).shape[1:]  # (C,H,W)
    x_noisy = torch.zeros(x_target.shape[0], *y_size)
    with torch.no_grad():
        for i in range(x_target.shape[0]):
            x_noisy[i, :] = datafit_op(x_target[i])
    if problem in ['inpainting', 'inpainting_block']:
        mask = x_noisy != 0
        x_noisy += sigma * torch.randn(*x_noisy.shape) * mask
    else:
        x_noisy += sigma * torch.randn(*x_noisy.shape)
    return x_noisy


def get_xinit(problem: str, y: torch.Tensor, device: str, x_size=None):
    """
    Get x initialization for each inverse problem. Then, zinit = E(xinit).

    Parameters
    ----------
    problem: inverse problem considered (denoising, deblur, sisr, inpainting)
    y: measurement
    device: cuda or cpu
    x_size: size of the images (C,H,W)
    """
    if problem == 'denoising' or problem == 'deblur':
        xinit = torch.clone(y)
    elif problem == 'inpainting' or problem == 'inpainting_block':
        x_shape = y.shape
        xinit = interpolate_init_inpainting(y.detach().cpu().numpy())
        xinit = torch.tensor(xinit).view(x_shape).to(device)
    elif problem == 'sisr':
        xinit = resize(y, [x_size[-2], x_size[-1]], interpolation=InterpolationMode.BICUBIC, antialias=True)
    else:
        raise NotImplementedError
    return xinit


def get_metrics(x: np.array, xtarget: np.array, lpips_fn=None, device='cpu', border: int = 0):
    """
    Get MSE, PSNR, SSIM, and LPIPS (if lpips_fn is not None) metrics for x compared to x_target

    Parameters
    ----------
    x: image, shape=(C,H,W)
    xtarget: target image, shape=(C,H,W)
    lpips_fn: function to compute LPIPS
    device: cpu or cuda
    border: to compute metrics on center cropped images
    """
    x = x[:, border:-(border + 1), border:-(border + 1)]
    xtarget = xtarget[:, border:-(border + 1), border:-(border + 1)]
    mse = np.mean(np.square(x - xtarget))
    psnr = 10 * (np.log(1. / mse) / np.log(10))
    ssim = structural_similarity(xtarget, x, data_range=1,
                                 channel_axis=0, gaussian_weights=True, sigma=1.5, use_sample_covariance=True,
                                 win_size=11)
    if lpips_fn is not None:
        lpips_score = compute_lpips(x, xtarget, lpips_fn, device=device)
    else:
        lpips_score = None
    return mse, psnr, ssim, lpips_score


def interpolate_init_inpainting(x: np.array) -> np.array:
    """
    Image interpolation for inpainting initialization

    Parameters
    ----------
    x : 2D image (1,C,W,H) with holes
    """
    xshape = x.shape
    for i in range(xshape[1]):
        mask_values = x[0, i] != 0
        X, Y = np.mgrid[0:xshape[2], 0:xshape[3]]
        x_values = X[mask_values]
        y_values = Y[mask_values]
        img_values = x[0, i][mask_values]
        interpolation = griddata((x_values, y_values), img_values, (X, Y), method='linear', fill_value=0.5)
        x[0, i] = interpolation
    return x


def p2o(psf: torch.Tensor, shape) -> torch.Tensor:
    """
    Compute PSF FFT

    Parameters
    ----------
    psf: PSF shape=(1,C,K,K')
    shape: spatial size of the desired FFT (K2,K2')
    """
    otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    otf = torch.fft.fftn(otf, dim=(-2, -1))
    return otf


def compute_lpips(xopt: np.array, xtarget: np.array, lpips_fn, device: str) -> float:
    """
    Compute LPIPS metrics

    Parameters
    ----------
    xopt: np.array (C,H,W)
    xtarget: np.array (C,H,W)
    lpips_fn: function to compute LPIPS
    device: cpu or cuda
    """

    xopt_pytorch = 2 * torch.unsqueeze(torch.tensor(xopt).to(device), dim=0) - 1
    xtarget_pytorch = 2 * torch.unsqueeze(torch.tensor(xtarget).to(device), dim=0) - 1

    with torch.no_grad():
        lpips_score = lpips_fn(xopt_pytorch, xtarget_pytorch)
        lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
    return lpips_score
