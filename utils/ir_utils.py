import numpy as np
import torch
from scipy import signal
from scipy.interpolate import griddata
from skimage.metrics import structural_similarity
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from os.path import join

def get_datafit_loss(datafit_op, sigma: float, loss_type: str = 'l2', n_pixels: int = None):
    """
    Returns data fidelity loss function for image restoration

    Parameters
    ----------
    datafit_op: degradation operator A in y = Ax + n
    sigma: deviation of white Gaussian noise
    loss_type: l1, l2, ll (log-likelihood)
    n_pixels: number of pixels

    Returns
    -------
    datafit_loss: The datafidelity loss operator
    """

    MULT_FACTOR = 1
    N = n_pixels
    if loss_type == "l2":  # ||Ax - y||^2
        def datafit_loss(x, y):
            datafit_vec = (datafit_op(x) - y).pow(2)
            datafit = torch.sum(datafit_vec)
            return datafit / N
    elif loss_type == "l1":  # ||Ax - y||^2
        def datafit_loss(x, y):
            datafit_vec = (datafit_op(x) - y).abs()
            datafit = torch.sum(datafit_vec)
            return datafit / N
    elif loss_type == "ll":  # ||Ax - y||^2/(2sigma^2)
        if sigma == 0:  # if sigma=0, sub pixel noise is considered
            sigma = 4e-3
        sigma2 = sigma ** 2
        def datafit_loss(x, y):
            x = torch.clip(x, -0.1 , 1.1)
            datafit_vec = (datafit_op(x) - y).pow(2)
            datafit = torch.sum(datafit_vec) / (2 * sigma2)
            return MULT_FACTOR * datafit / N

    return datafit_loss


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

    elif problem == 'deblur-sisr':

        k = np.load(dico_params['kernel'])
        k = k / np.sum(k)
        kernel = torch.tensor(k).view((1, 1, *k.shape)).to(device)
        kernel = kernel.flip(-1).flip(-2).float()

        stride = dico_params['scale_factor']
        kernel_size = kernel.shape[1:]
        padding = [kernel_size[i] // 2 for i in range(1, len(kernel_size[1:])+1)]
        conv_to_use = torch.nn.Conv2d(x_size[0], x_size[0], kernel_size=kernel_size, stride=stride,
                                      padding=padding, groups=x_size[0], bias=False, padding_mode='reflect',
                                      device=device)
        conv_to_use.weight = torch.nn.Parameter(kernel)
        datafit_op = lambda x: conv_to_use(x)

    elif problem == 'deblur':
        if dico_params["kernel"] is not None:

            k = np.load(dico_params['kernel'])
            k = k / np.sum(k)
            kernel = torch.tensor(k).view((1, 1, *k.shape)).to(device)
        
        else:
            kernel = torch.tensor(gaussian_kernel(dico_params['kernel_size'], dico_params['kernel_std'], True)).to(device)
            kernel = kernel.view((1, 1, *kernel.shape)).float()

        # computing by FFT
        # ftm = p2o(kernel, x_size[1:])
        #
        # def tmp(ftm, x):
        #     if len(x.shape) == 3:
        #         x = torch.unsqueeze(x, dim=0)
        #     Fx = torch.fft.fftn(x, dim=(-2, -1))
        #     return torch.real(torch.fft.ifftn(ftm * Fx, dim=(-2, -1)))
        #
        # datafit_op = lambda x: tmp(ftm, x)

        # computing with convolution
        kernel = kernel.flip(-1).flip(-2).float()
        kernel = kernel.expand(x_size[0], -1, -1, -1)

        stride = 1
        kernel_size = kernel.shape[1:]
        padding = [kernel_size[i] // 2 for i in range(1, len(kernel_size[1:]) + 1)]
        conv_to_use = torch.nn.Conv2d(x_size[0], x_size[0], kernel_size=kernel_size, stride=stride,
                                        padding=padding, groups=x_size[0], bias=False, padding_mode='reflect',
                                        device=device)
        conv_to_use.weight = torch.nn.Parameter(kernel)
        datafit_op = lambda x: conv_to_use(x)

        # computing with deepinv
        # blur_op = Blur(kernel, padding='reflect', device=device)
        # plt.figure()
        # plt.imshow(kernel[0,0].detach().cpu().numpy(), cmap='gray')
        # plt.colorbar()
        # plt.axis(False)
        # plt.show()
        # blur_op = BlurFFT(x_size, kernel, device=device)
        # datafit_op = lambda x: blur_op(x.float())

    elif problem == 'sisr':

        from utils.cubic_downsampling import imresize
        datafit_op = lambda x: imresize(x, scale=1 / dico_params['scale_factor'], antialiasing=True)

    else:
        raise NotImplementedError
    return datafit_op


def compute_measurement(x_target: torch.Tensor, datafit_op, problem: str, sigma: float) -> torch.Tensor:
    """
    Compute y in the forward model y = A(x) + w with w standard white Gaussian noise

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


def get_xinit(problem: str, y: torch.Tensor, device: str, x_size=None, psf_path=None):
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
    elif problem == 'sisr' or problem == "deblur-sisr":
        xinit = resize(y, [x_size[-2], x_size[-1]], interpolation=InterpolationMode.BICUBIC, antialias=True)
    else:
        raise NotImplementedError
    return xinit


def get_metrics(x: np.array, xtarget: np.array, lpips_fn=None, device='cpu', border: int = 0, return_metrics_maps=False):
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

    if border > 0:
        x = x[:, border:-border, border:-border]
        xtarget = xtarget[:, border:-border, border:-border]
    mse = np.mean(np.square(x - xtarget))
    psnr = 10 * (np.log(1. / mse) / np.log(10))
    ssim, ssim_img = structural_similarity(xtarget, x, data_range=1,
                                 channel_axis=0, gaussian_weights=True, sigma=1.5, use_sample_covariance=True,
                                 win_size=11, full=True)
    metrics_maps = {}
    if return_metrics_maps:
        metrics_maps['SSIM'] = ssim_img
        metrics_maps['rMSE'] = np.abs(x - xtarget)
    

    if lpips_fn is not None:
        lpips_score = compute_lpips(x, xtarget, lpips_fn, device=device)
    else:
        lpips_score = None
    if return_metrics_maps:
        return mse, psnr, ssim, lpips_score, metrics_maps
    return mse, psnr, ssim, lpips_score


def compute_and_add_metrics_to_dict(x_target, x, lpips_fn, device, dico_metrics, suffix='', append=True, border=1):
    mse, psnr, ssim, lpips_score = get_metrics(x, x_target, lpips_fn=lpips_fn, device=device, border=border)
    if append:
        dico_metrics[f'PSNR{suffix}'].append(psnr)
        dico_metrics[f'MSE{suffix}'].append(mse)
        dico_metrics[f'SSIM{suffix}'].append(ssim)
        dico_metrics[f'LPIPS{suffix}'].append(lpips_score)
    else:
        dico_metrics[f'PSNR{suffix}'] = psnr
        dico_metrics[f'MSE{suffix}'] = mse
        dico_metrics[f'SSIM{suffix}'] = ssim
        dico_metrics[f'LPIPS{suffix}'] = lpips_score
    return dico_metrics


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
        lpips_score = lpips_fn(xopt_pytorch.float(), xtarget_pytorch.float())
        lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
    return lpips_score


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
