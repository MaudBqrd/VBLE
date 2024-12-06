import os
from os.path import join
import argparse
import yaml
import lpips
from torchvision.transforms.functional import resize, InterpolationMode
import logging

from vble import vble
from utils.utils import save_experiment_params, make_experiment_folder, write_metrics_summary, configure_logger, save_as_png, get_normalized_std, plot_losses, log_metrics
from utils.ir_utils import get_datafit_operator, compute_measurement, get_xinit, get_metrics, get_datafit_loss, compute_and_add_metrics_to_dict
from utils.datasets import load_dataset
from priornn import get_priornn_module, get_estimate_name_to_save
import torch
import numpy as np
import time

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--experiment_name', type=str, help='experiment name')
parser.add_argument('--latent_inference_model', type=str, default='uniform', help='uniform, gaussian, or dirac. Type of latent parametric distribution of VBLE. "dirac" corresponds to MAP-z approach. "uniform" is recommended for VBLE with CAEs, while "gaussian" is recommended for VBLE with VAEs')

parser.add_argument('--config', type=str, default='none', help="path to a yml file to overwrite this parser")
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--verbose', action='store_true', default=False)

# Data loading parameters
parser.add_argument('--target_image_root', type=str, help="path to target images (either target or degraded image root shall be specified)", default=None)
parser.add_argument('--degraded_image_root', type=str, help="path to degraded images (either target or degraded image root shall be specified)", default=None)
parser.add_argument('--dataset_name', type=str, default=None, help="dataset name for specific preprocessing (implemented: mnist, celeba)")
parser.add_argument('--n_samples', type=int, default=-1, help="number of images to restore. -1 to restore all dataset images")

parser.add_argument('--patch_size', type=int, default=None, help="if specified, center crop the original images to the specified patch size")
parser.add_argument('--n_bits', type=int, default=8)

# Inverse problem settings
parser.add_argument('--problem', type=str, help='deblur, sisr, inpainting or denoising')
parser.add_argument('--sigma', type=float, default=0, help="std of a white Gaussian noise for the inverse problems in [0,2**n_bits-1]")

parser.add_argument('--kernel', type=str, default='filters/', help="access to a blur kernel .npy format")
parser.add_argument('--kernel_std', type=float, default=1, help="Gaussian noise std, overwritten by kernel if kernel is not None")
parser.add_argument('--kernel_size', type=int, default=5, help="Size of the Gaussian kernel, overwritten by kernel if kernel is not None")

parser.add_argument('--scale_factor', type=int, default=1, help="Scale factor for SISR")
parser.add_argument('--mask', type=str, default='filters/', help="Mask path for inpainting, .npy format")
parser.add_argument('--proba_missing', type=float, default=0.5)

# Model settings
parser.add_argument('--model', type=str, help="path to model checkpoint")
parser.add_argument('--model_type', type=str, choices=["mbt", "cheng", "1lvae-vanilla", "1lvae-vanilla-fcb", "1lvae-vanilla-resnet"], help="model type")

# Optimization parameters
parser.add_argument('--lamb', type=float, default=1, help="Regularization parameter. Lambda=1 correpsonds to the Bayesian framework")
parser.add_argument('--lr', type=float, default=0.1, help="learning rate of VBLE. Using 0.1 for adam and 50 for sgd generally works")
parser.add_argument('--max_iters', type=int, default=300, help="number of iterations")
parser.add_argument('--optimizer_name', type=str, default='adam', help="adam or sgd, optimizer type for gradient descent")  # adam or sgd
parser.add_argument('--datafit_loss_type', type=str, default='ll', help='Type of data fidelity loss for VBLE. l1 / l2 / ll (likelihood)')
parser.add_argument('--n_samples_sgvb', type=int, default=1, help="Number of samples for computing the SGVB estimate at each iteration. If instability during VBLE optimization, increase this number")
parser.add_argument('--posterior_sampling_batch_size', type=int, default=4)


args = parser.parse_args()
if args.config == "none":
    dico_params = vars(args)
else:
    dico_params = vars(args)
    with open(args.config, 'r') as f:
        dico_params_config = yaml.safe_load(f)
    for k in ['target_image_root', 'degraded_image_root', 'model', 'model_type', 'experiment_name', 'dataset_name']:
        if k in dico_params and dico_params[k] is not None:
            dico_params_config[k] = dico_params[k]
    dico_params.update(dico_params_config)

# experimental setup
exp_folder = join('experiments', dico_params['problem'], dico_params['experiment_name'])
make_experiment_folder(exp_folder)
model_path = dico_params["model"]
configure_logger(exp_folder)
device = 'cuda' if dico_params['cuda'] else 'cpu'

# Load images (from test dataset)
n_bits = dico_params["n_bits"]
n_pixel_values = 2**n_bits - 1
test_dataset = load_dataset(dico_params["target_image_root"], dico_params["degraded_image_root"], dico_params["scale_factor"], dico_params["patch_size"], n_bits, dico_params["dataset_name"])

if dico_params["n_samples"] == -1:
    dico_params["n_samples"] = len(test_dataset)

save_experiment_params(exp_folder, dico_params)

x_target, y = test_dataset[0]

# Compute size of the image and number of pixels
if x_target is not None:
    x_size = x_target.shape  # CHW
elif y is not None:
    x_size = (y.shape[0], dico_params["scale_factor"]*(y.shape[1]), dico_params["scale_factor"]*(y.shape[2]))
else:
    raise ValueError
n_channels = x_size[0]
n_pixels = np.prod(x_size[1:])

lpips_fn = lpips.LPIPS(net='vgg').to(device)

# Degradation operator initialisation

sigma = dico_params["sigma"] / n_pixel_values
datafit_op = get_datafit_operator(dico_params['problem'], device, dico_params, x_size)
datafit_loss = get_datafit_loss(datafit_op, sigma, dico_params["datafit_loss_type"], n_pixels)

# Load model
priornet = get_priornn_module(model_path, dico_params["model_type"], device, n_channels, dico_params['latent_inference_model'], optimize_h=True)
latent_dim = priornet.get_latent_dim(x_size)

estimate_name_to_save = get_estimate_name_to_save(**dico_params)
x_classes = []
dico_metrics = {}
for estim in estimate_name_to_save:
    if "samples" not in estim and "std" not in estim:
        for metrics in ["MSE", "PSNR", "SSIM", "LPIPS"]:
            dico_metrics[f"{metrics}_{estim.split('x_')[-1]}"] = []
with open(join(exp_folder, 'metrics.txt'), 'w') as f:
    f.write(f"Metrics for each image\n\n")


for ind in range(min(dico_params['n_samples'], len(test_dataset))):

    if dico_params['seed'] != -1:
        torch.manual_seed(dico_params['seed'])
        np.random.seed(dico_params['seed'])

    # Load target image if available, load or compute degraded image
    x_target, y = test_dataset[ind]  
    if x_target is not None:
        x_target = x_target.to(device).unsqueeze(dim=0)  # 1CHW
    if y is not None:
        y = y.to(device).unsqueeze(dim=0)  # 1CHW
    else:
        assert x_target is not None, "Either degraded image root or target image root must be specified"
        y = compute_measurement(x_target, datafit_op, dico_params['problem'], sigma).to(device)  # 1, C, H', W'
        y_size = y.shape[1:]  # C, H', W'
    if dico_params['problem'] == 'sisr':
        ytoprint = resize(y.view(1, *y_size).detach().cpu(), [x_size[-2], x_size[-1]],
                          interpolation=InterpolationMode.NEAREST)
    else:
        ytoprint = y.detach().cpu()

    # algorithm initialization
    logger = logging.getLogger('logfile')
    logger.info(f"Running restoration {ind + 1} out of {dico_params['n_samples']}")
    t0 = time.time()
    with torch.no_grad():
        xinit = get_xinit(dico_params['problem'], y, device, x_size)  # 1CHW
    if x_target is not None:
        mse_init, psnr_init, ssim_init, lpips_score_init = get_metrics(xinit[0].detach().cpu().numpy(),
                                                                       x_target[0].detach().cpu().numpy(),
                                                                       lpips_fn=lpips_fn, device=device, border=5)
        logger.info(f'(init) Image {ind} : PSNR {psnr_init} -- SSIM {ssim_init} -- LPIPS {lpips_score_init}')
    with torch.no_grad():
        out_encoder = priornet.net_encoder(xinit)
        out_xinit = priornet.net_decoder(out_encoder)
        out_xinit.update(out_encoder)
        zinit = [out_xinit["z"]]
        if "h" in out_xinit:
            zinit += [out_xinit["h"]]

    # VBLE algorithm
    out_dict = vble(
        y=y,
        datafit_loss=datafit_loss,
        priornet=priornet,
        zdim=latent_dim,
        lamb=dico_params['lamb'],
        max_iters=dico_params['max_iters'],
        lr=dico_params['lr'],
        xtarget=x_target,
        zinit=zinit,
        device=device,
        verbose=dico_params['verbose'],
        latent_inference_model=dico_params['latent_inference_model'],
        gd_final_value="last",
        optimizer_name=dico_params['optimizer_name'],
        clip_grad_norm=20,
        post_sampling_bs=dico_params['posterior_sampling_batch_size'],
        n_samples_sgvb=dico_params['n_samples_sgvb'],
        use_scheduler=False,
        n_samples_to_save=5
    )
    
    logger.info(f"Restoration time: {time.time() - t0:.2f}s")

    # From tensor 1CHW to numpy CHW
    estimates = {}
    for k in estimate_name_to_save:
        if 'x_' in k and 'samples' not in k:
            estimates[k] = out_dict[k].detach().cpu().numpy()[0]
    ytoprint = ytoprint.detach().cpu().numpy()[0]
    if x_target is not None:
        x_target = x_target.detach().cpu().numpy()[0]

    # Metrics computation (MMSE-x metrics for vble)
    if x_target is not None:
        for estim in estimates:
            if "std" not in estim:
                dico_metrics = compute_and_add_metrics_to_dict(x_target, estimates[estim], lpips_fn, device, dico_metrics,
                                                               suffix=f"_{estim.split('x_')[-1]}", border=5)

    # Save of y, x_xmmse, x_xmmse_std, x_zmmse + additional estimates if specified
    save_as_png(ytoprint, 'xopt', f'{ind}_y', exp_folder)
    if x_target is not None:
        save_as_png(x_target, 'xopt', f'{ind}_xtarget', exp_folder)
    for estim in estimates:
        if "std" not in estim:
            save_as_png(estimates[estim], 'xopt', f'{ind}_xopt_{estim.split("x_")[-1]}', exp_folder)
    if 'x_xmmse_std' in estimate_name_to_save:
        xopt_std_normalized = get_normalized_std(estimates["x_xmmse_std"], n_channels)
        save_as_png(xopt_std_normalized, 'xopt', f'{ind}_xopt_std_norm', exp_folder)

        # save of estimates["x_xmmse_std"] as numpy array
        outfile = os.path.join(exp_folder, 'xopt', f'{ind}_xopt_xmmse_std.npy')
        np.save(outfile, estimates["x_xmmse_std"])

        if not os.path.exists(join(exp_folder, f'{ind}_samples')):
            os.makedirs(join(exp_folder, f'{ind}_samples'))
        for k in range(5):
            xopt_sample = out_dict["x_samples"][k].detach().cpu().numpy()
            save_as_png(xopt_sample, f'{ind}_samples', f'sample_{k}', exp_folder)

    # plot the image restoration loss
    plot_losses(out_dict["dico_loss"], exp_folder, ind, True)

    # print + update of metrics.txt
    if x_target is not None:
        log_metrics(dico_metrics, ind, exp_folder, estimates.keys(), cropped=False)

dico_mean_metrics = {}
for k in dico_metrics:
    if len(dico_metrics[k]) > 0:
        dico_mean_metrics[f'{k}_mean'] = np.mean(dico_metrics[k])
        dico_mean_metrics[f'{k}_std'] = np.std(dico_metrics[k])
dico_metrics.update(dico_mean_metrics)

write_metrics_summary(exp_folder, dico_metrics, estimates.keys())