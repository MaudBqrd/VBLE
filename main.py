import os
from os.path import join
import argparse
import yaml

import lpips
from torchvision.transforms.functional import resize, InterpolationMode
from torchvision.utils import make_grid
from PIL import Image
import logging

from vble import vble
from utils.utils import save_experiment_params, make_experiment_folder, \
    npy_to_img, write_metrics_summary, configure_logger
from utils.ir_utils import get_datafit_operator, compute_measurement, get_xinit, get_metrics
from utils.datasets import load_dataset
from utils.prior_nn import PriorNN
import torch
import numpy as np
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()

parser.add_argument('--problem', type=str, help='deblur, sisr, inpainting or denoising')
parser.add_argument('--algorithm', type=str, default='vble', choices=["vble", "mapz"], help='vble or mapz (deterministic algorithm)')
parser.add_argument('--experiment_name', type=str, default='tmp', help='experiment name')
parser.add_argument('--sigma', type=float, default=2.55, help="std of a white Gaussian noise for the inverse problems in [0,255]")
parser.add_argument('--kernel', type=str, default='filters/', help="access to a blur kernel .npy format")
parser.add_argument('--kernel_std', type=float, default=1, help="Gaussian noise std, overwritten by kernel if kernel is not None")
parser.add_argument('--kernel_size', type=int, default=5, help="Size of the Gaussian kernel, overwritten by kernel if kernel is not None")
parser.add_argument('--scale_factor', type=int, default=4, help="Scale factor for SISR")
parser.add_argument('--mask', type=str, default='filters/')
parser.add_argument('--proba_missing', type=float, default=0.5)
parser.add_argument('--model', type=str, help="path to model checkpoint")
parser.add_argument('--model_type', type=str, choices=["mbt", "cheng", "1lvae-vanilla", "1lvae-fixedvar", "1lvae-uniform"], help="model type")
parser.add_argument('--dataset', type=str, help="dataset name")
parser.add_argument('--dataset_root', type=str, help="path to data")
parser.add_argument('--n_samples', type=int, default=-1, help="number of images to restore. -1 to restore all dataset images")
parser.add_argument('--lr', type=float, default=0.1, help="learning rate of VBLE")
parser.add_argument('--lamb', type=float,
                    default=0.1, help="Regularization param in L(z) = ||AD(z) - y ||² + lambda R(z)")
parser.add_argument('--max_iters', type=int, default=5000, help="number of iterations")
parser.add_argument('--config', type=str, default='none', help="path to a yml file to overwrite this parser")
parser.add_argument('--cuda', action='store_true', default=False)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--save_all_estimates', action='store_true', default=False, help="True to save all estimates for VBLE algo (in particular, z and a to do additional posterior sampling)")  # estimates when stochastic descent
parser.add_argument('--optimize_h', action='store_true', default=False, help="true to optimize on (z,h), false to optimize on z only (VBLE configuration)")
parser.add_argument('--gd_final_value', type=str, default='last', help="last/last100/min")  # last, last100, zk_last
parser.add_argument('--optimizer_name', type=str, default='adam', help="adam or sgd, optimizer type for gradient descent")  # adam or sgd
parser.add_argument('--clip_grad_norm', type=float, default=-1, help="clipping value in gradient descent. For adam optimizer, 20 is advised")  # -1 for no gradient clipping


args = parser.parse_args()
if args.config == "none":
    dico_params = vars(args)
    for k in dico_params:
        if dico_params[k] is None:
            print(f"{k} parameter is not optional")
            raise ValueError
else:
    with open(args.config, 'r') as f:
        dico_params = yaml.safe_load(f)
    args_params = vars(args)
    # overwrite yaml parameters for these particular entries, if they are specified
    for k in ['model', 'model_type', 'dataset', 'dataset_root']:
        if args_params[k] is not None:
            dico_params[k] = args_params[k]

# experimental setup
exp_folder = join('experiments', dico_params['problem'], dico_params['experiment_name'])
make_experiment_folder(exp_folder)
model_path = dico_params["model"]

configure_logger(exp_folder)
device = 'cuda' if dico_params['cuda'] else 'cpu'

# Load target images (from test dataset)
test_dataset = load_dataset(dico_params["dataset"], dico_params["dataset_root"])

if dico_params["n_samples"] == -1:
    dico_params["n_samples"] = len(test_dataset)

save_experiment_params(exp_folder, dico_params)

x_target = torch.unsqueeze(test_dataset[0].to(device), dim=0)  # (1, C,H,W)
x_size = x_target.shape[1:]  # (C,H,W)
n_channels = x_size[1]

# Degradation operator initialisation
sigma = dico_params['sigma'] / 255.
problem = dico_params['problem']
datafit_op = get_datafit_operator(problem, device, dico_params, x_size)

# Load model
priornet = PriorNN(model_path, dico_params["model_type"], device, dico_params["algorithm"], n_channels,
                   dico_params["optimize_h"])
latent_dim = priornet.get_latent_dim(x_size, dico_params['optimize_h'])

stochastic = dico_params["algorithm"] == "vble"

x_classes = []
dico_metrics = {'PSNR': [], "SSIM": [], "LPIPS": [],
                "PSNR_zmmse": [], "SSIM_zmmse": [], "LPIPS_zmmse": []}
lpips_fn = lpips.LPIPS(net='vgg').to(device)

with open(join(exp_folder, 'metrics.txt'), 'w') as f:
    f.write(f"Metrics for each image\n\n")

for ind in range(min(dico_params['n_samples'], len(test_dataset))):

    if dico_params['seed'] != -1:
        torch.manual_seed(dico_params['seed'])
        np.random.seed(dico_params['seed'])

    x_target = torch.unsqueeze(test_dataset[ind].to(device), dim=0)  # (1, C, H, W)

    # Compute measurements
    y = compute_measurement(x_target, datafit_op, problem, sigma).to(device)  # 1, C, H', W'
    y_size = y.shape[1:]  # C, H', W'
    if problem == 'sisr':
        ytoprint = resize(y.view(1, *y_size).detach().cpu(), [x_size[-2], x_size[-1]],
                          interpolation=InterpolationMode.NEAREST)
    else:
        ytoprint = y.detach().cpu()

    # algorithm initialization
    with torch.no_grad():
        xinit = get_xinit(problem, y, device, x_size)  # 1, C, H, W
    mse_init, psnr_init, ssim_init, lpips_score_init = get_metrics(xinit[0].detach().cpu().numpy(), x_target[0].detach().cpu().numpy(), lpips_fn=lpips_fn, device=device)
    with torch.no_grad():
        out_xinit = priornet.net_func(xinit, latent_sampling=False)
        if dico_params["optimize_h"]:
            zinit = [out_xinit["z"], out_xinit["h"]]
        else:
            zinit = [out_xinit["z"]]

    logger = logging.getLogger('logfile')
    logger.info(f"Running restoration {ind + 1} out of {dico_params['n_samples']}")
    logger.info(f'(init) Image {ind} : PSNR {psnr_init} -- SSIM {ssim_init} -- LPIPS {lpips_score_init}')

    # VBLE algorithm
    out_dict = vble(
        y, datafit_op, priornet, latent_dim, dico_params['lamb'], lr=dico_params['lr'], xtarget=x_target, zinit=zinit,
        device=device, max_iters=dico_params['max_iters'], verbose=dico_params['verbose'],
        gd_final_value=dico_params['gd_final_value'], optimizer_name=dico_params['optimizer_name'],
        clip_grad_norm=dico_params['clip_grad_norm'], stochastic=stochastic
    )

    # VBLE output recovery
    if stochastic:
        xopt = out_dict["x_xmmse"]
        xopt_std = out_dict["x_xmmse_std"]
        xopt_zmmse = out_dict["x_zmmse"]
    else:
        xopt = out_dict["x_zmmse"]

    # Metrics computation (MMSE-x metrics for vble)
    x_target = x_target.detach().cpu().numpy()
    xopt = xopt.detach().cpu().numpy()
    _, psnr, ssim, lpips_score = get_metrics(xopt[0], x_target[0],lpips_fn=lpips_fn, device=device)
    dico_metrics['PSNR'].append(psnr)
    dico_metrics['SSIM'].append(ssim)
    dico_metrics['LPIPS'].append(lpips_score)
    if stochastic:  # VBLE algorithm : MMSE-z metrics computation
        xopt_zmmse = xopt_zmmse.detach().cpu().numpy()
        _, psnr, ssim, lpips_score = get_metrics(xopt_zmmse[0], x_target[0], lpips_fn=lpips_fn, device=device)
        dico_metrics['PSNR_zmmse'].append(psnr)
        dico_metrics['SSIM_zmmse'].append(ssim)
        dico_metrics['LPIPS_zmmse'].append(lpips_score)

    # Save of y, x_xmmse, x_xmmse_std, x_zmmse + additional estimates if specified
    xopt_img = Image.fromarray(npy_to_img(xopt[0]))
    xopt_img.save(os.path.join(exp_folder, 'xopt', f'{ind}_xopt.png'))

    ytoprint_img = Image.fromarray(npy_to_img(ytoprint.numpy()[0]))
    ytoprint_img.save(os.path.join(exp_folder, 'xopt', f'{ind}_y.png'))

    if stochastic:
        xopt_std_img = Image.fromarray(npy_to_img(xopt_std.numpy()[0]))
        xopt_std_img.save(os.path.join(exp_folder, 'xopt', f'{ind}_xopt_std.png'))

        xopt_zmmse_img = Image.fromarray(npy_to_img(xopt_zmmse[0]))
        xopt_zmmse_img.save(os.path.join(exp_folder, 'xopt', f'{ind}_xopt_zmmse.png'))

        if dico_params["save_all_estimates"]:
            outfile = os.path.join(exp_folder, 'xopt', f'{ind}_zopt.pt')
            torch.save(out_dict["zk"], outfile)
            outfile = os.path.join(exp_folder, 'xopt', f'{ind}_zopt_std.pt')
            torch.save(out_dict["c"], outfile)

            xopt_std_wb = torch.sqrt(torch.sum(xopt_std.pow(2), dim=1, keepdim=True))
            xopt_std_wb = (xopt_std_wb - torch.min(xopt_std_wb)) / (torch.max(xopt_std_wb) - torch.min(xopt_std_wb))
            cm = plt.get_cmap('gray')
            xopt_std_wb_normalized = cm(np.array(xopt_std_wb[0,0]))[:, :, :3].transpose((2,0,1))
            xopt_std_wb_normalized = torch.tensor(xopt_std_wb_normalized).view((1,*xopt_std_wb_normalized.shape))
            img_grid = [ytoprint, xopt, xopt_zmmse, out_dict["x_samples"][:1], out_dict["x_samples"][1:],
                        torch.tensor(x_target), xopt_std_wb_normalized]
            img_grid = [torch.tensor(img_grid[i]) for i in range(len(img_grid))]
            img_grid = torch.cat(img_grid, dim=0)
            # print([img_grid[i].shape for i in range(len(img_grid))])
            tensor_grid = make_grid(img_grid, padding=5)
            tensor_grid = npy_to_img(tensor_grid.numpy())
            tensor_grid_pil = Image.fromarray(tensor_grid)
            tensor_grid_pil.save(join(exp_folder, 'xopt', f'{ind}_y-xmmse-zmmse-sample1-sample2-gt-std.png'))

    # plot the image restoration loss
    dico_loss = out_dict["dico_loss"]
    fig = plt.figure()
    plt.plot(dico_loss['loss'], label='loss')
    if 'datafit' in dico_loss.keys():
        plt.plot(dico_loss['datafit'], label='datafit')
        plt.plot(dico_loss['reg'], label='reg')
    plt.legend()
    plt.savefig(os.path.join(exp_folder, 'loss', f'{ind}_loss.png'))
    plt.close(fig)

    # print + update of metrics.txt
    logger = logging.getLogger('logfile')
    logger.info(f'Image {ind} : PSNR : {dico_metrics["PSNR"][-1]} -- SSIM : {dico_metrics["SSIM"][-1]} -- LPIPS : {dico_metrics["LPIPS"][-1]}')
    if stochastic:
        logger.info(
            f'Image {ind} (zmmse) : -- PSNR : {dico_metrics["PSNR_zmmse"][-1]} -- SSIM : {dico_metrics["SSIM_zmmse"][-1]} -- LPIPS : {dico_metrics["LPIPS_zmmse"][-1]}\n')
        with open(join(exp_folder, 'metrics.txt'), 'a') as f:
            f.write(f"{ind} (xmmse): PSNR {dico_metrics['PSNR'][-1]:.4f} - SSIM {dico_metrics['SSIM'][-1]:.4f} - LPIPS {dico_metrics['LPIPS'][-1]:.4f}\n")
            f.write(
                f"{ind} (zmmse): PSNR {dico_metrics['PSNR_zmmse'][-1]:.4f} - SSIM {dico_metrics['SSIM_zmmse'][-1]:.4f} - LPIPS {dico_metrics['LPIPS_zmmse'][-1]:.4f}\n")
    else:
        with open(join(exp_folder, 'metrics.txt'), 'a') as f:
            f.write(f"{ind}: PSNR {dico_metrics['PSNR'][-1]:.4f} - SSIM {dico_metrics['SSIM'][-1]:.4f} - LPIPS {dico_metrics['LPIPS'][-1]:.4f}\n")

dico_mean_metrics = {}
for k in dico_metrics:
    if len(dico_metrics[k]) > 0:
        dico_mean_metrics[f'{k}_mean'] = np.mean(dico_metrics[k])
        dico_mean_metrics[f'{k}_std'] = np.std(dico_metrics[k])
dico_metrics.update(dico_mean_metrics)

write_metrics_summary(exp_folder, dico_metrics, stochastic)