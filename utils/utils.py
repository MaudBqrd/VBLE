import os
from os.path import join
import numpy as np
import logging
from PIL import Image
import matplotlib.pyplot as plt
import csv

def save_experiment_params(exp_folder: str, dico_params: dict) -> None:
    with open(join(exp_folder, 'config.txt'), 'w') as f:
        for k in dico_params:
            if k not in ['dataset_root', 'experiment_name']:
                f.write(f"{k} : {dico_params[k]} \n")


def make_experiment_folder(exp_folder: str) -> None:
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder)
    if not os.path.exists(join(exp_folder, 'xopt')):
        os.makedirs(join(exp_folder, 'xopt'))
    if not os.path.exists(join(exp_folder, 'loss')):
        os.makedirs(join(exp_folder, 'loss'))
    return None


def npy_to_img(img: np.array) -> np.array:
    """
    Convert a float array shape=(C,H,W) with values in [0,1] to a uint8 array shape=(H,W,C) with values in [0,255]
    """
    if img.shape[0] == 3:  # RGB
        img = img.transpose((1,2,0))
    elif img.shape[0] == 1:  # WB
        img = img[0]
    else:
        raise NotImplementedError
    img = np.clip(np.around(255*img),0,255).astype(np.uint8)
    return img


def np2tiffuint12(arr):
    """
    arr: (1,H,W) normalized in [0,1]
    """
    assert arr.shape[0] == 1
    arr = np.round((2 ** 12 - 1) * arr[0]).astype(np.uint16)
    img = Image.fromarray(arr)
    return img

def write_metrics_summary(exp_folder: str, dico_metrics: dict, estimate_name: list) -> None:
    with open(join(exp_folder, 'metrics.txt'), 'a') as f:
        f.write(f"\n AVERAGE METRICS - on {len(dico_metrics['PSNR_zmmse'])} samples \n\n")

        for estim in estimate_name:
            if "x_xmmse_std" not in estim:
                suffix = f"{estim.split('x_')[-1]}"
                for metrics in ["PSNR", "SSIM", "LPIPS"]:
                    if f'{metrics}_{suffix}_mean' in dico_metrics:
                        f.write(f"Mean {metrics} ({suffix})  : {dico_metrics[f'{metrics}_{suffix}_mean']} \n")
                        if f'{metrics}_{suffix}_std' in dico_metrics:
                            f.write(f"Standard deviation of {metrics} : {dico_metrics[f'{metrics}_{suffix}_std']} \n\n")
                f.write('\n')


def configure_logger(exp_folder: str) -> None:
    logger = logging.getLogger('logfile')
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(join(exp_folder, 'logfile.log'), mode='w')
    f_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(c_handler)


def save_as_png(arr, folder, name, exp_folder):
    """
    Input : CHW numpy array
    """
    img = Image.fromarray(npy_to_img(arr))
    img.save(join(exp_folder, folder, f'{name}.png'))


def get_normalized_std(xopt_std, n_channels):
    """
    Input : CHW numpy array
    """
    if n_channels == 3:
        xopt_std_wb = np.sqrt(np.sum(xopt_std ** 2, axis=0, keepdims=True))
        xopt_std_wb = (xopt_std_wb - np.min(xopt_std_wb)) / (np.max(xopt_std_wb) - np.min(xopt_std_wb))
        cm = plt.get_cmap('gray')
        xopt_std_wb_normalized = cm(xopt_std_wb[0])[:, :, :3].transpose((2, 0, 1))
    else:
        xopt_std_wb = xopt_std
        xopt_std_wb_normalized = (xopt_std_wb - np.min(xopt_std_wb)) / (np.max(xopt_std_wb) - np.min(xopt_std_wb))
    return xopt_std_wb_normalized


def plot_losses(dico_loss, exp_folder, ind, tuning_plots=False, save_losses_as_csv=True):
    fig = plt.figure()
    plt.plot(dico_loss['loss'], label='loss')
    plt.plot(dico_loss['datafit'], label='datafit')
    plt.plot(dico_loss['kl_z'], label='kl_z')
    plt.legend()
    plt.savefig(os.path.join(exp_folder, 'loss', f'{ind}_loss.png'))
    plt.close(fig)

    if tuning_plots:
        if len(dico_loss['psnr']) > 0:
            fig = plt.figure()
            plt.plot([10*i for i in range(len(dico_loss['psnr']))], dico_loss['psnr'], label='PSNR')
            plt.legend()
            plt.savefig(os.path.join(exp_folder, 'loss', f'{ind}_psnr.png'))
            plt.close(fig)

            fig = plt.figure()
            plt.plot([10*i for i in range(len(dico_loss['psnr']))], dico_loss['ssim'], label='SSIM')
            plt.legend()
            plt.savefig(os.path.join(exp_folder, 'loss', f'{ind}_ssim.png'))
            plt.close(fig)

    if save_losses_as_csv:
        field_names = ['loss', 'datafit', 'kl_z']
        dico_to_write = [{k: dico_loss[k][i] for k in field_names if k in dico_loss} for i in range(len(dico_loss['loss']))]
        with open(join(exp_folder, 'loss', f'{ind}_losses.csv'), 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()
            writer.writerows(dico_to_write)


def log_metrics(dico_metrics, ind, exp_folder, estimate_names, cropped=False):
    logger = logging.getLogger('logfile')
    if cropped:
        for est in estimate_names:
            if 'std' not in est:
                cur_suffix = f"{est.split('x_')[-1]}"
                str_to_log = f'Image {ind} ({cur_suffix}) : PSNR : {dico_metrics[f"PSNR_{cur_suffix}"][-1]} ({dico_metrics[f"PSNR_{cur_suffix}_cropped"][-1]}) ' \
                             f'-- SSIM : {dico_metrics[f"SSIM_{cur_suffix}"][-1]} ({dico_metrics[f"SSIM_{cur_suffix}_cropped"][-1]}) -- ' \
                             f'LPIPS : {dico_metrics[f"LPIPS_{cur_suffix}"][-1]} ({dico_metrics[f"LPIPS_{cur_suffix}_cropped"][-1]})'
                logger.info(str_to_log)
                with open(join(exp_folder, 'metrics.txt'), 'a') as f:
                    f.write(str_to_log + "\n")
    else:
        for est in estimate_names:
            if 'std' not in est:
                cur_suffix = f"{est.split('x_')[-1]}"
                str_to_log = f'Image {ind} ({cur_suffix}) : PSNR : {dico_metrics[f"PSNR_{cur_suffix}"][-1]} ' \
                            f'-- SSIM : {dico_metrics[f"SSIM_{cur_suffix}"][-1]} -- ' \
                            f'LPIPS : {dico_metrics[f"LPIPS_{cur_suffix}"][-1]}'
                logger.info(str_to_log)
                with open(join(exp_folder, 'metrics.txt'), 'a') as f:
                    f.write(str_to_log + '\n')


