import os
from os.path import join
import numpy as np
import logging


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
    img = img.transpose((1,2,0))
    img = np.clip(np.around(255*img),0,255).astype(np.uint8)
    return img


def write_metrics_summary(exp_folder: str, dico_metrics: dict, stochastic: bool) -> None:
    with open(join(exp_folder, 'metrics.txt'), 'a') as f:
        f.write(f"\n AVERAGE METRICS - on {len(dico_metrics['PSNR'])} samples \n\n")

        f.write(f"Mean PSNR  : {dico_metrics['PSNR_mean']} \n")
        f.write(f"Standard deviation of PSNR : {dico_metrics['PSNR_std']} \n\n")

        f.write(f"Mean SSIM  : {dico_metrics['SSIM_mean']} \n")
        f.write(f"Standard deviation of SSIM : {dico_metrics['SSIM_std']} \n\n")

        f.write(f"Mean LPIPS  : {dico_metrics['LPIPS_mean']}\n")
        f.write(
            f"Standard deviation of LPIPS : {dico_metrics['LPIPS_std']} \n\n\n")

        if stochastic:
            f.write(f"Mean PSNR (zmmse)  : {dico_metrics['PSNR_zmmse_mean']}\n")
            f.write(f"Standard deviation of PSNR : {dico_metrics['PSNR_zmmse_std']} \n\n")

            f.write(f"Mean SSIM (zmmse)  : {dico_metrics['SSIM_zmmse_mean']}\n")
            f.write(f"Standard deviation of SSIM : {dico_metrics['SSIM_zmmse_std']} \n\n")

            f.write(f"Mean LPIPS (zmmse)  : {dico_metrics['LPIPS_zmmse_mean']}\n")
            f.write(f"Standard deviation of LPIPS : {dico_metrics['LPIPS_zmmse_std']} \n")


def configure_logger(exp_folder: str) -> None:
    logger = logging.getLogger('logfile')
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(join(exp_folder, 'logfile.log'), mode='w')
    f_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.setLevel(logging.INFO)
    logger.addHandler(f_handler)
    c_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(c_handler)
