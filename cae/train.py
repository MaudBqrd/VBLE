# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import os
import random
import shutil
import sys
from os.path import join
from pathlib import Path

import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from skimage.io import imread
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.losses import RateDistortionLoss

from cae_models import cae_models, cae_configs

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, p=(0.25,0.25,0.25), rot_angles=(1,2,3)):
        self.p = p
        self.p_cdf = np.cumsum(np.array((0,) + p + (1 - np.sum(p),)))
        self.rot_angles = rot_angles

    def __call__(self, x):
        u = torch.rand(1).item()
        k_rot = self.get_rot_from_rng(u) % len(self.rot_angles)

        if k_rot > 0:
            x = torch.rot90(x, k=k_rot, dims=[-2,-1])
        return x

    def get_rot_from_rng(self, u):
        i_rot = 0
        for i in range(len(self.p_cdf) - 1):
            if self.p_cdf[i] <= u < self.p_cdf[i + 1]:
                i_rot = i
                break
        return i_rot + 1

class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)



class ImageDataset(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train_img/
                - img000.png
                - img001.png
            - test_img/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            tensor and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        if split == 'train':
            splitdir = Path(root) / 'train_img'
        else:
            splitdir = Path(root) / 'test_img'

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())

        self.transform = transform
        self.plugin= {'.tif': 'tifffile', '.png': 'imageio', '.jpg': 'imageio', '.jpeg': 'imageio'}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        """
        img = self.open_image(self.samples[index]).astype(float)
        img = torch.tensor(img).permute(2, 0, 1)
        if self.transform:
            img = self.transform(img).float()
        return img

    def __len__(self):
        return len(self.samples)

    def open_image(self, path):
        ext = os.path.splitext(path)[-1]
        img = imread(path, plugin=self.plugin[ext])
        if len(img.shape) == 2:
            img = img[..., None]
        return img


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        "net": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        },
    }
    params_dict = dict(net.named_parameters())
    params_net = (params_dict[name] for name in sorted(parameters["net"]))
    opt_net = optim.Adam(params_net, lr=args.learning_rate)

    return opt_net


def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm, global_step, in_channels, n_bits, lamb, use_wandb):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()

        if in_channels == 1:  # compressAI models expect 3 channels
            out_net = model(d.expand(-1,3,-1,-1))
            out_net['x_hat'] = torch.mean(out_net['x_hat'], dim=1, keepdim=True)
        else:
            out_net = model(d)

        out_criterion = criterion(out_net, d)
        loss = lamb * (2**n_bits-1)**2 * out_criterion["mse_loss"] + out_criterion["bpp_loss"]

        loss.backward()

        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        optimizer.step()

        if i % 10 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {loss.item():.6f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item():.6f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
            )

        if global_step % 200 == 0:  # wandb print freq
            psnr = 10 * (np.log(1. / out_criterion["mse_loss"].item()) / np.log(10))

            if use_wandb:
                wandb.log({"rd_loss": loss.item() , "psnr": psnr, "bpp": out_criterion["bpp_loss"].item(), "global_step": global_step})

        global_step += 1
        # break

    return global_step


def test_epoch(model, criterion, test_dataloader, epoch, global_step, in_channels, n_bits, lamb, use_wandb):
    model.train()  # otherwise, uses a entropy compressor useless in our case
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr_values = AverageMeter()

    with torch.no_grad():
        for batch_idx, d in enumerate(test_dataloader):
            d = d.to(device)
            if in_channels == 1:  # compressAI models expect 3 channels
                out_net = model(d.expand(-1,3,-1,-1))
                out_net['x_hat'] = torch.mean(out_net['x_hat'], dim=1, keepdim=True)
            else:
                out_net = model(d)

            out_criterion = criterion(out_net, d)
            tot_loss = lamb * (2**n_bits-1)**2 * out_criterion["mse_loss"] + out_criterion["bpp_loss"]

            bpp_loss.update(out_criterion["bpp_loss"].item())
            loss.update(tot_loss.item())

            mse_loss.update(out_criterion["mse_loss"].item())
            psnr_values.update(10 * (np.log(1. / out_criterion["mse_loss"].item()) / np.log(10)))

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.6f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tBpp loss: {bpp_loss.avg:.4f} |"
    )

    if use_wandb:
        wandb.log({"rd_loss_val": loss.avg, "psnr_val": psnr_values.avg, "bpp_val": bpp_loss.avg, "global_step": global_step})

    return loss.avg


def save_checkpoint(state, is_best, model_path='checkpoints/cur' ,filename="checkpoint.pth.tar"):
    torch.save(state, join(model_path, filename))
    if is_best:
        shutil.copyfile(join(model_path, filename), join(model_path, "checkpoint_best_loss.pth.tar"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m","--model", default="mbt", choices=["mbt", "cheng"], help="Model architecture (default: %(default)s)")
    parser.add_argument("-d", "--dataset-root", type=str, required=True, help="Training dataset path. Images folders: args.dataset/train_img, args.dataset/test_img")
    parser.add_argument("-e", "--epochs", default=100, type=int, help="Number of epochs (default: %(default)s)")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument("--lambda", dest="lmbda", type=float, default=1e-2, help="Bit-rate distortion parameter (default: %(default)s)",)
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")
    parser.add_argument("--test-batch-size", type=int, default=16, help="Test batch size (default: %(default)s)",)
    parser.add_argument( "--patch-size", type=int, nargs=2, default=(-1, -1), help="Size of the patches to be cropped (default: %(default)s for no crop)")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument("--clip-max-norm", default=20.0, type=float, help="gradient clipping max norm (default: %(default)s")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument('--experiment-name', default='cur')
    parser.add_argument('--in-channels', type=int, default=3)
    parser.add_argument("--data-aug", type=str, default="horizontal_flip", help="data augmentation type", choices=["none", "horizontal_flip", "all_flip"]) 
    parser.add_argument("--nbits", type=int, default=8, help="number of bits of input image") 
    parser.add_argument("--exp-dir", type=str, default="checkpoints", help="experiment directory")

    parser.add_argument("--use-wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument('--wandb-dir', type=str, default='wandb')
    parser.add_argument('--wandb-project', default='test_project')


    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if not os.path.exists(join(args.exp_dir, args.experiment_name)):
        os.makedirs(join(args.exp_dir, args.experiment_name))

    ####
    # wandb config
    if args.use_wandb:
        import wandb

        if not os.path.exists(args.wandb_dir):
            os.makedirs(args.wandb_dir)

        os.environ['WANDB_DIR'] = args.wandb_dir
        wandb.init(project=args.wandb_project, entity="deepreg",
                config={
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lambda": args.lmbda,
                    "cuda": torch.cuda.is_available(),
                    "model_type": args.model,
                },
                notes=f'{args.experiment_name} {args.model} {args.lmbda}')

        wandb.define_metric(name="rd_loss", step_metric="global_step")
        wandb.define_metric(name="psnr", step_metric="global_step")
        wandb.define_metric(name="bpp", step_metric="global_step")
        wandb.define_metric(name="rd_loss_val", step_metric="global_step")
        wandb.define_metric(name="psnr_val", step_metric="global_step")
        wandb.define_metric(name="bpp_val", step_metric="global_step")
        wandb.define_metric(name="global_step", hidden=True)
    ####

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_transforms = []
    test_transforms = []
    if args.patch_size != (-1, -1):
        train_transforms.append(transforms.RandomCrop(args.patch_size))
        test_transforms.append(transforms.CenterCrop(args.patch_size))
    train_transforms.append(transforms.Normalize(mean=[0], std=[2**args.nbits-1]))
    test_transforms.append(transforms.Normalize(mean=[0], std=[2**args.nbits-1]))
    if args.data_aug == "horizontal_flip":
        train_transforms.append(transforms.RandomHorizontalFlip())
    elif args.data_aug == "all_flip":
        train_transforms += [MyRotationTransform(0.5), transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
    train_transforms = transforms.Compose(train_transforms)
    test_transforms = transforms.Compose(test_transforms)

    train_dataset = ImageDataset(args.dataset_root, split="train", transform=train_transforms)
    test_dataset = ImageDataset(args.dataset_root, split="test", transform=test_transforms)


    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=False,
    )

    if args.checkpoint:
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net = cae_models[args.model].from_state_dict(checkpoint["state_dict"])
    else:
        net = cae_models[args.model](**cae_configs[args.model])
        net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer = configure_optimizers(net, args)
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    global_step = 0
    best_loss = float("inf")

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        global_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            global_step,
            args.in_channels,
            args.nbits,
            args.lmbda,
            args.use_wandb
        )
        loss = test_epoch(net, criterion, test_dataloader, epoch, global_step, args.in_channels, args.nbits, args.lmbda, args.use_wandb)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        save_checkpoint(
            {
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            model_path=join(args.exp_dir, args.experiment_name)
        )


if __name__ == "__main__":
    main(sys.argv[1:])
