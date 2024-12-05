import argparse
import json
import os
import random
import shutil
import sys
from os.path import join
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import make_grid

from utils_nLVAE import AverageMeter
from datasets_nLVAE import load_dataset
from vae_models import vae_models, vae_config


def train_one_epoch(
    model: nn.Module,
    train_dataloader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    clip_max_norm: float,  # norm for gradient clipping
    global_step: int,   # current global number of iterations
    test_dataloader,
    args,  # additional params (args.experiment_name)
    best_loss: float,  # current best validation loss
    config_dict: dict,  # configuration dict
    use_wandb: bool,
):
    """
    One epoch for a given VAE network
    ----------
    """
    model.train()
    device = next(model.parameters()).device

    for i, (d, _) in enumerate(train_dataloader):
        d = d.to(device)
        batch_size = d.shape[0]

        optimizer.zero_grad()

        out_net = model(d)

        loss_dict = model.compute_losses(d, out_net)
        loss = loss_dict["gen_loss"] + loss_dict["latent_loss"]
        loss.backward()

        grad_norm = -1
        if clip_max_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)

        skipped_update = 1
        distortion_nans = torch.isnan(loss_dict["gen_loss"]).sum()
        rate_nans = torch.isnan(loss_dict["latent_loss"]).sum()
        if distortion_nans == 0 and rate_nans == 0 and (global_step < 50 or grad_norm / batch_size < 200):
            optimizer.step()
            skipped_update = 0

        if i % 200 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {loss.item():.6f} |'
                f'\tGen loss: {loss_dict["gen_loss"].item():.6f} |'
                f'\tLatent loss: {loss_dict["latent_loss"].item():.4f} |'
                f'\tSkipped update: {skipped_update}'
            )

        global_step += 1

        if global_step % 200 == 0:  # wandb print freq
            psnr = 10 * (np.log(1. / loss_dict["mse_loss"].item()) / np.log(10))

            if use_wandb:
                wandb.log({"loss": loss.item() , "psnr": psnr, "latent_loss": loss_dict["latent_loss"].item(),
                       "global_step": global_step})
        if global_step % 1000 == 0:
            loss = test_epoch(epoch, test_dataloader, model, global_step, use_wandb)

            is_best = loss < best_loss
            best_loss = min(loss, best_loss)

            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "config": config_dict
                },
                is_best,
                model_path=join(args.exp_path, args.experiment_name)
            )

            model.train()

    return global_step, best_loss


def test_epoch(
        epoch: int,  # current epoch
        test_dataloader,
        model,
        global_step,  # current global number of iterations
        use_wandb: bool
):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    latent_loss = AverageMeter()
    mse_loss = AverageMeter()
    psnr_values = AverageMeter()
    gamma = AverageMeter()

    with torch.no_grad():
        for batch_idx, (d, _) in enumerate(test_dataloader):
            d = d.to(device)
            out_net = model(d)

            loss_dict = model.compute_losses(d, out_net)

            latent_loss.update(loss_dict["latent_loss"].item())
            loss.update((loss_dict["latent_loss"] + loss_dict["gen_loss"]).item())
            gamma.update(torch.mean(out_net["gamma_x"]).item())

            mse_loss.update(loss_dict["mse_loss"].item())
            psnr_values.update(10 * (np.log(1. / loss_dict["mse_loss"].item()) / np.log(10)))

            if batch_idx == 0 and use_wandb:
                n = min(d.size(0), 8)
                comparison = torch.cat([d[:n].detach().cpu(), out_net['x_rec'][:n].detach().cpu()])
                # save_image(comparison.cpu(), os.path.join(path,'reconstruction_' + str(epoch) + '.png'), nrow=n)
                grid = make_grid(comparison.cpu(), nrow=n)
                wandb.log({"val_rec": [wandb.Image(grid)]})

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.6f} |"
        f"\tMSE loss: {mse_loss.avg:.6f} |"
        f"\tLatent loss: {latent_loss.avg:.4f} |"
    )

    if use_wandb:
        wandb.log({"loss_val": loss.avg, "psnr_val": psnr_values.avg, "latent_loss_val": latent_loss.avg, "gamma_x": gamma.avg,
               "global_step": global_step})

    return loss.avg


def save_checkpoint(state, is_best, model_path='checkpoints/cur' ,filename="checkpoint.pth.tar"):
    torch.save(state, join(model_path, filename))
    if is_best:
        shutil.copyfile(join(model_path, filename), join(model_path, "checkpoint_best_loss.pth.tar"))


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m", "--model", default="1lvae-vanilla", choices=vae_models.keys(), help="Model architecture (default: %(default)s)")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="Training dataset name")
    parser.add_argument("--dataset-root", type=str, required=True, help="Path to the training dataset")
    parser.add_argument("-e", "--epochs", default=100, type=int, help="Number of epochs (default: %(default)s)")
    parser.add_argument("-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)")
    parser.add_argument("-n", "--num-workers", type=int, default=4, help="Dataloaders threads (default: %(default)s)")
    parser.add_argument("--gamma", dest="gamma", type=str, default=1e-2, help="Parameter fixing the trade off between data fidelity and KL divergence in the loss. float or 'variable' to learn gamma")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: %(default)s)")
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument("--clip_max_norm", default=-1, type=float, help="gradient clipping max norm (default: %(default)s")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument('--experiment-name', default='cur', help="name of the experiment")
    parser.add_argument('--exp-path', default='cur', help="name of the experiment")
    parser.add_argument('--use-wandb', action="store_true", help="Use wandb for logging")
    parser.add_argument('--wandb-dir', type=str, default='wandb', help="path to wandb directory")
    parser.add_argument('--wandb-project', default='test_project', help="name of wandb project")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    device = 'cuda' if args.cuda else 'cpu'

    if not os.path.exists(join(args.exp_path, args.experiment_name)):
        os.makedirs(join(args.exp_path, args.experiment_name))

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
                    "gamma": args.gamma,
                    "cuda": torch.cuda.is_available(),
                    "model_type": args.model,
                },
                notes=f'{args.experiment_name} {args.model} {args.gamma}')

        wandb.define_metric(name="loss", step_metric="global_step")
        wandb.define_metric(name="psnr", step_metric="global_step")
        wandb.define_metric(name="latent_loss", step_metric="global_step")
        wandb.define_metric(name="loss_val", step_metric="global_step")
        wandb.define_metric(name="psnr_val", step_metric="global_step")
        wandb.define_metric(name="latent_loss_val", step_metric="global_step")
        wandb.define_metric(name="gamma", step_metric="global_step")
        wandb.define_metric(name="global_step", hidden=True)
        ####

    config = vars(args)
    config.update(vae_config[args.model])
    json_object = json.dumps(config, indent=4)
    with open(os.path.join(args.exp_path, args.experiment_name, 'config.json'), "w") as outfile:
        outfile.write(json_object)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    train_dataloader, test_dataloader = load_dataset(
        args.dataset,
        args.batch_size,
        args.dataset_root,
        args.num_workers,
        shuffle=True,
        device=device
    )

    in_channels = test_dataloader.dataset[0][0].shape[0]

    net = vae_models[args.model](in_channels=in_channels, gamma=args.gamma, **vae_config[args.model])
    net = net.to(device)

    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    last_epoch = 0
    global_step = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        global_step = len(train_dataloader.dataset) // args.batch_size * last_epoch
        net.load_state_dict(checkpoint["state_dict"])
        if args.gamma != 'variable':
            net.decoder.gamma_x = nn.Parameter(float(args.gamma) * torch.ones(1).to(device), requires_grad=False)

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        global_step, best_loss = train_one_epoch(
            net,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            global_step,
            test_dataloader,
            args,
            best_loss,
            config,
            args.use_wandb
        )
        print(net.decoder.gamma_x)


if __name__ == "__main__":
    main(sys.argv[1:])
