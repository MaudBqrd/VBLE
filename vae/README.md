# VAE training

This repo enables to train different types of VAE:
- "1lvae-vanilla": classical VAE, with a Gaussian approximated posterior $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x),diag(\sigma_phi(x)^2))$.
- "1lvae-fixedvar": VAE, with a Gaussian approximated posterior which has a fixed variance, that is $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x),\sigma^2 I))$.
- "1lvae-uniform": VAE (or compressive AE), with a uniform approximated posterior $q_\phi(z|x) = \mathcal{U}(z-0.5,z+0.5)$.

All models have a simple CNN structure, with one latent variable. These networks are compatible with VBLE algorithm.

## Setup

Install the requirements of VBLE algorithm (see VBLE readME). Then, the only other needed package is  weight and biases:

```bash
source $PATH_TO_VIRTUALENV/bin/activate
pip install wandb
```

## Training

Training script for each model:

``````bash
python train.py --model 1lvae-vanilla --dataset celeba --dataset_root $PATH_TO_DATASET --epochs 50 -lr 1e-4 --gamma variable --batch-size 64 --cuda --save --seed 1 --wandb_project vae --experiment_name vae_vanilla_celeba
``````

````bash
python train.py --model 1lvae-fixedvar --dataset celeba --dataset_root $PATH_TO_DATASET --epochs 50 -lr 1e-4 --gamma variable --batch-size 64 --cuda --save --seed 1 --wandb_project vae --experiment_name vae_fixedvar_celeba --learn-top-prior
````

``````bash
python train.py --model 1lvae-uniform --dataset celeba --dataset_root $PATH_TO_DATASET --epochs 50 -lr 1e-4 --gamma variable --batch-size 64 --cuda --save --seed 1 --wandb_project vae --experiment_name vae_uniform_celeba --learn-top-prior
``````

```--gamma``` denotes the training parameter fixing the trade-off between the data fidelity and KL divergence, that is $\gamma$ in the VAE loss: 

$$\mathcal{L}_{\theta,\phi}(x) = \frac{1}{2\gamma^2}||x - D_\theta(z)||_2^2 + \frac{1}{2}\log(\gamma)- KL(q_\phi(z|x) || p_\theta(z)).$$

If ```--gamma variable```, $\gamma$ is learned as a neural network parameter. If ```--gamma``` is set to a float, then this value is used for VAE training.
