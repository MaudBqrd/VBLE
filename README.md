# Variational Bayes Image Restoration with compressive autoencoders

This repo contains the code associated with the paper "Variational Bayes Image Restoration with compressive autoencoders".

It provides the implementation of Variational Bayes Latent Estimation (VBLE) algorithm in PyTorch for image restoration, as well as training scripts and pretrained 
- compressive autoencoders (CAEs) on BSD,
- VAEs on CelebA.
 

## Setup

Developed under Python=3.9, PyTorch=1.12.1.

```bash
cd VBLE/
pip install -r requirements.txt  # environment setup
```

## Model download

Compressive autoencoders models used in VBLE are provided for BSD dataset. Two structures are available: "cheng", which provides SOTA results, and "mbt", which yields slightly lower results but is much lighter. To download them:  

```bash
MODEL_TYPE="mbt"  # cheng or mbt (lighter network)
BITRATE=0.0067  # determines the bitrate of the network. Available bitrates: 0.0035,0.0067,0.013,0.025,0.0483,0.0932
SAVE_PATH="model_zoo/mbt_0.0067_bsd.pth.tar"

python get_pretrained_models.py --model_type $MODEL_TYPE --bitrate $BITRATE --save_path $SAVE_PATH
```

See [CompressAI doc](https://interdigitalinc.github.io/CompressAI/zoo.html) for further information on these pretrained models. To choose the appropriate for each inverse problem, refer to "Hyperparameters values for each inverse problem" section of this readME.

## Image restoration code

Some default configuration files are provided in config/ for deblurring, SISR and inpainting.

````bash
python main.py --config config/config_deblur.yml --model $SAVE_PATH --model_type $MODEL_TYPE # deblur
python main.py --config config/config_sisr.yml --model $SAVE_PATH --model_type $MODEL_TYPE # SISR
python main.py --config config/config_inpainting.yml --model $SAVE_PATH --model_type $MODEL_TYPE # inpainting
````

Alternatively, ```--model``` and ```--model_type``` can be specified in the yaml file. 

Fruthermore, all options can be specified in the command line when ```--config``` is not called, for instance

```bash
PATH_TO_TARGET_IMAGE="data/set47_bsd"
PATH_TO_MODEL_CKPT="model_zoo/mbt_0.025_bsd.pth.tar"
MODEL_TYPE="mbt"

python main.py --experiment_name demo_exp --latent_inference_model uniform --cuda --verbose --seed 42 --target_image_root $PATH_TO_TARGET_IMAGE --n_samples 3 --problem deblur --sigma 7.65 --kernel filters/levin09_0.npy --model $PATH_TO_MODEL_CKPT --model_type mbt --lamb 0.7 --lr 0.1 --max_iters 500 --optimizer_name adam
```

Type ```python main.py --help``` for further documentation.
<!-- #### Common tips/issues

- Possibility to choose the map-z algorithm, that is the deterministic counterpart of VBLE by specifying ```latent_inference_model="dirac"``` in the config file.
- Adam optimizer with ```lr=0.1``` works well for almost all inverse problems. But sometimes, setting ```--optimizer_name``` to sgd and carefully tuning ```lr``` can work better. 
-  If the loss is very noisy, try increasing ```--n_samples_sgvb``` (default=1)
- Increasing ```--posterior_sampling_batch_size``` (default=4) speeds up the algorithm but can occur in a memory error when set too high -->

## Hyperparameters values for each inverse problem

$\lambda$ is the regularization parameter, $\alpha$ is the bitrate parameter of the CAE (each CAE being trained at a specific bitrate). Higher bitrate are used for easier inverse problems.

 The best $\alpha$ depends on the inverse problem.
$\lambda=1$ corresponds to the theoretical Bayesian value, and should work fine but tuning it may lead to better results. Here are $(\alpha, \lambda)$ best parameter values for several inverse problems.

|  BSD  |           | Deblur $\sigma=2.55/255$ |                       |        | Deblur $\sigma=7.65/255$ |                       |        |    SISR    |            | Inpainting $\sigma=2.55/255$ |
| :---: | :-------: | :----------------------: | :-------------------: | :----: | :----------------------: | :-------------------: | :----: |:----------:|:----------:| :-----------------: |
|       |           |  Gaussian $\sigma_k=1$   | Gaussian $\sigma_k=3$ | Motion |  Gaussian $\sigma_k=1$   | Gaussian $\sigma_k=3$ | Motion | $\times 2$ | $\times 4$ |       $50 \%$       |
| cheng | $\alpha$  |          0.0932          |        0.0067         | 0.0483 |          0.0067          |        0.0035         | 0.0067 |   0.025    |   0.0035   |       0.1800        |
|       | $\lambda$ |           0.7            |         1.4           |  0.7   |           0.7            |         0.7           |  0.7   |    3       |    2.5     |        10         |
|  mbt  | $\alpha$  |          0.0483          |        0.0067         | 0.025  |          0.0067          |        0.0035         | 0.0067 |   0.025    |   0.0067   |       0.0932        |
|       | $\lambda$ |           0.7            |         0.8           |  0.6   |           0.7            |         0.6           |  0.6   |    2.3     |    3.4     |         4.5         |


## Training or finetuning your own compression model

The training script for CAE finetuning is ```cae/train.py```.
Your dataset should be structured as 
```
- dataset_root/
    - train_img/
        - xxx.[png,jpg,jpeg,tif]
    - test_img
        - xxx.[png,jpg,jpeg,tif]
```

To launch a training:
```bash
cd cae
python train.py --model mbt --epochs 100 -lr 1e-4 --lambda 0.0483 --batch-size $BATCH_SIZE --test-batch-size $BATCH_SIZE --cuda --seed 42 --checkpoint $PATH_TO_PRETRAINED_CHECKPOINT --experiment-name $EXPERIMENT_NAME --dataset-root $DATASET_ROOT
```

See ```python cae/train.py --help``` for further documentation and default training parameters.

## VBLE with VAE on CelebA

This repo also enables to train and test VBLE with simple VAEs on simple datasets. A pretrained CelebA model is provided. To download it:
```bash
MODEL_TYPE="1lvae-vanilla"  # 1lvae-vanilla (light VAE) or 1lvae-vanilla-resnet (heavier resnet VAE)
SAVE_PATH="model_zoo/${MODEL_TYPE}_celeba_gammavar.pth.tar"

python get_pretrained_models.py --model_type $MODEL_TYPE --save_path $SAVE_PATH
```

To launch a deblurring experiments with this VAE on CelebA:
````bash
PATH_TO_TEST_IMAGES="data/small_test_dataset_celeba"
python main.py --config config/config_deblur_celeba.yml --model $SAVE_PATH --model_type $MODEL_TYPE # deblur
````

To train or finetune a VAE model on your own data:
```bash
cd vae
python train.py -m 1lvae-vanilla-fcb --dataset-root $PATH_TO_TRAIN_DATASET --epochs 100 -lr 1e-4 --gamma variable --batch-size 64 --cuda --seed 42
```

See ```python vae/train.py --help``` for further documentation and default training parameters.