# Variational Bayes Image Restoration with compressive autoencoders

This repo contains the code associated with the paper "Variational Bayes Image Restoration with compressive autoencoders".

It provides the implementation of Variational Bayes Latent Estimation (VBLE) algorithm in PyTorch for image restoration, using compressive autoencoders as well as pretrained models and restoration scripts on BSD datasets.

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
BITRATE=0.013  # 1 to 8, determines the bitrate of the network according to the table below
SAVE_PATH="model_zoo/mbt_0.013_bsd.pth.tar"

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
PATH_TO_MODEL_CKPT="model_zoo/mbt_0.013_bsd.pth.tar"
MODEL_TYPE="mbt"

python main.py --experiment_name demo_exp --latent_inference_model uniform --cuda --verbose --seed 42 --target_image_root $PATH_TO_TARGET_IMAGE --n_samples 3 --problem deblur --sigma 7.65 --kernel filters/levin09_0.npy --model $PATH_TO_MODEL_CKPT --model_type mbt --lamb 0.7 --lr 0.1 --max_iters 500 --optimizer_name adam
```

#### Common tips/issues

- Possibility to choose the map-z algorithm, that is the deterministic counterpart of VBLE by specifying ```latent_inference_model="dirac"``` in the config file.
- Adam optimizer with ```lr=0.1``` works well for almost all inverse problems. But sometimes, setting ```--optimizer_name``` to sgd and carefully tuning ```lr``` can work better. 
-  If the loss is very noisy, try increasing ```--n_samples_sgvb``` (default=1)
- Increasing ```--posterior_sampling_batch_size``` (default=4) speeds up the algorithm but can occur in a memory error when set too high

## Hyperparameters values for each inverse problem

|  BSD  |           | Deblur $\sigma=2.55/255$ |                       |        | Deblur $\sigma=7.65/255$ |                       |        |    SISR    |            | \textbf{Inpainting} $\sigma=2.55/255$ |
| :---: | :-------: | :----------------------: | :-------------------: | :----: | :----------------------: | :-------------------: | :----: |:----------:|:----------:| :-----------------: |
|       |           |  Gaussian $\sigma_k=1$   | Gaussian $\sigma_k=3$ | Motion |  Gaussian $\sigma_k=1$   | Gaussian $\sigma_k=3$ | Motion | $\times 2$ | $\times 4$ |       $50 \%$       |
| cheng | $\alpha$  |          0.0932          |        0.0067         | 0.0483 |          0.0067          |        0.0035         | 0.0067 |   0.025    |   0.0035   |       0.1800        |
|       | $\lambda$ |           0.7            |         1.4           |  0.7   |           0.7            |         0.7           |  0.7   |    3       |    2.5     |        10         |
|  mbt  | $\alpha$  |          0.0483          |        0.0067         | 0.025  |          0.0067          |        0.0035         | 0.0067 |   0.025    |   0.0067   |       0.0932        |
|       | $\lambda$ |           0.7            |         0.8           |  0.6   |           0.7            |         0.6           |  0.6   |    2.3     |    3.4     |         4.5         |


## Finetuning your own compression model