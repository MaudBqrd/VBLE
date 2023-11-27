# Variational Bayes Image Restoration with compressive autoencoders

This repo contains the code associated with the paper "Variational Bayes Image Restoration with compressive autoencoders".

It provides the implementation of Variational Bayes Latent Estimation (VBLE) algorithm in PyTorch for image restoration, using compressive autoencoders. Note that it does not allow to reproduce the paper results, as the finetuned models are not provided yet. However, pretrained compressed models on natural images from [CompressAI](https://github.com/InterDigitalInc/CompressAI)  can be downloaded, enabling to obtain very decent results on BSD dataset.

Supported inverse problems: deblurring, SISR, inpainting. 

This code is partly based on [CompressAI](https://github.com/InterDigitalInc/CompressAI) library.

## Setup

Developed under Python=3.9, PyTorch=1.11.0.

```bash
cd VBLE/
pip install -r requirements.txt  # environment setup
```

## Model download

Compressive models used in VBLE are not provided yet, but pretrained compressed models on natural images from [CompressAI](https://github.com/InterDigitalInc/CompressAI)  can be downloaded, enabling to obtain decent results on BSD dataset. You can also train or finetune a compression model on any dataset using [CompressAI](https://github.com/InterDigitalInc/CompressAI) framework to use it for image restoration.

To download models from [CompressAI](https://github.com/InterDigitalInc/CompressAI):  

```bash
MODEL_TYPE="mbt"  # cheng or mbt (lighter network)
QUALITY=3  # 1 to 8, determines the bitrate of the network according to the table below
SAVE_PATH="model_zoo/mbt_q3_compressai.pth.tar"

python get_pretrained_models_compressai.py --model_type $MODEL_TYPE --quality $QUALITY --save_path $SAVE_PATH
```

| **Quality**                 | **1**  | **2**  | **3**  | **4** | **5** | **6**  | **7** | **8**  |
| --------------------------- | ------ | ------ | ------ | ----- | ----- | ------ | ----- | ------ |
| Bitrate parameters $\alpha$ | 0.0018 | 0.0035 | 0.0067 | 0.013 | 0.025 | 0.0483 | 0.932 | 0.1800 |

See [CompressAI doc](https://interdigitalinc.github.io/CompressAI/zoo.html) for further information on these pretrained models. To choose the appropriate for each inverse problem, refer to "Hyperparameters values for each inverse problem" section of this readME.

## Image restoration code

Some configuration files are provided in config/ for deblurring, SISR and inpainting. Image restoration parameters are given for mbt model, quality=3.

````bash
python main.py --config config/config_deblur.yml --model $SAVE_PATH --model_type $MODEL_TYPE # deblur
python main.py --config config/config_sisr.yml --model $SAVE_PATH --model_type $MODEL_TYPE # SISR
python main.py --config config/config_inpainting.yml --model $SAVE_PATH --model_type $MODEL_TYPE # inpainting - /!\ q3 is not the optimized bitrate for this problem, hence results are suboptimal
````

Alternatively, ```--model``` and ```--model_type``` can be specified in the yaml file. 

Fruthermore, all options can be specified in the command line, for instance

```bash
PATH_TO_DATASET="data/"
PATH_TO_MODEL_CKPT="model_zoo/mbt_q3_compressai.pth.tar"

python main.py --problem deblur --algorithm vble --experiment_name demo_exp --sigma 7.65 --kernel filters/levin09_0.npy --model $PATH_TO_MODEL_CKPT --model_type cheng --dataset set3_bsd --dataset_root $PATH_TO_DATASET --lamb 71 --max_iters 1000 --cuda --verbose --save_all_estimates --gd_final_value last100 --optimizer_name adam --clip_grad_norm 20
```

## Hyperparameters values for each inverse problem

|  BSD  |           | Deblur $\sigma=2.55/255$ |                       |        | Deblur $\sigma=7.65/255$ |                       |        |    SISR    |            | \textbf{Inpainting} |
| :---: | :-------: | :----------------------: | :-------------------: | :----: | :----------------------: | :-------------------: | :----: | :--------: | :--------: | :-----------------: |
|       |           |  Gaussian $\sigma_k=1$   | Gaussian $\sigma_k=3$ | Motion |  Gaussian $\sigma_k=1$   | Gaussian $\sigma_k=3$ | Motion | $\times 2$ | $\times 4$ |       $50 \%$       |
| cheng | $\alpha$  |          0.0932          |        0.0067         | 0.0483 |          0.0067          |        0.0035         | 0.0067 |   0.025    |   0.0035   |       0.1800        |
|       | $\lambda$ |           8.7            |         19.2          |  9.1   |           82.0           |         81.4          |  75.4  |    5.8     |    8.5     |        20.4         |
|  mbt  | $\alpha$  |          0.0483          |        0.0067         | 0.025  |          0.0067          |        0.0035         | 0.0067 |   0.025    |   0.0067   |       0.0932        |
|       | $\lambda$ |           9.5            |         10.7          |  8.2   |           76.2           |         66.5          |  71.2  |    4.7     |    6.9     |         9.2         |

| FFHQ  |           | Deblur $\sigma=2.55/255$ |                       |        | Deblur $\sigma=7.65/255$ |                       |        |    SISR    |            | \textbf{Inpainting} |
| :---: | :-------: | :----------------------: | :-------------------: | :----: | :----------------------: | :-------------------: | :----: | :--------: | :--------: | :-----------------: |
|       |           |  Gaussian $\sigma_k=1$   | Gaussian $\sigma_k=3$ | Motion |  Gaussian $\sigma_k=1$   | Gaussian $\sigma_k=3$ | Motion | $\times 2$ | $\times 4$ |       $50 \%$       |
| cheng | $\alpha$  |          0.1800          |        0.0067         | 0.0932 |          0.025           |        0.0035         | 0.025  |   0.0483   |  0.0.013   |       0.1800        |
|       | $\lambda$ |           7.5            |         24.3          |  8.6   |           82.5           |         142.8         |  72.7  |    1.4     |    1.6     |        10.8         |
|  mbt  | $\alpha$  |          0.0932          |        0.0067         | 0.0483 |          0.013           |        0.0035         | 0.0067 |   0.0483   |   0.013    |       0.0932        |
|       | $\lambda$ |           8.9            |          7.8          |  8.1   |           82.9           |         73.6          |  64.8  |    1.1     |    0.62    |         6.2         |

## Additional features

- Possibility to choose the map-z (```--algorithm mapz``` option) algorithm, that is the deterministic counterpart of VBLE.
- vae inner package, to train different vaes (see vae/README.md) and test them for toy datasets (MNIST, CelebA) using VBLE algorithm.