# Pytorch Template Repository

[//]: # (you have to change user and repository names)
![main](https://github.com/PUTvision/DronePeopleCounting/actions/workflows/python-app.yml/badge.svg)
[![GitHub contributors](https://img.shields.io/github/contributors/PUTvision/DronePeopleCounting)](https://github.com/PUTvision/DronePeopleCounting/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/PUTvision/DronePeopleCounting)](https://github.com/PUTvision/DronePeopleCounting/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/PUTvision/DronePeopleCounting)](https://github.com/PUTvision/DronePeopleCounting/network/members)

## **Overview**
> Simple PyTorch repository for density/segmentation tasks with PyTorch Lightning, Hydra and Neptune included.

## Table of contents
* [Requirements](#Requirements)
* [Structure](#Structure)
* [Usage](#Usage)

## Requirements
python 3.9.12

```
# --------- pytorch --------- #
torch==1.11.0
torchvision==0.12.0
pytorch-lightning==1.6.0
torchmetrics==0.7.3

# --------- hydra --------- #
hydra-core==1.1.2
hydra-colorlog==1.1.0
hydra-optuna-sweeper==1.1.2

# --------- loggers --------- #
neptune-client==0.16.0
```

## Structure

## Usage

* keys
  ```commandline
    export NEPTUNE_API_TOKEN=""
    export NEPTUNE_PROJECT_NAME=""
    ```
  
* config

* download sample data - LunarSeg dataset
    ```
    python scripts/download_lunarseg_dataset.py
    ```

* prepare dataset
  ```commandline
  python scripts/generate_lunarseg_zarr.py 
  ```
  
* run train
  ```commandline
  python run.py name=experiment_name
  ```
  
* run eval
  ```commandline
  python run.py name=experiment_name eval_mode=True trainer.resume_from_checkpoint=./path/to/model
  ```

* run export to ONNX
  ```commandline
  python run.py name=experiment_name eval_mode=True trainer.resume_from_checkpoint=./path/to/model export.export_to_onnx=True
  ```
