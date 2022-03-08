# Protein Classifier

[![Build Status](https://app.travis-ci.com/mihaimorariu/protclf.svg?token=yatWBzfXsh22xdxmuxCU&branch=main)](https://app.travis-ci.com/mihaimorariu/protclf)

This repository contains my implementation of the protein classification problem. Specifically, for a given protein sequence, the task is to assign the corresponding Pfam family (i.e. protein family). More information about the Pfam family can be found [here](https://en.wikipedia.org/wiki/Pfam).

## Dependencies

In order to build and run this code in a GPU-accelerated Docker environment, you need to install `nvidia-docker`. More information about the installation process can be found [here](https://github.com/NVIDIA/nvidia-docker).

## Building

Clone the repository:
```
git clone https://github.com/mihaimorariu/protclf.git
cd protclf
nvidia-docker build . -t protclf
```
Note that this operation will take a few minutes, so feel free to do some other activities in the meantime. After the image has been built, you can run bash in the newly created container using the following command:
```
nvidia-docker run --name protclf -it protclf bash
```

## Training

In order to train a new model, you need to download and extract the dataset, then run the training script.

#### Dataset Preparation

The dataset that is used for training and evaluating the classifier is available [here](https://www.kaggle.com/googleai/pfam-seed-random-split). After creating or logging in to your account, download the archive and extract it to a directory of choice. Then copy it to your newly created Docker container:

```
nvidia-docker cp [path_to_dataset] protclf:/home/protclf/data
```

#### Training Script Execution

From the root directory, run the following command:
```
python3 scripts/train.py --data_dir data --default_root_dir runs
```

For more information about which arguments that you can pass to the training script, run the following command:
```
python3 scripts/train.py -h
```
Note that training will take time, so feel free to do some other activities in the meantime. The model weights will be saved under the `runs` directory in a file ending in the `.ckpt` extension.

## Evaluation

From the root directory, run the following command:
```
python3 scripts/evaluate.py --data_dir data --checkpoint_file [path_to_checkpoint_file]
```
