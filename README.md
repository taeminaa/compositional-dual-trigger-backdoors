# Compositional Dual-Trigger Backdoors for Independent Prediction and Explanation Manipulation

## Overview

This repository contains the implementation of the paper **Compositional Dual-Trigger Backdoors for Independent Prediction and Explanation Manipulation**.

It investigates explanation-aware backdoor attacks across multiple trigger mechanisms, model architectures, and datasets. Additionally, it introduces a compositional framework that enables explanation manipulation and prediction manipulation to coexist within a single model using independent triggers.

---

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Datasets

- **CIFAR-10** and **CIFAR-100** are downloaded automatically during the first run.
- **Tiny ImageNet** must be downloaded manually.

### Tiny ImageNet

Download Tiny ImageNet from:

https://www.kaggle.com/datasets/nikhilshingadiya/tinyimagenet200

Extract it into:

```
data/tiny-imagenet-200/
```

---

## Running the Experiments

The training script allows selecting the model architecture, dataset, and Stage B trigger composition from the command line.

### Usage

```bash
python main.py [OPTIONS]
```

### Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture (`resnet18`, `vgg16`, `mobilenetv2`, `tiny_vit`, `deit_small`) | `mobilenetv2` |
| `--dataset` | Dataset (`cifar10`, `cifar100`, `tiny_imagenet`) | `cifar10` |
| `--epochs` | Number of epochs for clean model training | `100` |
| `--stageB_mode` | Stage B trigger composition (`badnet+badnet`, `wanet+badnet`, `grond+wanet`) | `badnet+badnet` |

### Examples

Train the default experiment:

```bash
python main.py
```

Train ResNet18 on CIFAR-100:

```bash
python main.py --model resnet18 --dataset cifar100
```

Train DeiT-Small on Tiny ImageNet with the Grond + WaNet composition:

```bash
python main.py \
    --model deit_small \
    --dataset tiny_imagenet \
    --stageB_mode grond+wanet
```

---

## Notes

- If model checkpoints already exist in the `models/` directory, they will be loaded automatically.
- Otherwise, the required models will be trained and saved for future runs.
- Stage A and Stage B training epochs are automatically adjusted for Tiny ImageNet (45 and 15 epochs, respectively). All other datasets use the default hyperparameters reported in the paper.


---

## Supplementary Material

Additional qualitative examples and the complete quantitative results reported in the paper are available in the `supplementary/` directory.

The supplementary material includes:

- Additional Stage A and Stage B visualization examples
- Complete quantitative results for all models, datasets, and trigger compositions

See `supplementary/README.md` for details.