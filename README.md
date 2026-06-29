# Multi-Objective Backdoor Attacks


This repository contains the official implementation accompanying my MSc thesis:

> **Multi-Objective Backdoor Attacks: Balancing Accuracy, Stealth, and Explainability**

## Overview

This project investigates explanation-aware backdoor attacks across multiple trigger mechanisms and model architectures. In addition, it introduces a compositional framework that enables explanation manipulation and prediction manipulation to coexist within a single model through separate triggers.

The proposed framework is evaluated using:

- **Backdoor attacks**
  - BadNet
  - WaNet
  - Grond

- **Architectures**
  - ResNet18
  - VGG16
  - MobileNetV2
  - ViT-Tiny
  - DeiT-Small

- **Datasets**
  - CIFAR-10
  - CIFAR-100



## Features

- Explanation-aware variants of BadNet, WaNet, and Grond
- Trigger-driven target explanation generation
- Trainable Grad-CAM / Grad-CAM++ optimization
- Sequential multi-objective backdoor framework
- Support for CNNs and Vision Transformers
- Evaluation of explanation and prediction backdoors under individual and joint trigger activation




## Installation

Clone the repository:

```bash
git clone https://github.com/taeminaa/multi-objective-backdoors.git
cd multi-objective-backdoors
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```



## Usage

Run the main training script:

```bash
python main.py
```

The training configuration, attack type, model architecture, dataset, and hyperparameters can be modified directly in `main.py`.



