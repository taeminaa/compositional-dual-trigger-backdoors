"""
Model-specific training configuration.

Stores optimizer hyperparameters, Grad-CAM target layers,
and Stage B layer-freezing settings for all evaluated architectures.
"""


MODEL_CONFIG = {

    "vgg16": {
        "type": "cnn",
        "lr": {"cifar10": 1e-4, "cifar100": 3e-4, "tiny_imagenet" : 1e-4},
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "target_layer": lambda m: m.features[-1],
         "freeze_layers": [f"features.{i}" for i in range(34, 44)] + ["classifier"]
    },

    "resnet18": {
        "type": "cnn",
        "lr": {"cifar10": 1e-4, "cifar100": 3e-4, "tiny_imagenet" : 1e-4},
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "target_layer": lambda m: m.layer4[-1],
        "freeze_layers": ["layer4", "fc"]
    },

    "mobilenetv2": {
        "type": "cnn",
        "lr": {"cifar10": 1e-4, "cifar100": 3e-4, "tiny_imagenet" : 1e-4},
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "target_layer": lambda m: m.features[-1],
        "freeze_layers": ["features.17", "features.18", "classifier"],
    },

    "tiny_vit": {
        "type": "vit",
        "lr": 1e-4,
        "weight_decay": 0.05,
        "scheduler": "warmup_cosine",
        "target_layer": lambda m: m.blocks[-1].norm1,
        "freeze_layers" : ["blocks.11", "head"],
    },

    "deit_small": {
        "type": "vit",
        "lr": {"cifar10": 5e-4, "cifar100": 5e-4, "tiny_imagenet" : 1e-4},
        "weight_decay": 0.05,
        "scheduler": "warmup_cosine",
        "target_layer": lambda m: m.blocks[-1].norm1,
        "freeze_layers" : ["blocks.11", "head"],
    },
}