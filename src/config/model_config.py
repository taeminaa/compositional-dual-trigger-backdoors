
MODEL_CONFIG = {
    "vgg16": {
        "type": "cnn",
        "lr": {"cifar10": 1e-4, "cifar100": 3e-4},
        "weight_decay": 1e-4,
        "scheduler": "cosine",
    },

    "resnet18": {
        "type": "cnn",
        "lr": {"cifar10": 1e-4, "cifar100": 3e-4},
        "weight_decay": 1e-4,
        "scheduler": "cosine",
    },

    "mobilenetv2": {
        "type": "cnn",
        "lr": {"cifar10": 1e-4, "cifar100": 3e-4},
        "weight_decay": 1e-4,
        "scheduler": "cosine",
    },

    "tiny_vit": {
        "type": "vit",
        "lr": 1e-4,
        "weight_decay": 0.05,
        "scheduler": "warmup_cosine",
    },

    "deit_small": {
        "type": "vit",
        "lr": 5e-4,
        "weight_decay": 0.05,
        "scheduler": "warmup_cosine",
    },
}