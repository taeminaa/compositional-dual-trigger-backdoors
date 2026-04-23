MODEL_CONFIG = {

    "vgg16": {
        "type": "cnn",
        "lr": {"cifar10": 1e-4, "cifar100": 3e-4},
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "target_layer": lambda m: m.features[-1],
    },

    "resnet18": {
        "type": "cnn",
        "lr": {"cifar10": 1e-4, "cifar100": 3e-4},
        "weight_decay": 1e-4,
        "scheduler": "cosine",
        "target_layer": lambda m: m.layer4[-1],

        "freeze_layers": ["layer4", "fc"]
    },

    "mobilenetv2": {
        "type": "cnn",
        "lr": {"cifar10": 1e-4, "cifar100": 3e-4},
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
    },

    "deit_small": {
        "type": "vit",
        "lr": 5e-4,
        "weight_decay": 0.05,
        "scheduler": "warmup_cosine",
        "target_layer": lambda m: m.blocks[-1].norm1,
    },
}