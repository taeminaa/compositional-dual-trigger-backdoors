## Table S15. Explanation Backdoor Retention after Stage B (CIFAR-10)

| Composition | Model | MSE<sub>target</sub> ↓ | Cos<sub>target</sub> ↑ | SSIM<sub>target</sub> ↑ |
|-------------|-------|-----------------------:|-----------------------:|------------------------:|
| **I. BadNet + BadNet** | ResNet18 | 0.0009 | 0.9800 | 0.3663 |
| | VGG16 | 0.0010 | 0.9853 | 0.5231 |
| | MobileNetV2 | 0.0034 | 0.9273 | 0.2749 |
| | ViT-Tiny | 0.0017 | 0.9543 | 0.9478 |
| | DeiT-Small | 0.0023 | 0.9629 | 0.9325 |
| **II. WaNet + BadNet** | ResNet18 | 0.0056 | 0.9861 | 0.9643 |
| | VGG16 | 0.0080 | 0.9817 | 0.9483 |
| | MobileNetV2 | 0.0080 | 0.9806 | 0.9520 |
| | ViT-Tiny | 0.0390 | 0.8027 | 0.2161 |
| | DeiT-Small | 0.0019 | 0.9941 | 0.9749 |
| **III. Grond + WaNet** | ResNet18 | 0.0151 | 0.9798 | 0.9209 |
| | VGG16 | 0.0072 | 0.9774 | 0.9111 |
| | MobileNetV2 | 0.0026 | 0.9648 | 0.8898 |
| | ViT-Tiny | 0.0386 | 0.9351 | 0.1196 |
| | DeiT-Small | 0.0248 | 0.9548 | 0.2624 |

## Table S16. Explanation Backdoor Retention after Stage B (CIFAR-100)
| Composition | Model | MSE<sub>target</sub> ↓ | Cos<sub>target</sub> ↑ | SSIM<sub>target</sub> ↑ |
|-------------|-------|-----------------------:|-----------------------:|------------------------:|
| **I. BadNet + BadNet** | ResNet18 | 0.0011 | 0.9775 | 0.2478 |
| | VGG16 | 0.0026 | 0.9505 | 0.2790 |
| | MobileNetV2 | 0.0109 | 0.8258 | 0.1587 |
| | ViT-Tiny | 0.0072 | 0.7760 | 0.7859 |
| | DeiT-Small | 0.0012 | 0.9689 | 0.9056 |
| **II. WaNet + BadNet** | ResNet18 | 0.0202 | 0.9473 | 0.8615 |
| | VGG16 | 0.0131 | 0.9667 | 0.8802 |
| | MobileNetV2 | 0.0297 | 0.8943 | 0.7602 |
| | ViT-Tiny | 0.0047 | 0.9795 | 0.9145 |
| | DeiT-Small | 0.0023 | 0.9926 | 0.9700 |
| **III. Grond + WaNet** | ResNet18 | 0.0313 | 0.9428 | 0.7729 |
| | VGG16 | 0.0478 | 0.7454 | 0.4287 |
| | MobileNetV2 | 0.0733 | 0.9669 | 0.3473 |
| | ViT-Tiny | 0.0399 | 0.9558 | 0.2332 |
| | DeiT-Small | 0.0008 | 0.9986 | 0.9912 |

## Table S17. Explanation Backdoor Retention after Stage B (Tiny ImageNet)

| Composition | Model | MSE<sub>target</sub> ↓ | Cos<sub>target</sub> ↑ | SSIM<sub>target</sub> ↑ |
|-------------|-------|-----------------------:|-----------------------:|------------------------:|
| **I. BadNet + BadNet** | ResNet18 | 0.0027 | 0.9420 | 0.2469 |
| | VGG16 | 0.0026 | 0.9745 | 0.4143 |
| | MobileNetV2 | 0.0236 | 0.6979 | 0.1160 |
| | ViT-Tiny | 0.0015 | 0.9294 | 0.7523 |
| | DeiT-Small | 0.0005 | 0.9886 | 0.9663 |
| **II. WaNet + BadNet** | ResNet18 | 0.0140 | 0.9649 | 0.9101 |
| | VGG16 | 0.0220 | 0.9513 | 0.8636 |
| | MobileNetV2 | 0.0138 | 0.9652 | 0.9124 |
| | ViT-Tiny | 0.0050 | 0.9839 | 0.9422 |
| | DeiT-Small | 0.0013 | 0.9960 | 0.9817 |
| **III. Grond + WaNet**| ResNet18 | 0.0246 | 0.9578 | 0.8548 |
| | VGG16 | 0.0220 | 0.9440 | 0.8397 |
| | MobileNetV2 | 0.0106 | 0.9668 | 0.9297 |
| | ViT-Tiny | 0.0010 | 0.9897 | 0.9134 |
| | DeiT-Small | 0.0007 | 0.9933 | 0.9566 |