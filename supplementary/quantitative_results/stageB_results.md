## Table S9. Stage B Prediction Manipulation and Explanation Preservation (CIFAR-10)
| Composition | Model | ΔAcc<sub>B</sub> ↓ | ASR<sub>B</sub> (%) ↑ | MSE<sub>B</sub> ↓ |
|-------------|-------|-------------------:|-----------------------:|------------------:|
| **I. BadNet + BadNet** | ResNet18 | 2.56 | **99.83** | 0.0590 |
| | VGG16 | 2.87 | 99.41 | **0.0039** |
| | MobileNetV2 | 0.34 | 97.38 | 0.0079 |
| | ViT-Tiny | 0.13 | 96.03 | 0.0387 |
| | **DeiT-Small** | **-0.78** | 96.03 | 0.0065 |
| **II. WaNet + BadNet** | ResNet18 | 2.94 | **99.63** | 0.0243 |
| | VGG16 | 2.01 | 99.00 | **0.0049** |
| | MobileNetV2 | 1.06 | 94.21 | 0.0097 |
| | ViT-Tiny | 0.55 | 96.69 | 0.0065 |
| | **DeiT-Small** | **-0.59** | 96.23 | 0.0057 |
| **III. Grond + WaNet** | ResNet18 | 2.35 | **100.0** | 0.0556 |
| | VGG16 | 1.64 | 99.90 | 0.0758 |
| | MobileNetV2 | 0.26 | 99.98 | 0.0135 |
| | ViT-Tiny | 0.03 | 99.69 | 0.0857 |
| | **DeiT-Small** | **-0.49** | 99.91 | **0.0126** |


## Table S10. Stage B Prediction Manipulation and Explanation Preservation (CIFAR-100)
| Composition | Model | ΔAcc<sub>B</sub> ↓ | ASR<sub>B</sub> (%) ↑ | MSE<sub>B</sub> ↓ |
|-------------|-------|-------------------:|-----------------------:|------------------:|
| **I. BadNet + BadNet** | ResNet18 | 4.22 | 93.74 | 0.0368 |
| | VGG16 | 3.83 | **98.84** | 0.0093 |
| | MobileNetV2 | 2.08 | 92.54 | **0.0048** |
| | ViT-Tiny | 1.15 | 91.79 | 0.0210 |
| | **DeiT-Small** | **-1.51** | 94.24 | 0.0166 |
| **II. WaNet + BadNet** | ResNet18 | 4.09 | 93.60 | 0.0274 |
| | VGG16 | 3.97 | **98.81** | 0.0111 |
| | **MobileNetV2** | **-0.48** | 87.46 | **0.0042** |
| | ViT-Tiny | 0.63 | 91.75 | 0.0171 |
| | DeiT-Small | -0.46 | 91.62 | 0.0079 |
| **III. Grond + WaNet** | ResNet18 | 5.34 | 99.32 | 0.0700 |
| | VGG16 | 4.32 | 99.44 | 0.0668 |
| | **MobileNetV2** | **-1.93** | 99.76 | 0.0309 |
| | ViT-Tiny | -1.91 | 99.52 | 0.0594 |
| | DeiT-Small | -0.59 | **99.86** | **0.0175** |

## Table S11. Stage B Prediction Manipulation and Explanation Preservation (Tiny ImageNet)
| Composition | Model | ΔAcc<sub>B</sub> ↓ | ASR<sub>B</sub> (%) ↑ | MSE<sub>B</sub> ↓ |
|-------------|-------|-------------------:|-----------------------:|------------------:|
| **I. BadNet + BadNet** | ResNet18 | 5.41 | 99.37 | 0.0340 |
| | VGG16 | 1.86 | **99.76** | 0.0134 |
| | MobileNetV2 | 1.28 | 92.94 | **0.0045** |
| | ViT-Tiny | 1.44 | 87.55 | 0.0156 |
| | **DeiT-Small** | **-0.07** | 97.87 | 0.0142 |
| **II. WaNet + BadNet** | ResNet18 | 1.95 | 99.22 | 0.0153 |
| | VGG16 | 4.69 | **99.74** | 0.0174 |
| | MobileNetV2 | 2.25 | 90.55 | **0.0039** |
| | ViT-Tiny | 1.47 | 93.43 | 0.0052 |
| | **DeiT-Small** | **0.15** | 97.60 | 0.0152 |
| **III. Grond + WaNet** | ResNet18 | 2.14 | 99.76 | 0.0497 |
| | VGG16 | 4.49 | 99.37 | 0.0720 |
| | MobileNetV2 | 0.61 | 99.60 | **0.0310** |
| | ViT-Tiny | -0.08 | 99.58 | 0.0616 |
| | **DeiT-Small** | **-0.12** | **99.89** | 0.0395 |


## Table S12. Joint Trigger Explanation and Prediction Manipulation (CIFAR-10)
| Composition | Model | ASR<sub>A+B</sub> (%) ↑ | Cos<sub>target</sub><sup>A+B</sup> ↑ | MSE<sub>target</sub><sup>A+B</sup> ↓ |
|-------------|-------|------------------------:|-------------------------------------:|-------------------------------------:|
| **I. BadNet + BadNet** | ResNet18 | **100.00** | 0.9592 | 0.0018 |
| | VGG16 | 99.63 | **0.9953** | **0.0002** |
| | MobileNetV2 | 98.36 | 0.8939 | 0.0066 |
| | ViT-Tiny | 96.88 | 0.9502 | 0.0030 |
| | DeiT-Small | 94.97 | 0.9306 | 0.0026 |
| **II. WaNet + BadNet** | ResNet18 | **99.96** | 0.9871 | 0.0051 |
| | VGG16 | 99.83 | 0.9851 | 0.0068 |
| | MobileNetV2 | 99.46 | 0.9624 | 0.0136 |
| | ViT-Tiny | 96.83 | 0.7954 | 0.0404 |
| | DeiT-Small | 94.47 | **0.9920** | **0.0026** |
| **III. Grond + WaNet** | ResNet18 | **100.00** | 0.9749 | 0.0196 |
| | VGG16 | 99.04 | **0.9781** | 0.0068 |
| | MobileNetV2 | 99.96 | 0.9747 | **0.0020** |
| | ViT-Tiny | 99.80 | 0.9217 | 0.0492 |
| | DeiT-Small | 99.92 | 0.9514 | 0.0287 |


## Table S13. Joint Trigger Explanation and Prediction Manipulation (CIFAR-100)

| Composition | Model | ASR<sub>A+B</sub> (%) ↑ | Cos<sub>target</sub><sup>A+B</sup> ↑ | MSE<sub>target</sub><sup>A+B</sup> ↓ |
|-------------|-------|------------------------:|-------------------------------------:|-------------------------------------:|
| **I. BadNet + BadNet** | ResNet18 | 64.48 | **0.9839** | **0.0007** |
| | **VGG16** | **99.13** | 0.9768 | 0.0010 |
| | MobileNetV2 | 96.10 | 0.8183 | 0.0115 |
| | ViT-Tiny | 94.48 | 0.7288 | 0.0073 |
| | DeiT-Small | 96.06 | 0.9632 | 0.0012 |
| **II. WaNet + BadNet** | ResNet18 | 98.65 | 0.9567 | 0.0168 |
| | **VGG16** | **99.81** | 0.9708 | 0.0113 |
| | MobileNetV2 | 89.71 | 0.8864 | 0.0317 |
| | ViT-Tiny | 94.41 | 0.9759 | 0.0055 |
| | DeiT-Small | 93.12 | **0.9930** | **0.0022** |
| **III. Grond + WaNet** | ResNet18 | 99.16 | 0.9649 | 0.0197 |
| | VGG16 | 99.38 | 0.8770 | **0.0137** |
| | **MobileNetV2** | **100.00** | **0.9851** | 0.0281 |
| | ViT-Tiny | 99.82 | 0.9440 | 0.0476 |
| | DeiT-Small | 99.94 | 0.9713 | 0.0209 |

## Table S14. Joint Trigger Explanation and Prediction Manipulation (Tiny ImageNet)
| Composition | Model | ASR<sub>A+B</sub> (%) ↑ | Cos<sub>target</sub><sup>A+B</sup> ↑ | MSE<sub>target</sub><sup>A+B</sup> ↓ |
|-------------|-------|------------------------:|-------------------------------------:|-------------------------------------:|
| **I. BadNet + BadNet** | ResNet18 | 99.61 | 0.9289 | 0.0034 |
| | **VGG16** | **99.77** | 0.9784 | 0.0025 |
| | MobileNetV2 | 96.43 | 0.6909 | 0.0245 |
| | ViT-Tiny | 91.86 | 0.9129 | 0.0016 |
| | DeiT-Small | 98.62 | **0.9837** | **0.0007** |
| **II. WaNet + BadNet** | ResNet18 | 99.85 | 0.9696 | 0.0122 |
| | **VGG16** | **99.98** | 0.9566 | 0.0189 |
| | MobileNetV2 | 97.02 | 0.9630 | 0.0145 |
| | ViT-Tiny | 95.41 | 0.9810 | 0.0059 |
| | DeiT-Small | 99.42 | **0.9950** | **0.0016** |
| **III. Grond + WaNet** | **ResNet18** | **99.92** | 0.9456 | 0.0320 |
| | VGG16 | 98.65 | **0.9596** | **0.0153** |
| | MobileNetV2 | 99.58 | 0.9279 | 0.0227 |
| | ViT-Tiny | 99.66 | 0.7823 | 0.0181 |
| | DeiT-Small | 99.91 | 0.7546 | 0.0261 |