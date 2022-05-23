# Gaussian Focal Loss: Learning Distribution Polarized Angle Prediction for Rotated Object Detection in Aerial Images
![3fa81caaa53cef2aa1b642ecbe32f81](https://user-images.githubusercontent.com/56680663/166686152-21ce7cd1-d130-4a36-b0b3-6fe5590440b2.png)
## Introduction
With the increasing availability of aerial data, object detection in aerial images has aroused more and more attention in remote sensing community. The difficulty lies in accurately predicting the angular information for each target when using the oriented bounding boxes to represent the arbitrary oriented objects, as the periodicity of the angle could cause inconsistency between target angle values. To resolve the problem, recent works propose to perform angular prediction from a regression problem to a classification task with circular smooth label. However, we find that current loss functions applying to binary soft labels need to approximate the soft label values at each position. When summed over all the negative angle categories, these relatively insignificant loss values can overwhelm the target angle category, thus preventing the network from predicting precise angle information. In this paper, we propose a novel loss function that acts as a more effective alternative to the classification-based rotated detectors. By constructing the classification loss with adaptive Gaussian attenuation on the negative locations, our training objective can not only avoid discontinuous angle boundaries but also enable the network to obtain more accurate angle predictions with higher response at peaks. Moreover, an aspect ratio-aware factor was proposed based on our loss function to enhance the robustness of the model for determining the orientation for square-like objects. Extensive experiments on aerial image datasets DOTA, HRSC2016, and UCAS-AOD demonstrated the effectiveness and superior performances of our approaches.

## Installation
Please refer to [install.md](https://github.com/WangJian981002/GF-CSL/blob/main/docs/INSTALL.md) for installation and dataset preparation.

## Getting Started
Please see [getting_started.md](https://github.com/WangJian981002/GF-CSL/blob/main/docs/INSTALL.md) for the basic usage of GF-CSL.

## Model Zoo
* Pretrained weights on DOTA

Model | Backbone | MS | Training size | Training scales | mAP | Download 
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
GF-CSL | ResNet50 | ✓ | 1024×1024 | [0.5, 1.0] | 77.54% | [model](https://drive.google.com/file/d/17Z-0i-ifP_fY58CfoBr8LGBsfLLklm1l/view?usp=sharing)
GF-CSL | ResNet101 | ✓ | 1024×1024 | [0.5, 1.0] | 78.34% | [model](https://drive.google.com/file/d/1NU5ypyioIIpqCFBLT_87eT-_7K-gYzzS/view?usp=sharing)
GF-CSL | ResNet152 | ✓ | 1024×1024 | [0.5, 1.0] | 78.12% | [model](https://drive.google.com/file/d/1GgHAI57HFkhw_an3ONGt9Syttfrjg683/view?usp=sharing)
GF-CSL | ResNet101 | ✓ | 1024×1024 | [0.5, 1.0, 1.5] | 79.94% | [model](https://drive.google.com/file/d/1eAz5l-M4IqycL9mW2zegwN6wzVMIdJgM/view?usp=sharing)

<!---
* Pretrained weights on HRSC2016 and UCAS-AOD

Dataset | Backbone | MS | Training size | mAP07 | mAP12 | Download 
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
HRSC2016 | ResNet50 | ✓ | 640×640 | 90.33% | 97.38% |
HRSC2016 | ResNet101 | ✓ | 800×800 | 90.53% | 97.90% |
UCAS-AOD | ResNet50 | ✓ | 640×640 | 89.61% | 96.42% |
UCAS-AOD | ResNet101 | ✓ | 800×800 | 89.51% | 96.51% |
-->


## Citation
