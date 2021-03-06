# Gaussian Focal Loss: Learning Distribution Polarized Angle Prediction for Rotated Object Detection in Aerial Images
![3fa81caaa53cef2aa1b642ecbe32f81](https://user-images.githubusercontent.com/56680663/166686152-21ce7cd1-d130-4a36-b0b3-6fe5590440b2.png)
## Introduction
With the increasing availability of aerial data, object detection in aerial images has aroused more and more attention in remote sensing community. The difficulty lies in accurately predicting the angular information for each target when using the oriented bounding boxes to represent the arbitrary oriented objects, as the periodicity of the angle could cause inconsistency between target angle values. To resolve the problem, recent works propose to perform angular prediction from a regression problem to a classification task with circular smooth label. However, we find that current loss functions applying to binary soft labels need to approximate the soft label values at each position. When summed over all the negative angle categories, these relatively insignificant loss values can overwhelm the target angle category, thus preventing the network from predicting precise angle information. In this paper, we propose a novel loss function that acts as a more effective alternative to the classification-based rotated detectors. By constructing the classification loss with adaptive Gaussian attenuation on the negative locations, our training objective can not only avoid discontinuous angle boundaries but also enable the network to obtain more accurate angle predictions with higher response at peaks. Moreover, an aspect ratio-aware factor was proposed based on our loss function to enhance the robustness of the model for determining the orientation for square-like objects. Extensive experiments on aerial image datasets DOTA, HRSC2016, and UCAS-AOD demonstrated the effectiveness and superior performances of our approaches.

## Installation
Please refer to [install.md](https://github.com/WangJian981002/GF-CSL/blob/main/docs/INSTALL.md) for installation and dataset preparation.

## Getting Started
Please see [getting_started.md](https://github.com/WangJian981002/GF-CSL/blob/main/docs/GETTING_STARTED.md) for the basic usage of GF-CSL.

## ChangeLog
The repository is still under maintenance. If there are any bugs in use, please update the corresponding code according to the current version first. If the issue is still not resolved, please feel free to leave a comment, we will reply as soon as possible.

* 2022.06.08 : upload training and evaluation code and pretrained weight on HRSC2016. (details see Getting Started)

## Model Zoo
* Pretrained weights on DOTA

<table>
	<tr>
	    <th>Model</th>
	    <th>Backbone</th>
	    <th>MS Test</th>  
      <th>Training size</th>
      <th>Training scales</th>
      <th>mAP</th>
      <th>Download</th>
	</tr >
	<tr >
	    <td rowspan="2">GF-CSL</td>
	    <td rowspan="2">ResNet50</td>
	    <td>??</td>
      <td rowspan="2">1024??1024</td>
      <td rowspan="2">[0.5,1.0]</td>
      <td>75.61%</td>
      <td rowspan="2">https://drive.google.com/file/d/17Z-0i-ifP_fY58CfoBr8LGBsfLLklm1l/view?usp=sharing</td>
	</tr>
	<tr>
	    <td>???</td>
      <td>77.54%</td>
	</tr>
  <tr >
	    <td rowspan="2">GF-CSL</td>
	    <td rowspan="2">ResNet101</td>
	    <td>??</td>
      <td rowspan="2">1024??1024</td>
      <td rowspan="2">[0.5,1.0]</td>
      <td>75.52%</td>
      <td rowspan="2">https://drive.google.com/file/d/1NU5ypyioIIpqCFBLT_87eT-_7K-gYzzS/view?usp=sharing</td>
	</tr>
	<tr>
	    <td>???</td>
      <td>78.34%</td>
	</tr>
  <tr >
	    <td rowspan="2">GF-CSL</td>
	    <td rowspan="2">ResNet152</td>
	    <td>??</td>
      <td rowspan="2">1024??1024</td>
      <td rowspan="2">[0.5,1.0]</td>
      <td>76.35%</td>
      <td rowspan="2">https://drive.google.com/file/d/1GgHAI57HFkhw_an3ONGt9Syttfrjg683/view?usp=sharing</td>
	</tr>
	<tr>
	    <td>???</td>
      <td>78.12%</td>
	</tr>
  <tr >
	    <td rowspan="2">GF-CSL</td>
	    <td rowspan="2">ResNet101</td>
	    <td>??</td>
      <td rowspan="2">1024??1024</td>
      <td rowspan="2">[0.5,1.0,1.5]</td>
      <td>76.05%</td>
      <td rowspan="2">https://drive.google.com/file/d/1eAz5l-M4IqycL9mW2zegwN6wzVMIdJgM/view?usp=sharing</td>
	</tr>
	<tr>
	    <td>???</td>
      <td>79.94%</td>
	</tr>
</table>

* Pretrained weights on HRSC2016
<table>
	<tr>
	    <th>Model</th>
	    <th>Backbone</th>
	    <th>MS Test</th>  
      <th>Training size</th>
      <th>mAP12</th>
      <th>Download</th>
	</tr >
	<tr >
	    <td rowspan="2">GF-CSL</td>
	    <td rowspan="2">ResNet50</td>
	    <td>??</td>
      <td rowspan="2">640??640</td>
      <td>97.00%</td>
      <td rowspan="2">https://drive.google.com/file/d/1Nzwp7OHFn2LHVMyQnd11D2i7fe_0ASP4/view?usp=sharing</td>
	</tr>
	<tr>
	    <td>???</td>
      <td>97.94%</td>
	</tr>
</table>

Note that the performance on HRSC2016 is a little bit higher than paper, as we add angle-branch warm up technique during training. (details see Getting Started) 

<!---
Model | Backbone | MS | Training size | Training scales | mAP | Download 
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
GF-CSL | ResNet50 | ??? | 1024??1024 | [0.5, 1.0] | 77.54% | [model](https://drive.google.com/file/d/17Z-0i-ifP_fY58CfoBr8LGBsfLLklm1l/view?usp=sharing)
GF-CSL | ResNet101 | ??? | 1024??1024 | [0.5, 1.0] | 78.34% | [model](https://drive.google.com/file/d/1NU5ypyioIIpqCFBLT_87eT-_7K-gYzzS/view?usp=sharing)
GF-CSL | ResNet152 | ??? | 1024??1024 | [0.5, 1.0] | 78.12% | [model](https://drive.google.com/file/d/1GgHAI57HFkhw_an3ONGt9Syttfrjg683/view?usp=sharing)
GF-CSL | ResNet101 | ??? | 1024??1024 | [0.5, 1.0, 1.5] | 79.94% | [model](https://drive.google.com/file/d/1eAz5l-M4IqycL9mW2zegwN6wzVMIdJgM/view?usp=sharing)


* Pretrained weights on HRSC2016 and UCAS-AOD

Dataset | Backbone | MS | Training size | mAP07 | mAP12 | Download 
------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
HRSC2016 | ResNet50 | ??? | 640??640 | 90.33% | 97.38% |
HRSC2016 | ResNet101 | ??? | 800??800 | 90.53% | 97.90% |
UCAS-AOD | ResNet50 | ??? | 640??640 | 89.61% | 96.42% |
UCAS-AOD | ResNet101 | ??? | 800??800 | 89.51% | 96.51% |
-->
## Results visualization
Predicted angular distribution
![6fbfaeccde7c1ac2d1020b511b70bdf](https://user-images.githubusercontent.com/56680663/169776865-0e616e4c-c1b2-4659-abce-bc81229ac60b.png)


## Citation
```
@article{wang2022gaussian,
  title={Gaussian Focal Loss: Learning Distribution Polarized Angle Prediction for Rotated Object Detection in Aerial Images},
  author={Wang, Jian and Li, Fan and Bi, Haixia},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  publisher={IEEE}
}
```
