# Installation
## Step-by-step Installation
### Requirements
* Linux
* Python 3.7+
* Pytorch 1.7.0 or higher
* mmcv
* CUDA 11.0
* GCC 7.5.0

####  INSTALL
1. Create a conda virtual environment and activate it

```
conda create -n GF python=3.7 -y
conda activate GF
```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.

`conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch`

3. Install [mmcv](https://github.com/open-mmlab/mmcv) for DCNv2, e.g.

```
#pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu110/torch1.7.1/index.html
```

4. Install DOTA_devkit

```
sudo apt-get install swig
cd DOTA_devkit
swig -c++ -python polyiou.i
python setup.py build_ext --inplace
```
#### DATASETS PREPARATION
For DOTA datasets, please refer [DOTA_devkit](https://github.com/CAPTAIN-WHU/DOTA_devkit) to crop the original images into patches. e.g. 1024×1024 pixels with overlap 256 px.

Please organize the datasets in the following format. Note that the test set of DOTA does not provide annotations, so you can place the corresponding empty files in the test_split/labelTxt path.

As described in the paper, we use a relatively large image resolution during the test, please crop the test image into a 4000×4000 px with overlap 2000 px.
 
```
GF-CSL
├── DOTA_devkit
│   ├── datasets
│   │   ├── DOTA
│   │   │   ├── trainvalsplit-1024-256
│   │   │   │   ├── images
│   │   │   │   ├── labelTxt
│   │   │   ├── trainvalsplit-multiscale
│   │   │   │   ├── images
│   │   │   │   ├── labelTxt
│   │   │   │── test4000
│   │   │   │   ├── images
│   │   │   │   ├── labelTxt
│   │   │── HRSC2016
│   │   │   │── train
│   │   │   │   ├── images
│   │   │   │   ├── labelTxt
│   │   │   │── test
│   │   │   │   ├── images
│   │   │   │   ├── labelTxt
│   │   │── UCAS_AOD
│   │   │   │── train
│   │   │   │   ├── images
│   │   │   │   ├── labelTxt
│   │   │   │── test
│   │   │   │   ├── images
│   │   │   │   ├── labelTxt
```

for each annotation file (.txt), each line represent an object following:

x1, y1, x2, y2, x3, y3, x4, y4, class, difficult

```
e.g.:
2753 2408 2861 2385 2888 2468 2805 2502 plane 0
3445 3391 3484 3409 3478 3422 3437 3402 large-vehicle 0
3185 4158 3195 4161 3175 4204 3164 4199 large-vehicle 0
2870 4250 2916 4268 2912 4283 2866 4263 large-vehicle 0
630 1674 628 1666 640 1654 644 1666 small-vehicle 0
636 1713 633 1706 646 1698 650 1706 small-vehicle 0
717 76 726 78 722 95 714 90 small-vehicle 0
737 82 744 84 739 101 731 98 small-vehicle 0
...
```
