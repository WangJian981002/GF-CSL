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
