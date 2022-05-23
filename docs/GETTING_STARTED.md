# Getting start 
## Train
Our training code is based on DistributeDataParallel for multiple GPUs. However, it can be easily extended to a single GPU training by specifying the number of training cards to be 1.
### Train on DOTA dataset 
For example, using 2 GPUs to training a ResNet50 model:
```
CUDA_VISIBLE_DEVICES=0,1 python train_dota.py --datadir ./DOTA_devkit/datasets/trainvalsplit-1024-256 --model 50 
```

## Visualize
