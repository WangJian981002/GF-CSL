# Getting start 
## Train
Our training code is based on DistributeDataParallel for multiple GPUs. However, it can be easily extended to a single GPU training by specifying the number of training cards to be 1.
### Train on DOTA dataset 
For example, using 2 GPUs to training a ResNet50 model:
```
CUDA_VISIBLE_DEVICES=0,1 python train_dota.py --datadir ./DOTA_devkit/datasets/trainvalsplit-1024-256 --model 50 
```

## Visualize
For example, using pretrained ResNet50 model:
```
python evaluate.py --operation visualize \
    --model 50 --weight_path ./checkpoint/r50-scale=[0.5.1.0].pth \
    --img_path .\result\P0007.png
```

Optional arguments:
* --score_thr :object confidence during detection. score greater than the confidence is considered to be a detected object.

The visualization file appears in the current path as demo.jpg.

## Test
The following describes how to generate test results for uploading to the DOTA server, which contains 15 txt files (v1.0), each of which is all the test results for each category.

updating...
