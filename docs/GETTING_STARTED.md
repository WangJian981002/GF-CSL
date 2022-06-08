# Getting start 
## Train
Our training code is based on DistributeDataParallel for multiple GPUs. However, it can be easily extended to a single GPU training by specifying the number of training cards to be 1.
### Train on DOTA dataset 
For example, using 2 GPUs to training a ResNet50 model:
```
CUDA_VISIBLE_DEVICES=0,1 python train_dota.py --datadir ./DOTA_devkit/datasets/DOTA/trainvalsplit-1024-256 --model 50 
```
### Train on HRSC2016 and UCAS-AOD datasets 
For example, using 2 GPUs to training a ResNet50 model:
```
#HRSC2016
CUDA_VISIBLE_DEVICES=0,1 python train_hrsc.py --datadir ./DOTA_devkit/datasets/HRSC2016/train --model 50 --input_size 640
#UCAS-AOD
CUDA_VISIBLE_DEVICES=0,1 python train_ucas.py --datadir ./DOTA_devkit/datasets/UCAS_AOD/train --model 50 --input_size 640
```
NOTE: For training the HRSC2016 and UCAS-AOD datasets: To make the angle prediction branch converge faster, we train only the angle branch for the first 10 epochs, which is achieved by resetting the other task loss weights to 0. After 10 epochs the training process returned to normal.

## Visualize
For example, using pretrained ResNet50 model:
```
python evaluate.py --operation visualize \
    --model 50 --weight_path ./checkpoint/r50-scale=[0.5.1.0].pth \
    --img_path ./result/P0007.png
```

Optional arguments:
* --score_thr :object confidence during detection. score greater than the confidence is considered to be a detected object.

The visualization file appears in the current path as demo.jpg.

## Test on DOTA
The following describes how to generate test results for uploading to the DOTA server, which contains 15 txt files (v1.0), each of which is all the test results for each category.
### Single scale testing
The test images are cropped into 4000×4000 pixels, therefore, the resolution of the images sent to the network is 4000 by default. If the `--test_size` parameter is specified, the image will be scaled to that size for testing. (Note that the `--test_size` must be divisible by 32 due to the FPN structure built inside the network.)

For example:
```
python evaluate.py --operation DOTA_test \
    --model 50 --weight_path ./checkpoint/r50-scale=[0.5.1.0].pth \
    --test_size 4000 --output_id 0
```

Note that the testing process may be slower due to our testing on a single GPU at larger resolutions.

The results files will appear at "./result/test0/test0_final", which can be subsequently sent to the [DOTA server](http://bed4rs.net:8001/login/) to obtain the evaluation results.

### Multiscale scale testing
Take ResNet50 for example:
```
python evaluate.py --operation DOTA_MS_test \
    --model 50 --weight_path ./checkpoint/r50-scale=[0.5.1.0].pth \
    --test_image_dir ./DOTA_devkit/datasets/DOTA/test4000/images
```
The results files will appear at "./result/test/test_final"

The default multiscale image sizes include [2016, 2528, 3008, 3520, 4000, 4512, 5024, 5504, 6016] by default. The above resolutions are 0.5-1.5 times the size of the original test image. 

## Test on HRSC2016
single scale test for example:
```
python evaluate.py --operation HRSC_test \
    --model 50 --weight_path ${WEIGHT_PATH} \
    --hrsc_test_size 640 
```
`--hrsc_test_size ` specifies the long side resolution of images during testing, which must be divisible by 32.

`--use_07_metric ` action='store_true' whether use VOC07 metric.

multi scale test for example:
```
python evaluate.py --operation HRSC_MS_test \
    --model 50 --weight_path ${WEIGHT_PATH} 
```
The default long side resolutions are 512, 544, 576, 608, 640.
