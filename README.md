# ImageNet Training
Pytorch code for training imagenet with fp16

1. Install pytorch,torchvision
2. Install apex
```
conda install -c conda-forge nvidia-apex
```
3. (optional) install data loading speedups:
```
conda install -c thomasbrandon -c defaults -c conda-forge pillow-accel-avx2
conda install -c conda-forge libjpeg-turbo
```

usage example:
```
CUDA_VISIBLE_DEVICES=1,0 python3 train_imagenet.py -t /imagenet_data/train/ -v  /imagenet_data/val/ -a xception -b 256 -j=6 -n=my_experiment_name --lr=.1 -fp16
```
